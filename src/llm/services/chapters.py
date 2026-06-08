import json

from src.common.utils.logging import get_logger
from src.llm.clients.llm import LLMClient
from src.llm.schemas.chapters import (
    ChaptersGenerateRequest,
    ChaptersGenerateResponse,
    GeneratedChapter,
)

logger = get_logger(__name__)

# Bound the prompt: cap window text + total windows. CMS controls window
# coarseness (window size), so this is just a safety net for huge transcripts.
MAX_WINDOW_CHARS = 220
MAX_WINDOWS = 600
MAX_TITLE_CHARS = 120
MAX_SUMMARY_CHARS = 280


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return text.strip()


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}" if h else f"{m:d}:{sec:02d}"


SYSTEM_PROMPT = """
You are a video editor creating chapter markers for a long audio/video transcript.
You are given the transcript split into numbered time WINDOWS. Group consecutive
windows into coherent CHAPTERS by topic — a new chapter starts when the subject
clearly shifts. Chapters should be VARIABLE length: a quick intro might be one
window, a deep discussion many. Do NOT force equal lengths.

For each chapter return:
- "start_index": the window index where the chapter STARTS (an integer that
  exists in the input; the first chapter MUST start at the smallest index).
- "title": a concise, specific title (3-8 words) in the SAME language as the
  transcript (Arabic transcript => Arabic title).
- "summary": one short sentence describing the chapter (same language).

Hard rules:
- start_index values must be strictly increasing and must be real window indices.
- Never invent timestamps — only pick window indices.
- No hashtags, no surrounding quotes.

Return ONLY a JSON object of this shape:
{"chapters": [{"start_index": 0, "title": "...", "summary": "..."}, ...]}
""".strip()


class ChaptersGenerationService:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def _build_instructions(self, req: ChaptersGenerateRequest) -> str:
        parts: list[str] = []
        if req.mode == "count" and req.target_count:
            parts.append(f"Create approximately {req.target_count} chapters.")
        elif req.mode == "duration" and req.target_duration_sec:
            mins = max(1, round(req.target_duration_sec / 60))
            parts.append(f"Aim for chapters about {mins} minute(s) long each.")
        else:
            parts.append("Create as many chapters as the content naturally has.")
        if req.min_sec:
            parts.append(f"No chapter shorter than ~{max(1, round(req.min_sec / 60))} minute(s).")
        if req.max_sec:
            parts.append(f"No chapter longer than ~{max(1, round(req.max_sec / 60))} minute(s).")
        if not req.with_summary:
            parts.append('Omit summaries (use empty string for "summary").')
        if req.language:
            parts.append(f"Write titles and summaries in this language: {req.language}.")
        return " ".join(parts)

    async def generate(self, req: ChaptersGenerateRequest) -> ChaptersGenerateResponse:
        windows = req.windows[:MAX_WINDOWS]
        valid_indices = {w.index for w in windows}
        first_index = min(valid_indices)

        lines = [
            f"[{w.index}] ({_fmt_ts(w.start_sec)}) {w.text.strip()[:MAX_WINDOW_CHARS]}"
            for w in windows
        ]
        user_prompt = (
            f"{self._build_instructions(req)}\n\n"
            f"Transcript windows:\n\n" + "\n".join(lines)
        )

        raw = await self.llm.complete(
            SYSTEM_PROMPT,
            user_prompt,
            max_tokens=2048,
            temperature=0.3,
            operation="chapters_generate",
        )

        chapters = self._parse(raw, valid_indices, first_index, req.with_summary)
        return ChaptersGenerateResponse(chapters=chapters)

    def _parse(
        self,
        raw: str,
        valid_indices: set[int],
        first_index: int,
        with_summary: bool,
    ) -> list[GeneratedChapter]:
        try:
            parsed = json.loads(_strip_fences(raw))
            items = parsed.get("chapters", []) if isinstance(parsed, dict) else []
        except Exception:
            logger.warning("chapters_parse_failed", raw_preview=raw[:200])
            items = []

        cleaned: list[GeneratedChapter] = []
        seen: set[int] = set()
        for it in items:
            if not isinstance(it, dict):
                continue
            try:
                idx = int(it.get("start_index"))
            except (TypeError, ValueError):
                continue
            title = str(it.get("title", "")).strip()[:MAX_TITLE_CHARS]
            if idx not in valid_indices or idx in seen or not title:
                continue
            summary = None
            if with_summary:
                summary = str(it.get("summary", "")).strip()[:MAX_SUMMARY_CHARS] or None
            cleaned.append(GeneratedChapter(start_index=idx, title=title, summary=summary))
            seen.add(idx)

        cleaned.sort(key=lambda c: c.start_index)

        # Always cover the start: force the first chapter to begin at index 0.
        if cleaned:
            if cleaned[0].start_index != first_index:
                cleaned[0].start_index = first_index
        return cleaned
