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
- "end_index": the last window index INCLUDED in the chapter.
- "title": a concise, specific title (3-8 words) in the SAME language as the
  transcript (Arabic transcript => Arabic title).
- "summary": one short sentence describing the chapter (same language).
- "context_label": 1-4 words naming the main context/topic.
- "confidence": 0..1 confidence that the cut is coherent and feed-ready.
- "boundary_reason": why this start/end is a good contextual boundary.
- "standalone_score": 0..1 score for whether this chapter completes its idea.
- "contains_sponsor_or_intro": true if the chapter includes sponsor/intro/outro.
- "needs_review_reason": short string only when a human should review.

Hard rules:
- start_index values must be strictly increasing and must be real window indices.
- end_index values must be real window indices and >= start_index.
- Never invent timestamps — only pick window indices.
- Prefer 5-30 minute coherent chapters; allow up to 40 minutes only if the same
  context clearly continues. Short coherent episodes can be one chapter.
- Do not cut in the middle of an argument, story, answer, or topic transition.
- No hashtags, no surrounding quotes.

Return ONLY a JSON object of this shape:
{"chapters": [{"start_index": 0, "end_index": 12, "title": "...", "summary": "...", "context_label": "...", "confidence": 0.9, "boundary_reason": "...", "standalone_score": 0.9, "contains_sponsor_or_intro": false, "needs_review_reason": null}, ...]}
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
        sorted_indices = sorted(valid_indices)
        last_index = sorted_indices[-1]
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
            end_index = it.get("end_index")
            try:
                end_idx = int(end_index) if end_index is not None else None
            except (TypeError, ValueError):
                end_idx = None
            if end_idx is not None and (end_idx not in valid_indices or end_idx < idx):
                end_idx = None
            summary = None
            if with_summary:
                summary = str(it.get("summary", "")).strip()[:MAX_SUMMARY_CHARS] or None
            confidence = _clamp01(it.get("confidence"), 0.72)
            standalone_score = _clamp01(it.get("standalone_score"), confidence)
            needs_review_reason = str(it.get("needs_review_reason") or "").strip() or None
            cleaned.append(
                GeneratedChapter(
                    start_index=idx,
                    end_index=end_idx,
                    title=title,
                    summary=summary,
                    context_label=str(it.get("context_label") or "").strip()[:80] or None,
                    confidence=confidence,
                    boundary_reason=str(it.get("boundary_reason") or "").strip()[:240] or None,
                    standalone_score=standalone_score,
                    contains_sponsor_or_intro=bool(it.get("contains_sponsor_or_intro") or False),
                    needs_review_reason=needs_review_reason[:240] if needs_review_reason else None,
                )
            )
            seen.add(idx)

        cleaned.sort(key=lambda c: c.start_index)

        # Always cover the start: force the first chapter to begin at index 0.
        if cleaned:
            if cleaned[0].start_index != first_index:
                cleaned[0].start_index = first_index
            for i, chapter in enumerate(cleaned):
                if chapter.end_index is None:
                    next_start = cleaned[i + 1].start_index if i + 1 < len(cleaned) else None
                    if next_start is None:
                        chapter.end_index = last_index
                    else:
                        prev_indices = [idx for idx in sorted_indices if idx < next_start]
                        chapter.end_index = prev_indices[-1] if prev_indices else chapter.start_index
        return cleaned


def _clamp01(value: object, fallback: float) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, numeric))
