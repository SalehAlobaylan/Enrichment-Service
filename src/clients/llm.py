from src.config import Settings
from src.middleware.error_handler import LLMError
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    def __init__(self, settings: Settings):
        self.provider = (settings.LLM_PROVIDER or "").strip().lower()
        self.model = settings.LLM_MODEL
        self._openai_client = None
        self._anthropic_client = None
        self._gemini_client = None

        if self.provider == "openai" and settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI

            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        elif self.provider == "anthropic" and settings.ANTHROPIC_API_KEY:
            from anthropic import AsyncAnthropic

            self._anthropic_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif self.provider == "gemini" and settings.GEMINI_API_KEY:
            # google-genai is Google's current SDK for Gemini 2.x and 3.x.
            # The client itself is sync-only; we call it via asyncio.to_thread
            # in _gemini_complete so we don't block the event loop.
            from google import genai

            self._gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)

    async def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        try:
            if self.provider == "openai" and self._openai_client:
                return await self._openai_complete(
                    system_prompt, user_prompt, max_tokens, temperature
                )
            elif self.provider == "anthropic" and self._anthropic_client:
                return await self._anthropic_complete(
                    system_prompt, user_prompt, max_tokens, temperature
                )
            elif self.provider == "gemini" and self._gemini_client:
                return await self._gemini_complete(
                    system_prompt, user_prompt, max_tokens, temperature
                )
            else:
                raise LLMError(
                    f"LLM provider '{self.provider}' is not configured. "
                    "Set the appropriate API key."
                )
        except LLMError:
            raise
        except Exception as exc:
            logger.error("llm_request_failed", provider=self.provider, error=str(exc))
            raise LLMError(f"LLM request failed: {exc}") from exc

    async def _openai_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        assert self._openai_client is not None
        response = await self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMError("LLM returned empty response")
        return content

    async def _anthropic_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        assert self._anthropic_client is not None
        response = await self._anthropic_client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not response.content:
            raise LLMError("LLM returned empty response")
        return response.content[0].text

    async def _gemini_complete(
        self, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float
    ) -> str:
        assert self._gemini_client is not None
        import asyncio

        from google.genai import types

        # google-genai uses a single `contents` argument; the system prompt
        # is supplied via system_instruction in the generation config.
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        # The SDK is synchronous — offload to a thread so the FastAPI event
        # loop stays responsive under concurrent translate/summarize calls.
        response = await asyncio.to_thread(
            self._gemini_client.models.generate_content,
            model=self.model,
            contents=user_prompt,
            config=config,
        )

        text = getattr(response, "text", None)
        if not text:
            raise LLMError("LLM returned empty response")
        return text
