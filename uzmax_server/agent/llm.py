import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


def configured_secret(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.upper()
    if any(marker in upper for marker in ("YOUR_", "YOUR-", "YOUR", "_HERE", "KEY_HERE")):
        return None
    return value


class OpenAIClient:
    """OpenAI-only LLM client for the UzMAX chatbot."""

    def __init__(self, api_key: str = None, model: str = None):
        openai_key = configured_secret(api_key or os.getenv("OPENAI_API_KEY"))
        self.client = AsyncOpenAI(api_key=openai_key) if openai_key else None
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    async def _stream_text(self, messages: list):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        async for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

    async def get_response_stream(self, messages: list, tools=None, tool_executor=None,
                                  max_tool_rounds: int = 4):
        """Stream the assistant reply.

        When ``tools`` and ``tool_executor`` are provided, the model decides when to
        call them (OpenAI function calling). Tool plumbing (assistant tool_call
        messages and tool results) stays inside this method on a working copy, so the
        caller's ``messages`` history only ever receives the final spoken text.
        ``tool_executor(name, args_dict)`` may be sync or async and returns a
        JSON-serializable result.
        """
        if self.client is None:
            yield "OpenAI API kaliti sozlanmagan. Iltimos, .env faylga haqiqiy OPENAI_API_KEY kiriting."
            return

        if not tools or tool_executor is None:
            async for content in self._stream_text(messages):
                yield content
            return

        working = list(messages)
        for _ in range(max_tool_rounds):
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=working,
                tools=tools,
                tool_choice="auto",
                stream=True,
            )
            content_acc = ""
            tool_acc: dict[int, dict] = {}   # index -> {id, name, args}
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if getattr(delta, "content", None):
                    content_acc += delta.content
                    yield delta.content        # stream tokens to the caller (TTS) immediately
                for tc in (getattr(delta, "tool_calls", None) or []):
                    slot = tool_acc.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            slot["name"] += tc.function.name
                        if tc.function.arguments:
                            slot["args"] += tc.function.arguments

            if not tool_acc:
                return   # plain answer already streamed token-by-token

            ordered = [tool_acc[i] for i in sorted(tool_acc)]
            working.append({
                "role": "assistant",
                "content": content_acc or None,
                "tool_calls": [
                    {
                        "id": s["id"],
                        "type": "function",
                        "function": {"name": s["name"], "arguments": s["args"] or "{}"},
                    }
                    for s in ordered
                ],
            })
            for s in ordered:
                try:
                    args = json.loads(s["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = tool_executor(s["name"], args)
                    if hasattr(result, "__await__"):
                        result = await result
                except Exception as exc:  # surface tool failure to the model, don't crash
                    result = {"error": str(exc)}
                working.append({
                    "role": "tool",
                    "tool_call_id": s["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # Tool rounds exhausted — produce a final answer without further tool calls.
        async for content in self._stream_text(working):
            yield content

    async def transcribe(self, wav_bytes: bytes, language: str | None = None,
                         prompt: str | None = None, model: str | None = None) -> str:
        """Transcribe a WAV clip via OpenAI (Whisper). Non-streaming: the whole
        utterance is sent at once and the text returned. Used as the STT engine
        because OpenAI is reachable/fast here and handles Uzbek well.

        Note: OpenAI rejects ``language='uz'`` (Uzbek is not in the supported set),
        so for Uzbek pass language=None to auto-detect and use ``prompt`` to bias it.
        """
        if self.client is None:
            return ""
        model = model or os.getenv("STT_MODEL", "whisper-1")
        # Per-request timeout: OpenAI transcription occasionally stalls (seen ~38s);
        # the SDK default is far too long and would freeze the turn. Bound it here.
        request_timeout = float(os.getenv("STT_REQUEST_TIMEOUT", "12"))
        kwargs = {"model": model, "file": ("audio.wav", wav_bytes, "audio/wav"),
                  "timeout": request_timeout}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        try:
            resp = await self.client.audio.transcriptions.create(**kwargs)
        except Exception as exc:
            # Retry without language if the model rejects the code (e.g. 'uz').
            if "language" in kwargs and "language" in str(exc).lower():
                kwargs.pop("language", None)
                resp = await self.client.audio.transcriptions.create(**kwargs)
            else:
                raise
        return (getattr(resp, "text", "") or "").strip()

    async def extract_person_name(self, text: str) -> dict | None:
        if self.client is None:
            return None

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the person's name from the phrase. A single first name is enough. "
                        "Return only JSON in this exact shape: "
                        '{"first_name":"...","last_name":"...","is_confident":true}. '
                        "If there is no surname or last name, return an empty string for last_name. "
                        "If you are not confident, return is_confident=false. "
                        "Do not add markdown."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        content = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            data = json.loads(content)
        except Exception:
            return None
        if not data.get("first_name"):
            return None
        return data
