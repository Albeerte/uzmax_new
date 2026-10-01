import json
import logging
import os
import asyncio
import re
import requests

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

logger = logging.getLogger("uzmax.llm")

try:
    from google import genai
    _HAS_GEMINI = True
except ImportError:
    genai = None
    _HAS_GEMINI = False


def configured_secret(value: str | None) -> str | None:
    if not value:
        return None
    upper = value.upper()
    if any(marker in upper for marker in ("YOUR_", "YOUR-", "YOUR", "_HERE", "KEY_HERE")):
        return None
    return value


def safe_error(exc: Exception) -> str:
    text = str(exc)
    text = re.sub(r"(api key:\s*)\S+", r"\1***", text, flags=re.IGNORECASE)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9._=\-]+", r"\1***", text, flags=re.IGNORECASE)
    text = re.sub(r"(Api-Key\s+)[A-Za-z0-9._=\-]+", r"\1***", text, flags=re.IGNORECASE)
    return text


class OpenAIClient:
    """Multi-provider Cloud LLM Client (DeepSeek/OpenAI chat + Google Gemini fallback)."""

    def __init__(self, api_key: str = None, model: str = None):
        deepseek_key = configured_secret(os.getenv("DEEPSEEK_API_KEY"))
        openai_key = configured_secret(api_key or os.getenv("OPENAI_API_KEY"))
        self.client = None
        self.transcription_client = None
        self.yandex_api_key = configured_secret(os.getenv("YANDEX_GPT_API_KEY") or os.getenv("YANDEX_API_KEY"))
        self.yandex_folder_id = configured_secret(os.getenv("YANDEX_GPT_FOLDER_ID") or os.getenv("YANDEX_CATALOG_ID"))
        self.yandex_model = os.getenv("YANDEX_GPT_MODEL", "yandexgpt")
        self.yandex_base_url = os.getenv("YANDEX_GPT_BASE_URL", "https://llm.api.cloud.yandex.net")
        self.yandex_available = bool(self.yandex_api_key and self.yandex_folder_id)
        if deepseek_key:
            try:
                self.client = AsyncOpenAI(
                    api_key=deepseek_key,
                    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                )
                self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
                self.provider = "deepseek"
            except Exception as exc:
                logger.warning("[LLM] DeepSeek client init failed: %s", safe_error(exc))
                self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self.provider = "none"
        elif openai_key:
            try:
                self.client = AsyncOpenAI(api_key=openai_key)
                self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self.provider = "openai"
            except Exception as exc:
                logger.warning("[LLM] OpenAI client init failed (%s). Run 'py -m pip install \"httpx<0.28.0\" --upgrade openai'", safe_error(exc))
                self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                self.provider = "none"
        else:
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.provider = "none"

        if openai_key:
            try:
                self.transcription_client = AsyncOpenAI(api_key=openai_key)
            except Exception as exc:
                logger.warning("[LLM] OpenAI transcription client init failed: %s", safe_error(exc))

        # Gemini Client Fallback
        gemini_key = configured_secret(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        if _HAS_GEMINI and gemini_key:
            try:
                self.gemini_client = genai.Client(api_key=gemini_key)
                self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            except Exception as exc:
                logger.warning("[LLM] Gemini client init failed: %s", safe_error(exc))
                self.gemini_client = None
        else:
            self.gemini_client = None

    def _yandex_model_uri(self) -> str:
        if self.yandex_model.startswith("gpt://"):
            return self.yandex_model
        model = self.yandex_model.strip("/") or "yandexgpt"
        if "/" in model:
            return f"gpt://{self.yandex_folder_id}/{model}"
        return f"gpt://{self.yandex_folder_id}/{model}/latest"

    @staticmethod
    def _message_text(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text") or item.get("content") or ""))
                else:
                    parts.append(str(item))
            return "\n".join(part for part in parts if part).strip()
        return str(content or "")

    async def _stream_yandex_text(self, messages: list):
        if not self.yandex_available:
            return

        yc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            if role not in {"system", "user", "assistant"}:
                continue
            text = self._message_text(msg.get("content"))
            if text:
                yc_messages.append({"role": role, "text": text})

        if not yc_messages:
            return

        payload = {
            "modelUri": self._yandex_model_uri(),
            "completionOptions": {
                "stream": False,
                "temperature": float(os.getenv("YANDEX_GPT_TEMPERATURE", "0.25")),
                "maxTokens": str(int(os.getenv("YANDEX_GPT_MAX_TOKENS", "800"))),
            },
            "messages": yc_messages,
        }
        headers = {
            "Authorization": f"Api-Key {self.yandex_api_key}",
            "x-folder-id": self.yandex_folder_id,
            "Content-Type": "application/json",
        }
        url = self.yandex_base_url.rstrip("/") + "/foundationModels/v1/completion"

        def request_completion():
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            alternatives = ((data.get("result") or {}).get("alternatives") or [])
            if not alternatives:
                return ""
            return ((alternatives[0].get("message") or {}).get("text") or "").strip()

        text = await asyncio.to_thread(request_completion)
        if text:
            yield text

    async def _stream_fallback_text(self, messages: list):
        if self.gemini_client:
            prompt_parts = []
            for m in messages:
                prompt_parts.append(f"{m['role'].upper()}: {m['content']}")
            prompt = "\n\n".join(prompt_parts)

            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                None,
                lambda: self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                )
            )
            if res and res.text:
                yield res.text
                return

        if self.yandex_available:
            async for content in self._stream_yandex_text(messages):
                yield content

    async def _stream_text(self, messages: list):
        if self.client:
            try:
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
                return
            except Exception as exc:
                logger.warning("[LLM] %s request failed, trying fallback: %s", self.provider, safe_error(exc))

        async for content in self._stream_fallback_text(messages):
            yield content

    async def get_response_stream(self, messages: list, tools=None, tool_executor=None,
                                  max_tool_rounds: int = 4):
        """Stream assistant reply with tool calling & fallback support."""
        if self.client is None and self.gemini_client is None and not self.yandex_available:
            yield "LLM API kaliti sozlanmagan. Iltimos, .env fayliga DEEPSEEK_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY yoki YANDEX_API_KEY kiriting."
            return

        if self.client is None or not tools or tool_executor is None:
            async for content in self._stream_text(messages):
                yield content
            return

        working = list(messages)
        for _ in range(max_tool_rounds):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=working,
                    tools=tools,
                    tool_choice="auto",
                )
            except Exception as exc:
                logger.warning("[LLM] tool-capable provider failed, using text fallback: %s", safe_error(exc))
                async for content in self._stream_fallback_text(messages):
                    yield content
                return
            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

            if not tool_calls:
                if msg.content:
                    yield msg.content
                return

            working.append(msg)
            for call in tool_calls:
                fn_name = call.function.name
                try:
                    args_dict = json.loads(call.function.arguments or "{}")
                except Exception:
                    args_dict = {}

                if asyncio.iscoroutinefunction(tool_executor):
                    result = await tool_executor(fn_name, args_dict)
                else:
                    result = tool_executor(fn_name, args_dict)

                result_str = json.dumps(result, ensure_ascii=False)
                working.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result_str,
                })

        async for content in self._stream_text(working):
            yield content

    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "speech.wav",
        language: str | None = None,
        prompt: str | None = None,
        request_timeout: float = 12.0,
    ) -> str:
        """Transcribe audio using OpenAI Whisper API."""
        if self.transcription_client is None:
            return ""

        file_obj = (filename, audio_bytes, "audio/wav")
        kwargs = {"model": "whisper-1", "file": file_obj, "timeout": request_timeout}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        try:
            resp = await self.transcription_client.audio.transcriptions.create(**kwargs)
        except Exception as exc:
            if "language" in kwargs and "language" in str(exc).lower():
                kwargs.pop("language", None)
                resp = await self.transcription_client.audio.transcriptions.create(**kwargs)
            else:
                raise
        return (getattr(resp, "text", "") or "").strip()

    async def extract_person_name(self, text: str) -> dict | None:
        if self.client is None and self.gemini_client is None:
            return None

        if self.client:
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
        else:
            prompt = (
                "Extract the person's name from the phrase. A single first name is enough. "
                "Return only JSON in this exact shape: "
                '{"first_name":"...","last_name":"...","is_confident":true}.\n'
                f"Phrase: {text}"
            )
            loop = asyncio.get_event_loop()
            res = await loop.run_in_executor(
                None,
                lambda: self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                )
            )
            content = res.text if res else "{}"

        content = content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            data = json.loads(content)
        except Exception:
            return None
        if not data.get("first_name"):
            return None
        return data
