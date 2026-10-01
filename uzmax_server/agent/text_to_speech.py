
import numpy as np
import sounddevice as sd
import base64
import requests
from io import BytesIO

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

import io, os, re, time, wave, json, hashlib, asyncio, queue as _queue, grpc, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Tuple

try:
    import yandex.cloud.ai.tts.v3.tts_pb2 as tts_pb2
    import yandex.cloud.ai.tts.v3.tts_service_pb2_grpc as tts_grpc
except Exception as _exc:
    tts_pb2 = None
    tts_grpc = None


class YandexTTS:
    """
    Yandex SpeechKit v1 TTS на LPCM + конверсия в WAV
    """
    def __init__(self,
                 folder_id: str,
                 iam_token: str,
                 voice: str = "filipp",
                 emotion: str = "neutral",
                 lang: str = "ru-RU",
                 sample_rate: int = 48000):
        self.url = 'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize'
        self.headers = {'Authorization': 'Api-Key ' + iam_token}
        self.params = {
            'folderId': folder_id,
            'lang': lang,
            'voice': voice,
            'emotion': emotion,
            'format': 'lpcm',
            'sampleRateHertz': sample_rate,
        }
        self.sample_rate = sample_rate

    def synthesize(self, text: str) -> bytes:
        """
        Запрашивает LPCM, оборачивает в WAV и возвращает готовый WAV-байтстрим.
        """
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            print(f"[TTS] Попытка {attempt}/{max_retries}...")
            try:
                resp = requests.post(
                    self.url,
                    headers=self.headers,
                    data={**self.params, 'text': text},
                    stream=True,
                    timeout=10
                )
                if resp.status_code != 200:
                    print(f"[TTS] Ошибка {resp.status_code}: {resp.text}")
                resp.raise_for_status()

                # Собираем все PCM-чанки
                pcm = b''.join(resp.iter_content(chunk_size=1024))
                print(f"[TTS] Успех: получили {len(pcm)} байт LPCM, конвертирую в WAV…")

                # Конверсия в WAV
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(self.sample_rate)
                    wf.writeframes(pcm)
                wav_bytes = buf.getvalue()
                print(f"[TTS] Конверсия завершена, итог {len(wav_bytes)} байт WAV")
                return wav_bytes

            except Exception as e:
                print(f"[TTS] Ошибка на попытке {attempt}: {e}")
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    raise RuntimeError("TTS synthesis failed after 3 attempts") from e

    def save_audio(self, wav_bytes: bytes, path: str) -> None:
        """
        Сохраняет готовый WAV в файл.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(wav_bytes)
        print(f"[TTS] Сохранено в {path}")

    def play_audio(self, wav_bytes: bytes) -> None:
        """
        Воспроизводит WAV из байтовой строки.
        """
        arr = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        sd.play(arr, samplerate=self.sample_rate)
        sd.wait()
        print("[TTS] Воспроизведение завершено")


class YandexTTSv3:
    """
    Клиент Yandex SpeechKit TTS API v3 (REST).
    Возвращает сразу готовый WAV‑файл (container_audio: WAV).
    """
    def __init__(self,
                 folder_id: str,
                 iam_token: str,
                 voice: str = "anton",
                 emotion: str | None = None,
                 lang: str = "ru-RU"):
        self.url = "https://tts.api.cloud.yandex.net:443/tts/v3/utteranceSynthesis"
        self.headers = {
            "Authorization": f"Bearer {iam_token}",
            "x-folder-id": folder_id,
            "Content-Type": "application/json"
        }
        self.voice = voice
        self.role = emotion
        self.lang = lang

    def synthesize(self, text: str, speed: float, max_retries: int = 3, max_chars: int = 200,
                   parallel: bool = False, max_workers: int = 4) -> bytes:
        """
        Синтезирует длинный текст по кускам, возвращает склеенные WAV-байты.
        Если parallel=True, синтезирует чанки параллельно.
        """
        chunks = split_text(text, max_chars=max_chars)
        if len(chunks) == 1:
            return self._synthesize_chunk(chunks[0], speed, max_retries)

        print(f"[TTSv3] Текст разбит на {len(chunks)} чанков.")

        # 1. Синтез каждого чанка (параллельно или последовательно)
        if parallel:
            wav_chunks = self._synthesize_chunks_parallel(chunks,speed, max_retries, max_workers)
        else:
            wav_chunks = []
            for idx, chunk in enumerate(chunks):
                print(f"[TTSv3] Обрабатывается чанк {idx + 1}/{len(chunks)}: {repr(chunk[:60])}...")
                wav = self._synthesize_chunk(chunk, speed, max_retries)
                wav_chunks.append(wav)

        # 2. Склейка WAV-фрагментов
        result_buf = BytesIO()
        params_set = False
        all_data = []

        for w in wav_chunks:
            buf = BytesIO(w)
            with wave.open(buf, "rb") as wf:
                if not params_set:
                    nchannels, sampwidth, framerate, _, comptype, compname = wf.getparams()
                    result_wf = wave.open(result_buf, "wb")
                    result_wf.setnchannels(nchannels)
                    result_wf.setsampwidth(sampwidth)
                    result_wf.setframerate(framerate)
                    params_set = True
                data = wf.readframes(wf.getnframes())
                all_data.append(data)

        result_wf.writeframes(b"".join(all_data))
        result_wf.close()
        result_buf.seek(0)
        final_wav = result_buf.read()
        print(f"[TTSv3] Все чанки объединены: {len(final_wav)} байт WAV")
        return final_wav

    def _synthesize_chunk(self, text: str, speed: float, max_retries: int = 3) -> bytes:
        """
        Синтезирует один чанк текста, возвращает WAV-байты.
        """
        hints = [{"voice": self.voice}, {"speed": speed}]
        if self.role:
            hints.append({"role": self.role})

        body = {
            "text": text,
            "hints": hints,
        }

        for attempt in range(1, max_retries + 1):
            print(f"[TTSv3] Попытка {attempt}/{max_retries}…")
            resp = requests.post(self.url, headers=self.headers, json=body, stream=True, timeout=10)
            if resp.status_code != 200:
                print(f"[TTSv3] Ошибка {resp.status_code}: {resp.text}")
            try:
                resp.raise_for_status()
                wav_bytes = b""
                for line in resp.iter_lines():
                    if not line:
                        continue
                    obj = json.loads(line)
                    chunk_b64 = obj["result"]["audioChunk"]["data"]
                    wav_bytes += base64.b64decode(chunk_b64)
                print(f"[TTSv3] Успех: получено {len(wav_bytes)} байт WAV")
                return wav_bytes

            except Exception as e:
                print(f"[TTSv3] Ошибка на попытке {attempt}: {e}")
                if attempt < max_retries:
                    time.sleep(1)
                else:
                    raise RuntimeError("TTS v3 synthesis failed after retries") from e

    def _synthesize_chunks_parallel(self, chunks, speed, max_retries, max_workers):
        """
        Синтезирует список чанков параллельно.
        """
        results = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._synthesize_chunk, chunk, speed, max_retries): idx for idx, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"[TTSv3] Chunk {idx + 1} generated an exception: {exc}")
        return results


class GoogleTTSv1:
    """
    Клиент Gemini TTS (gemini-2.5-flash-preview-tts).
    Возвращает готовые WAV-байты, умеет бить текст на чанки и объединять.
    Поддерживает temperature/top_p/voice, кэширование и параллельный синтез.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash-preview-tts",
        voice_name: str = "Enceladus",
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        out_cache_dir: Optional[str] = None,
        rpm_hard_limit: int = 10,  # ваш Tier 1 лимит на эту модель
    ):
        """
        api_key: можно не передавать, если установлен GEMINI_API_KEY в окружении.
        out_cache_dir: если указать, будет кэшировать WAV по ключу (voice+hash(text+params)).
        rpm_hard_limit: простой лимитер (скользящее окно на минуту).
        """
        if genai is None or types is None:
            raise RuntimeError("Google Gemini TTS is unavailable in this Python environment.")
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model
        self.voice_name = voice_name
        self.temperature = temperature
        self.top_p = top_p
        self.out_cache_dir = out_cache_dir
        self.rpm_hard_limit = rpm_hard_limit
        self._recent_calls_ts: List[float] = []  # для простого лимитера

        if out_cache_dir:
            os.makedirs(out_cache_dir, exist_ok=True)

    # ---------- публичный API ----------
    def synthesize(
        self,
        text: str,
        style_prompt: Optional[str] = None,
        max_chars: int = 350,
        parallel: bool = False,
        max_workers: int = 3,
        retries: int = 3,
    ) -> bytes:
        """
        Синтезирует длинный текст по чанкам и возвращает склеенный WAV.
        """
        # Собираем входной текст со стилем (если задан)
        full_text = self._merge_style(text, style_prompt)

        # Кэш всего текста (целиком). Если есть — сразу отдаем
        if self.out_cache_dir:
            key = self._cache_key(full_text)
            path = os.path.join(self.out_cache_dir, key + ".wav")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    return f.read()

        # chunks = split_text(full_text, max_chars=max_chars)

        # одиночный чанк — быстрее и ровнее
        if True:
            wav = self._synthesize_chunk_wav(full_text, retries=retries)
            if self.out_cache_dir:
                with open(os.path.join(self.out_cache_dir, self._cache_key(full_text) + ".wav"), "wb") as f:
                    f.write(wav)
            return wav

        # несколько чанков
        # if parallel:
        #     wav_parts = self._synthesize_chunks_parallel(chunks, max_workers=max_workers, retries=retries)
        # else:
        #     wav_parts = [self._synthesize_chunk_wav(c, retries=retries) for c in chunks]
        #
        # # склейка WAV-данных (одинаковые параметры 16-bit/24kHz/mono)
        # data_blocks: List[bytes] = []
        # for w in wav_parts:
        #     with wave.open(io.BytesIO(w), "rb") as wf:
        #         data_blocks.append(wf.readframes(wf.getnframes()))
        #
        # out = io.BytesIO()
        # with wave.open(out, "wb") as wf:
        #     wf.setnchannels(1)
        #     wf.setsampwidth(2)
        #     wf.setframerate(24000)
        #     wf.writeframes(b"".join(data_blocks))
        # final_wav = out.getvalue()
        #
        # if self.out_cache_dir:
        #     with open(os.path.join(self.out_cache_dir, self._cache_key(full_text) + ".wav"), "wb") as f:
        #         f.write(final_wav)
        # return final_wav

    # ---------- внутр. методы ----------
    def _synthesize_chunks_parallel(self, chunks: List[str], max_workers: int, retries: int) -> List[bytes]:
        res = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(self._synthesize_chunk_wav, chunk, retries): i for i, chunk in enumerate(chunks)}
            for fut in as_completed(futs):
                idx = futs[fut]
                res[idx] = fut.result()
        return res  # type: ignore

    def _synthesize_chunk_wav(self, text: str, retries: int = 3) -> bytes:
        """
        Генерирует один чанк и возвращает WAV-байты.
        Уважает лимиты RPM и 429 RetryInfo.
        """
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                self._respect_rpm_limit()

                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        temperature=self.temperature,
                        top_p=self.top_p,
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice_name)
                            )
                        ),
                    ),
                )

                # учёт вызова в наш локальный RPM-лимитер
                self._recent_calls_ts.append(time.time())

                part = resp.candidates[0].content.parts[0]
                pcm_bytes = part.inline_data.data  # уже bytes
                return lpcm_to_wav_bytes(pcm_bytes, sample_rate=24000, channels=1, sample_width=2)

            except Exception as e:
                last_err = e
                s = str(e)
                # Если это 429 — поспим согласно RetryInfo
                if "429" in s or "RESOURCE_EXHAUSTED" in s or "retryDelay" in s:
                    delay = parse_retry_delay_seconds(e, default_sec=40)
                    time.sleep(delay)
                else:
                    # другие ошибки — мягкий бэкофф
                    time.sleep(min(2 ** attempt, 8))
        # если так и не удалось
        raise RuntimeError(f"Gemini TTS chunk failed after {retries} retries") from last_err

    def _respect_rpm_limit(self):
        """
        Простой локальный лимитер: не больше rpm_hard_limit вызовов за последние 60s.
        """
        if not self.rpm_hard_limit:
            return
        now = time.time()
        # вычищаем окно
        self._recent_calls_ts[:] = [t for t in self._recent_calls_ts if now - t < 60.0]
        if len(self._recent_calls_ts) >= self.rpm_hard_limit:
            # ждём до безопасного момента
            sleep_for = 60.0 - (now - min(self._recent_calls_ts))
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _cache_key(self, text: str) -> str:
        h = hashlib.sha1(f"{self.model}|{self.voice_name}|{self.temperature}|{self.top_p}|{text}".encode("utf-8")).hexdigest()
        return h[:16]

    @staticmethod
    def _merge_style(text: str, style_prompt: Optional[str]) -> str:
        if not style_prompt:
            return text
        return (
                    "Style instructions"
                    f"{style_prompt}\n\n"
                    "TEXT"
                    f"{text}"
                )


def save_lpcm_as_wav(lpcm_bytes: bytes, file_path: str,
                     sample_rate: int = 48000,
                     channels: int = 1,
                     sample_width: int = 2) -> None:
    """
    Обёртка raw LPCM → полноценный WAV с RIFF-заголовком.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(lpcm_bytes)

def lpcm_to_wav_bytes(pcm: bytes, sample_rate: int = 24000,
                      channels: int = 1, sample_width: int = 2) -> bytes:
    """
    Упаковывает raw LPCM (LE) → WAV.
    Gemini TTS сейчас отдает 16-bit/24kHz mono PCM в inline_data.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()

def split_text(text, max_chars=250):
    """
    Делит текст на чанки по словам, но завершает чанк на конце предложения или при превышении лимита символов.
    """
    words = text.strip().split()
    chunks = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > max_chars:
            if current:
                chunks.append(current.strip())
                current = ""
        if current:
            current += " "
        current += word
        if re.search(r'[.!?…]$', word):
            chunks.append(current.strip())
            current = ""
    if current:
        chunks.append(current.strip())
    return chunks

def parse_retry_delay_seconds(err: Exception, default_sec: int = 40) -> int:
    """
    Пытается вытащить RetryInfo.retryDelay из текста ошибки 429.
    Если не получилось — возвращает default_sec.
    """
    s = str(err)
    m = re.search(r"retryDelay['\"]?:\s*['\"]?(\d+)\s*s?", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)\w", s)
    if m2:
        return int(m2.group(1))
    return default_sec


class YandexStreamingSynthesizer:
    """
    Yandex SpeechKit TTS v3 gRPC StreamSynthesis.
    Потоковый синтез речи через bidirectional gRPC stream.
    """
    def __init__(self, folder_id: str, iam_token: str, voice: str = "anton", role: str | None = None):
        self.folder_id = folder_id
        self.iam_token = iam_token
        self.voice = voice
        self.role = role
        self.stub = self._create_stub()

    def _create_stub(self):
        try:
            creds = grpc.ssl_channel_credentials()
            channel = grpc.secure_channel('tts.api.cloud.yandex.net:443', creds)
            return tts_grpc.SynthesizerStub(channel)
        except Exception as e:
            logging.error(f"Failed to connect to Yandex TTS: {e}")
            return None

    def _generate_requests(self, text_queue: _queue.Queue, speed: float = 1.0, sample_rate: int = 48000, voice: str = None):
        """
        gRPC request generator:
        1. Отправляет options (настройки синтеза)
        2. Затем отправляет текст из очереди (None = конец)
        """
        # Настройки аудио формата
        audio_format = tts_pb2.AudioFormatOptions(
            raw_audio=tts_pb2.RawAudio(
                audio_encoding=tts_pb2.RawAudio.LINEAR16_PCM,
                sample_rate_hertz=sample_rate
            )
        )

        # Формируем опции синтеза (без hints!)
        synthesis_options = tts_pb2.SynthesisOptions(
            voice=voice or self.voice,
            speed=speed,
            output_audio_spec=audio_format
        )

        # Добавляем role если указан
        if self.role:
            synthesis_options.role = self.role

        # Первый запрос с опциями
        yield tts_pb2.StreamSynthesisRequest(options=synthesis_options)

        # Отправляем текст из очереди
        while True:
            text_chunk = text_queue.get()
            if text_chunk is None:
                break
            yield tts_pb2.StreamSynthesisRequest(
                synthesis_input=tts_pb2.SynthesisInput(text=text_chunk)
            )

    def synthesize_streaming(
        self,
        text_queue: _queue.Queue,
        audio_queue: _queue.Queue,
        speed: float = 1.0,
        sample_rate: int = 48000,
        voice: str = None
    ):
        """
        Запускается в отдельном потоке.
        Читает текст из text_queue, отправляет в gRPC stream,
        получает аудио чанки и складывает в audio_queue.
        """
        if not self.stub:
            logging.error("Stub not initialized; cannot synthesize.")
            audio_queue.put(None)
            return

        try:
            responses = self.stub.StreamSynthesis(
                self._generate_requests(text_queue, speed, sample_rate, voice),
                metadata=[('authorization', f'Api-Key {self.iam_token}')]
            )

            for resp in responses:
                if hasattr(resp, 'audio_chunk') and resp.audio_chunk.data:
                    audio_queue.put(resp.audio_chunk.data)

            # Сигнализируем об окончании
            audio_queue.put(None)

        except grpc.RpcError as e:
            logging.error(f"gRPC streaming TTS error: {e.code()} — {e.details()}")
            audio_queue.put(None)
        except Exception as e:
            logging.error(f"Unexpected error in synthesize_streaming: {e}")
            audio_queue.put(None)


class TtsStreamingSession:
    """
    Потоковая сессия TTS для одного ответа.
    Отправляет текст в gRPC stream и получает аудио чанки в реальном времени.
    """
    def __init__(
        self,
        synthesizer: YandexStreamingSynthesizer,
        speed: float,
        sample_rate: int,
        loop: asyncio.AbstractEventLoop,
        audio_queue: asyncio.Queue,
        voice: str = None
    ):
        self._synth = synthesizer
        self._speed = speed
        self._sample_rate = sample_rate
        self._loop = loop
        self._audio_queue = audio_queue
        self._text_queue: _queue.Queue = _queue.Queue()
        self._task: asyncio.Task | None = None
        self._drain_task: asyncio.Task | None = None
        self._thread_audio_queue: _queue.Queue | None = None
        self._voice = voice
        self._started = False

    def start(self):
        """Arm the session. The gRPC synthesis stream opens lazily on the first
        non-empty text chunk (see feed), so a response that produces no text never
        opens a stream and never triggers Yandex 'Empty text' errors."""
        self._started = False

    def _ensure_started(self):
        if self._started:
            return
        self._started = True
        thread_audio_queue = _queue.Queue()
        self._thread_audio_queue = thread_audio_queue
        self._task = self._loop.create_task(
            asyncio.to_thread(
                self._synth.synthesize_streaming,
                self._text_queue,
                thread_audio_queue,
                self._speed,
                self._sample_rate,
                self._voice
            )
        )
        # Запускаем задачу для перекладывания аудио чанков из thread queue в async queue
        self._drain_task = asyncio.create_task(self._drain_audio(thread_audio_queue))

    async def _drain_audio(self, thread_queue: _queue.Queue):
        """Перекладывает аудио чанки из thread queue в async queue."""
        while True:
            chunk = await asyncio.to_thread(thread_queue.get)
            if chunk is None:
                await self._audio_queue.put(None)
                break
            await self._audio_queue.put(chunk)

    def feed(self, text: str):
        """Queue text for synthesis. Empty/whitespace text is dropped (T1) and the
        gRPC stream is opened on the first real chunk (T2)."""
        if not text or not text.strip():
            return
        self._ensure_started()
        self._text_queue.put(text)

    async def finish(self):
        """Сигнализирует об окончании текста и ждет завершения синтеза."""
        if not self._started:
            # T3: nothing was synthesized — unblock the audio consumer cleanly.
            await self._audio_queue.put(None)
            return
        self._text_queue.put(None)
        if self._task:
            await self._task
        if self._drain_task:
            await self._drain_task

    def cancel(self):
        """Прерывает потоковый синтез и останавливает перенос аудио-чанков."""
        if not self._started:
            try:
                self._audio_queue.put_nowait(None)
            except Exception:
                pass
            return
        try:
            self._text_queue.put_nowait(None)
        except Exception:
            pass

        if self._thread_audio_queue is not None:
            try:
                self._thread_audio_queue.put_nowait(None)
            except Exception:
                pass

        if self._task and not self._task.done():
            self._task.cancel()

        if self._drain_task and not self._drain_task.done():
            self._drain_task.cancel()


# if __name__ == "__main__":
#     # Настройте переменные:
#     FOLDER_ID = ""
#     IAM_TOKEN = ""
#
#     tts = YandexTTS(folder_id=FOLDER_ID, iam_token=IAM_TOKEN)
#
#     text = "Привет! Это тест синтеза речи через Yandex SpeechKit в формате WAV."
#     wav_data = tts.synthesize(text)
#
#     # Сохраняем и/или проигрываем:
#     out_file = os.path.join("data", "plitonit", "audio", "test_output.wav")
#     tts.save_audio(wav_data, out_file)
#     tts.play_audio(wav_data)
