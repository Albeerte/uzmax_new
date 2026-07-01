import asyncio
import queue as _queue
import grpc
import logging
import struct
import time as _time
import requests
import numpy as np
import speech_recognition as sr
import soundfile as sf
from scipy.signal import resample_poly, butter, sosfilt

try:
    import noisereduce as _nr
except Exception:
    _nr = None


def denoise_pcm(pcm: bytes, sample_rate: int, prop_decrease: float = 0.8) -> bytes:
    """Clean a speech clip before sending it to the recognizer: a high-pass to drop
    low rumble/hum, then spectral-gating noise reduction. Returns the original audio
    unchanged on any failure so recognition never breaks."""
    import os as _os
    if _os.getenv("STT_DENOISE", "1") == "0":
        return pcm
    try:
        y = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if y.size < int(sample_rate * 0.1):
            return pcm
        sos = butter(2, 80.0 / (sample_rate / 2), btype="highpass", output="sos")
        y = sosfilt(sos, y).astype(np.float32)
        if _nr is not None:
            y = _nr.reduce_noise(y=y, sr=sample_rate, prop_decrease=prop_decrease, stationary=True)
        y = np.clip(y, -1.0, 1.0)
        return (y * 32767.0).astype(np.int16).tobytes()
    except Exception as exc:
        logging.warning("[STT] denoise failed, using raw audio: %s", exc)
        return pcm
from pydub import AudioSegment, utils
from tempfile import NamedTemporaryFile
from pathlib import Path
import os


def pcm16_to_wav(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw int16 PCM in a minimal WAV container (for the transcription API)."""
    byte_rate = sample_rate * channels * 2
    block_align = channels * 2
    data_len = len(pcm)
    return (
        b"RIFF" + struct.pack("<I", 36 + data_len) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, 16)
        + b"data" + struct.pack("<I", data_len) + pcm
    )


class WhisperSttSession:
    """Per-turn STT session backed by OpenAI (Whisper).

    Same interface as SttStreamingSession (start/feed/finish + a ``_chunks`` shim so
    the websocket handler's stop/interrupt code keeps working), but instead of
    streaming to Yandex it buffers the raw PCM and transcribes the whole utterance
    once on finish(). No live partials — accuracy and reliability over streaming.
    """

    # OpenAI rejects language='uz', so Uzbek uses auto-detect (None) + a STRONG prompt.
    # The prompt anchors both language and Latin script — without it Whisper mis-detects
    # Uzbek as Arabic/Turkish. The example words pin Uzbek orthography.
    # Keep the prompt free of concrete phrases: Whisper regurgitates example words
    # verbatim on low-content audio. Anchor only language + script.
    # Domain context (hospital, patient says name + health complaints) primes the
    # model's vocabulary and lifts accuracy on medical terms — WITHOUT listing concrete
    # complaint phrases (those get echoed back, and would collide with the echo-guard).
    _LANG_MAP = {"uz-UZ": None, "en-US": "en", "ru-RU": "ru"}
    _PROMPT_MAP = {
        "uz-UZ": ("Bu O'zbekistondagi yuqumli kasalliklar shifoxonasida shifokor-robot va bemor "
                  "o'rtasidagi suhbat. Bemor o'z ism-familiyasi va sog'lig'i bo'yicha gapiradi. "
                  "Matnni faqat o'zbek tilida, lotin yozuvida yoz."),
        "ru-RU": ("Это разговор в инфекционной больнице между роботом-помощником и пациентом. "
                  "Пациент называет своё имя и рассказывает о самочувствии. Пиши только на русском языке."),
        "en-US": ("This is a conversation at an infectious-disease hospital between an assistant "
                  "robot and a patient who gives their name and describes how they feel."),
    }
    MIN_SECONDS = 0.3    # ignore sub-300ms blips (noise) without calling the API
    MAX_SECONDS = 30.0   # cap buffer so a stuck-open mic can't send a huge clip
    TRANSCRIBE_TIMEOUT_S = 14.0   # backstop so a stalled OpenAI call can't freeze the turn
    # Energy gate: int16 RMS below this is silence/noise, not speech. Whisper invents
    # words ("Salom", subtitle credits) on near-silent audio, so skip the API entirely.
    NOISE_RMS = float(os.getenv("STT_NOISE_RMS", "200"))

    def __init__(self, transcriber, sample_rate, loop, partial_queue, language_code="uz-UZ"):
        self._transcriber = transcriber
        self._sr = int(sample_rate)
        self._loop = loop
        self._partial_queue = partial_queue        # unused: Whisper has no partials
        self._lang = self._LANG_MAP.get(language_code, None)
        self._prompt = self._PROMPT_MAP.get(language_code, self._PROMPT_MAP["uz-UZ"])
        self._buf = bytearray()
        self._fed_audio = False
        self._chunks: _queue.Queue = _queue.Queue()  # compat shim for stop/interrupt

    def start(self):
        pass

    def feed(self, pcm_bytes: bytes):
        self._fed_audio = True
        max_bytes = int(self._sr * 2 * self.MAX_SECONDS)
        if len(self._buf) < max_bytes:
            self._buf.extend(pcm_bytes)

    async def finish(self) -> str:
        min_bytes = int(self._sr * 2 * self.MIN_SECONDS)
        if not self._fed_audio or len(self._buf) < min_bytes:
            return ""
        # Noise gate: skip transcription on near-silent audio so Whisper can't
        # hallucinate phantom phrases from background noise.
        samples = np.frombuffer(bytes(self._buf), dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        if rms < self.NOISE_RMS:
            logging.info("[STT] skipped: RMS %.0f < %.0f (silence/noise)", rms, self.NOISE_RMS)
            return ""
        pcm = await asyncio.to_thread(denoise_pcm, bytes(self._buf), self._sr)
        wav = pcm16_to_wav(pcm, self._sr)
        t = _time.perf_counter()
        try:
            text = await asyncio.wait_for(
                self._transcriber.transcribe(wav, language=self._lang, prompt=self._prompt),
                timeout=self.TRANSCRIBE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logging.warning("[STT] Whisper transcribe timed out after %.0fs", self.TRANSCRIBE_TIMEOUT_S)
            return ""
        except Exception as exc:
            logging.warning("[STT] Whisper transcribe failed: %s", exc)
            return ""
        logging.info("[STT] Whisper %.2fs (%.1fs audio, lang=%s) text=%r",
                     _time.perf_counter() - t, len(self._buf) / (self._sr * 2),
                     self._lang, (text or "")[:80])
        if self._is_hallucination(text):
            logging.info("[STT] dropped hallucination/prompt-echo: %r", (text or "")[:80])
            return ""
        return text or ""

    @staticmethod
    def _words(s: str) -> list:
        import re
        return re.sub(r"[^0-9a-zà-ÿʻ'\s]", " ", (s or "").lower()).split()

    def _is_hallucination(self, text: str) -> bool:
        """Drop output that is really the prompt echoed back, punctuation/junk, or a
        known Whisper noise artifact — Whisper invents these on low-content audio."""
        words = self._words(text)
        if not words:
            return True   # empty / punctuation-only (e.g. '###')
        prompt_words = set(self._words(self._prompt))
        overlap = sum(1 for w in words if w in prompt_words) / len(words)
        if overlap >= 0.6:
            return True   # mostly prompt words -> the model echoed the instruction
        low = (text or "").lower()
        ARTIFACTS = ("amara.org", "altyazı", "subtitle", "subscribe", "редактор субтитров", "субтитр")
        return any(a in low for a in ARTIFACTS)


class YandexSttSession:
    """Per-turn STT via Yandex v1 *synchronous* recognition (non-streaming).

    Same buffer-and-recognise-on-finish shape as WhisperSttSession, but sends the whole
    clip to Yandex's short-audio REST endpoint. The v2 gRPC *stream* was unreliable here
    (empty results, 20-30s hangs); the sync endpoint is fast (~0.9s) and accurate for
    Uzbek. No live partials.
    """

    _LANG_MAP = {"uz-UZ": "uz-UZ", "en-US": "en-US", "ru-RU": "ru-RU"}
    MIN_SECONDS = 0.3
    MAX_SECONDS = 28.0    # stay under Yandex's ~1MB / 30s short-audio limit
    RECOGNIZE_TIMEOUT_S = 14.0
    NOISE_RMS = float(os.getenv("STT_NOISE_RMS", "200"))

    def __init__(self, recognizer, sample_rate, loop, partial_queue, language_code="uz-UZ"):
        self._rec = recognizer
        self._sr = int(sample_rate)
        self._loop = loop
        self._partial_queue = partial_queue        # unused: sync mode has no partials
        self._lang = self._LANG_MAP.get(language_code, "uz-UZ")
        self._buf = bytearray()
        self._fed_audio = False
        self._chunks: _queue.Queue = _queue.Queue()  # compat shim for stop/interrupt

    def start(self):
        pass

    def feed(self, pcm_bytes: bytes):
        self._fed_audio = True
        max_bytes = int(self._sr * 2 * self.MAX_SECONDS)
        if len(self._buf) < max_bytes:
            self._buf.extend(pcm_bytes)

    async def finish(self) -> str:
        min_bytes = int(self._sr * 2 * self.MIN_SECONDS)
        if not self._fed_audio or len(self._buf) < min_bytes:
            return ""
        samples = np.frombuffer(bytes(self._buf), dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(samples ** 2))) if samples.size else 0.0
        if rms < self.NOISE_RMS:
            logging.info("[STT] skipped: RMS %.0f < %.0f (silence/noise)", rms, self.NOISE_RMS)
            return ""
        pcm = await asyncio.to_thread(denoise_pcm, bytes(self._buf), self._sr)
        t = _time.perf_counter()
        try:
            text = await asyncio.wait_for(
                asyncio.to_thread(self._rec.recognize_short, pcm, self._sr, self._lang),
                timeout=self.RECOGNIZE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logging.warning("[STT] Yandex sync timed out after %.0fs", self.RECOGNIZE_TIMEOUT_S)
            return ""
        except Exception as exc:
            logging.warning("[STT] Yandex sync failed: %s", exc)
            return ""
        logging.info("[STT] Yandex-sync %.2fs (%.1fs audio, lang=%s) text=%r",
                     _time.perf_counter() - t, len(pcm) / (self._sr * 2), self._lang, (text or "")[:80])
        return text or ""

import yandex.cloud.ai.stt.v2.stt_service_pb2 as stt_pb2_v2
import yandex.cloud.ai.stt.v2.stt_service_pb2_grpc as stt_pb2_grpc_v2


class YandexSpeechRecognizer:
    CHUNK_SIZE = 4096
    SUPPORTED_RATES = {8000, 16000, 48000}

    def __init__(self, folder_id: str, iam_token: str):
        self.folder_id = folder_id
        self.iam_token = iam_token
        self.last_error = None
        self.stub = self._create_stub()

    def _create_stub(self):
        try:
            creds = grpc.ssl_channel_credentials()
            channel = grpc.secure_channel('stt.api.cloud.yandex.net:443', creds)
            return stt_pb2_grpc_v2.SttServiceStub(channel)
        except Exception as e:
            logging.error(f"Failed to connect to Yandex STT: {e}")
            return None

    def recognize_short(self, pcm: bytes, sample_rate: int, lang: str = "uz-UZ") -> str:
        """Synchronous short-audio recognition (v1 REST). Sends the whole LPCM clip in
        one POST — far more reliable here than the v2 gRPC stream. Limit ~1MB / ~30s."""
        try:
            r = requests.post(
                "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize",
                headers={"Authorization": f"Api-Key {self.iam_token}"},
                params={"folderId": self.folder_id, "lang": lang,
                        "format": "lpcm", "sampleRateHertz": sample_rate},
                data=pcm, timeout=15,
            )
            if r.status_code == 200:
                self.last_error = None
                return (r.json().get("result") or "").strip()
            self.last_error = f"{r.status_code}: {r.text[:200]}"
            logging.error("Yandex sync STT %s: %s", r.status_code, r.text[:200])
            return ""
        except Exception as exc:
            self.last_error = str(exc)
            logging.error("Yandex sync STT error: %s", exc)
            return ""

    def _generate_requests(self, audio_data: sr.AudioData, partial_results: bool = False):
        spec = stt_pb2_v2.RecognitionSpec(
            language_code='ru-RU',
            profanity_filter=True,
            model='general',
            partial_results=partial_results,
            audio_encoding='LINEAR16_PCM',
            sample_rate_hertz=audio_data.sample_rate
        )
        config = stt_pb2_v2.RecognitionConfig(
            specification=spec,
            folder_id=self.folder_id
        )
        yield stt_pb2_v2.StreamingRecognitionRequest(config=config)

        raw = audio_data.get_raw_data()
        for i in range(0, len(raw), self.CHUNK_SIZE):
            yield stt_pb2_v2.StreamingRecognitionRequest(audio_content=raw[i:i + self.CHUNK_SIZE])

    def _recognize_audio_data(self, audio_data: sr.AudioData, on_partial=None) -> str:
        """
        Распознаёт аудио. Если передан on_partial(text), включает partial_results
        и вызывает callback для промежуточных результатов.
        """
        if not self.stub:
            logging.error("Stub not initialized; cannot recognize.")
            return ""
        try:
            responses = self.stub.StreamingRecognize(
                self._generate_requests(audio_data, partial_results=on_partial is not None),
                metadata=[('authorization', f'Api-Key {self.iam_token}')]
            )
            texts = []
            for resp in responses:
                for chunk in getattr(resp, 'chunks', []):
                    if chunk.final:
                        texts.extend([alt.text for alt in chunk.alternatives])
                    elif on_partial:
                        for alt in chunk.alternatives:
                            if alt.text:
                                on_partial(alt.text)
            return ' '.join(texts).strip()
        except grpc.RpcError as e:
            logging.error(f"gRPC error: {e.code()} — {e.details()}")
            return ""
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            return ""

    def transcribe_bytes(self, wav_bytes: bytes, on_partial=None) -> str:
        """
        Транскрибирует сырые WAV-байты (без записи в файл).
        Если передан on_partial(text), будет вызываться для промежуточных результатов.
        """
        from io import BytesIO
        buf = BytesIO(wav_bytes)
        data, sr_rate = sf.read(buf, dtype='int16')
        if data.ndim > 1:
            data = data.mean(axis=1).astype(np.int16)
        raw = data.tobytes()

        target_sr = min(self.SUPPORTED_RATES, key=lambda r: abs(r - sr_rate))
        if sr_rate not in self.SUPPORTED_RATES:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
            arr_rs = resample_poly(arr, target_sr, sr_rate)
            raw = (arr_rs * np.iinfo(np.int16).max).astype(np.int16).tobytes()
            logging.info(f"transcribe_bytes: resampled {sr_rate}→{target_sr} Hz")

        audio_data = sr.AudioData(raw, target_sr, 2)
        return self._recognize_audio_data(audio_data, on_partial=on_partial)

    def load_audio_data(self, filepath: str) -> sr.AudioData:
        """
        Reads .wav/.ogg/.flac via soundfile, others via pydub, converts to mono & resamples.
        """
        ext = Path(filepath).suffix.lower()
        if ext in {'.wav', '.ogg', '.flac'}:
            data, sr_rate = sf.read(filepath, dtype='int16')
            if data.ndim > 1:
                data = data.mean(axis=1).astype(np.int16)
            raw = data.tobytes()
            sample_width = 2
        else:
            seg = AudioSegment.from_file(filepath)
            seg = seg.set_channels(1)
            raw = seg.raw_data
            sr_rate = seg.frame_rate
            sample_width = seg.sample_width

        target_sr = min(self.SUPPORTED_RATES, key=lambda r: abs(r - sr_rate))
        if sr_rate not in self.SUPPORTED_RATES:
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / np.iinfo(np.int16).max
            arr_rs = resample_poly(arr, target_sr, sr_rate)
            raw = (arr_rs * np.iinfo(np.int16).max).astype(np.int16).tobytes()
            logging.info(f"Resampled from {sr_rate}→{target_sr} Hz")

        return sr.AudioData(raw, target_sr, sample_width)

    def _generate_streaming_requests(self, sample_rate: int, chunk_queue: _queue.Queue, language_code: str = 'ru-RU'):
        """gRPC request generator: config first, then raw int16 PCM chunks from queue (None = end)."""
        spec = stt_pb2_v2.RecognitionSpec(
            language_code=language_code,
            profanity_filter=True,
            model='general',
            partial_results=True,
            audio_encoding='LINEAR16_PCM',
            sample_rate_hertz=sample_rate,
        )
        yield stt_pb2_v2.StreamingRecognitionRequest(
            config=stt_pb2_v2.RecognitionConfig(specification=spec, folder_id=self.folder_id)
        )
        while True:
            chunk = chunk_queue.get()
            if chunk is None:
                break
            yield stt_pb2_v2.StreamingRecognitionRequest(audio_content=chunk)

    def recognize_streaming(self, sample_rate: int, chunk_queue: _queue.Queue, on_partial=None, language_code: str = 'ru-RU') -> str:
        """Runs in thread. Streams raw int16 PCM chunks from queue to Yandex STT."""
        self.last_error = None
        if not self.stub:
            self.last_error = "Yandex STT client is not initialized."
            logging.error("Stub not initialized; cannot recognize.")
            return ""
        try:
            import time as _t
            t_start = _t.perf_counter()
            responses = self.stub.StreamingRecognize(
                self._generate_streaming_requests(sample_rate, chunk_queue, language_code),
                metadata=[('authorization', f'Api-Key {self.iam_token}')]
            )
            texts = []
            first_partial_at = None
            first_final_at = None
            for resp in responses:
                for chunk in getattr(resp, 'chunks', []):
                    if chunk.final:
                        if first_final_at is None:
                            first_final_at = _t.perf_counter() - t_start
                        texts.extend([alt.text for alt in chunk.alternatives])
                    elif on_partial:
                        if first_partial_at is None:
                            first_partial_at = _t.perf_counter() - t_start
                            logging.info("[STT] first partial in %.2fs (rate=%d, lang=%s)",
                                         first_partial_at, sample_rate, language_code)
                        for alt in chunk.alternatives:
                            if alt.text:
                                on_partial(alt.text)
            total = _t.perf_counter() - t_start
            logging.info("[STT] done in %.2fs (first_final=%s) text=%r",
                         total,
                         f"{first_final_at:.2f}s" if first_final_at is not None else "—",
                         (' '.join(texts).strip())[:80])
            return ' '.join(texts).strip()
        except grpc.RpcError as e:
            if "you should send at least one audio fragment" in e.details():
                # Ignore this specific error, it just means the stream was closed before audio arrived.
                pass
            else:
                self.last_error = f"{e.code().name}: {e.details()}"
                logging.error(f"gRPC streaming error: {e.code()} — {e.details()}")
            return ""
        except Exception as e:
            self.last_error = str(e)
            logging.error(f"Unexpected error in recognize_streaming: {e}")
            return ""

    def transcribe(self, filepath: str) -> str:
        """
        Splits long audio into <=5min chunks, streams each through STT, and merges text.
        """
        # Load full audio as mono
        audio = AudioSegment.from_file(filepath)
        audio = audio.set_channels(1)

        chunk_ms = 1 * 60 * 1000
        chunks = utils.make_chunks(audio, chunk_ms)

        all_texts = []
        for idx, chunk in enumerate(chunks):
            with NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                chunk.export(tmp.name, format='wav')
                tmp_path = tmp.name

            # Convert to AudioData
            audio_data = self.load_audio_data(tmp_path)
            os.remove(tmp_path)

            # Recognize chunk
            part_text = self._recognize_audio_data(audio_data)
            if part_text:
                all_texts.append(part_text)

        return ' '.join(all_texts)


class SttStreamingSession:
    """
    Per-turn streaming STT session.
    Feeds raw int16 PCM chunks into a live gRPC StreamingRecognize call
    as the user speaks, so partial results arrive in real time.
    """

    def __init__(
        self,
        recognizer: YandexSpeechRecognizer,
        sample_rate: int,
        loop: asyncio.AbstractEventLoop,
        partial_queue: asyncio.Queue,
        language_code: str = 'ru-RU'
    ):
        self._rec = recognizer
        self._sr = min(recognizer.SUPPORTED_RATES, key=lambda r: abs(r - sample_rate))
        self._loop = loop
        self._partial_queue = partial_queue
        self._chunks: _queue.Queue = _queue.Queue()
        self._task: asyncio.Task | None = None
        self._lang = language_code
        self._fed_audio = False
        self._started = False
        self._last_partial = ""

    # Max seconds to wait after end-of-speech for Yandex's result (S2). Tuned for
    # this network where Yandex's first result can lag 5-9s on a degraded link:
    # too low cuts off a real answer, too high re-introduces 20-30s hangs.
    # If we already captured a partial we settle faster (FINAL_TIMEOUT_WITH_PARTIAL_S).
    FINAL_TIMEOUT_S = 9.0
    FINAL_TIMEOUT_WITH_PARTIAL_S = 3.0

    def start(self):
        """Arm the session. The gRPC stream is opened lazily on the first audio
        chunk (see feed) so Yandex's "send a chunk within 5s" timer does not start
        during microphone warm-up/permission/silence, which otherwise aborts the
        stream with INVALID_ARGUMENT."""
        self._started = False

    def _ensure_stream(self):
        if self._started:
            return
        self._started = True
        self._task = self._loop.create_task(
            asyncio.to_thread(
                self._rec.recognize_streaming,
                self._sr,
                self._chunks,
                self._on_partial,
                self._lang
            )
        )
        # If finish() returns early on timeout, this task is orphaned; consume its
        # result/exception so it doesn't surface as "exception never retrieved".
        self._task.add_done_callback(lambda t: t.cancelled() or t.exception())

    def _on_partial(self, text: str):
        if text and text.strip():
            self._last_partial = text   # S3: keep latest partial as fallback final
        asyncio.run_coroutine_threadsafe(self._partial_queue.put(text), self._loop)

    def feed(self, pcm_bytes: bytes):
        """Push a raw int16 PCM chunk into the stream (opening it on first chunk)."""
        self._fed_audio = True
        self._ensure_stream()
        self._chunks.put(pcm_bytes)

    async def finish(self) -> str:
        """Signal end-of-stream and await the final transcription, bounded by
        FINAL_TIMEOUT_S. On timeout return the latest partial so a slow/never-arriving
        final does not block the turn for 20-30s (S2/S3)."""
        if not self._fed_audio or not self._task:
            # No audio was sent — nothing to recognise. Drain so the thread (if any)
            # exits without a gRPC INVALID_ARGUMENT.
            self._chunks.put(None)
            if self._task:
                try:
                    await asyncio.wait_for(asyncio.shield(self._task), timeout=1.0)
                except (asyncio.TimeoutError, Exception):
                    pass
            return ""

        self._chunks.put(None)
        # If partials already arrived (Yandex is responding), only a short wait for
        # the final is needed; otherwise allow longer for a slow first result.
        timeout = self.FINAL_TIMEOUT_WITH_PARTIAL_S if self._last_partial else self.FINAL_TIMEOUT_S
        try:
            final = await asyncio.wait_for(asyncio.shield(self._task), timeout=timeout)
            return (final or self._last_partial).strip()
        except asyncio.TimeoutError:
            logging.info("[STT] final timed out after %.1fs; using last partial %r",
                         timeout, self._last_partial[:60])
            return self._last_partial.strip()
        except Exception as exc:
            logging.warning("[STT] finish error: %s", exc)
            return self._last_partial.strip()
