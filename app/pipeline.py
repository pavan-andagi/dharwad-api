"""
The four model stages of the voice pipeline, each loaded once and kept in memory.

  asr      : faster-whisper (English speech -> English text)
  en_kn    : ai4bharat/indictrans2-en-indic-dist-200M (English -> standard Kannada)
  dialect  : fine-tuned IndicTrans2-1B (standard Kannada -> Dharwad Kannada)
  tts      : ai4bharat/indic-parler-tts (Kannada text -> speech)

Every stage has its own lock: one request can be in TTS while another is in ASR,
but two requests never run the same model on the GPU at the same time.
MOCK_MODEL=1 replaces every model with a cheap fake for tests/CI.
"""

import io
import os
import threading
import time
import wave
from typing import Optional

MOCK = os.getenv("MOCK_MODEL", "0") == "1"

DIALECT_MODEL = os.getenv("MODEL_ID", "pavan007123/dharwad-kannada-indictrans2")
DIALECT_REVISION = os.getenv("MODEL_REVISION", "main")
EN_KN_MODEL = os.getenv("EN_KN_MODEL", "ai4bharat/indictrans2-en-indic-dist-200M")
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "small.en")
TTS_MODEL = os.getenv("TTS_MODEL", "ai4bharat/indic-parler-tts")
TTS_DESCRIPTION = os.getenv(
    "TTS_DESCRIPTION",
    "Suresh speaks in a clear, natural Kannada voice at a moderate pace with a neutral tone. The recording is very high quality with no background noise.",
)
NUM_BEAMS = int(os.getenv("NUM_BEAMS", "5"))
MAX_LEN = int(os.getenv("MAX_LENGTH", "256"))
ASR_SAMPLE_RATE = 16_000
IT2_BATCH = int(os.getenv("IT2_BATCH", "8"))          # sentences per GPU batch
TTS_PAUSE_SECONDS = float(os.getenv("TTS_PAUSE_SECONDS", "0.3"))
MAX_SENTENCE_CHARS = 300


def split_sentences(text: str) -> list[str]:
    """
    Split a paragraph into sentences for English and Kannada.
    The models were trained on single sentences (max 128 tokens), so long input is
    translated and spoken sentence by sentence instead of in one go.
    """
    import re
    parts = []
    for line in re.split(r"\n+", text):
        parts += re.split(r"(?<=[.!?\u0964])\s+", line.strip())   # . ! ? and the danda
    out = []
    for p in (x.strip() for x in parts):
        if not p:
            continue
        # very long sentence without full stops: break at commas / semicolons
        while len(p) > MAX_SENTENCE_CHARS:
            cut = max(p.rfind(c, 0, MAX_SENTENCE_CHARS) for c in ",;:")
            if cut <= 0:
                cut = p.rfind(" ", 0, MAX_SENTENCE_CHARS)
            if cut <= 0:
                cut = MAX_SENTENCE_CHARS
            out.append(p[:cut + 1].strip())
            p = p[cut + 1:].strip()
        if p:
            out.append(p)
    return out


def wav_bytes(samples, sample_rate: int) -> bytes:
    """float32 mono samples in [-1, 1] -> 16-bit PCM WAV bytes (stdlib only)."""
    import numpy as np
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _silence_wav(seconds=0.5, sr=16_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


def _cuda():
    import torch
    return torch.cuda.is_available()


class Stage:
    """Base class: background loading, status for /health, and a per-model lock."""

    name = "stage"
    model_id = ""

    def __init__(self):
        self.ready = False
        self.error: Optional[str] = None
        self.device = "mock" if MOCK else "pending"
        self.load_seconds: Optional[float] = None
        self._lock = threading.Lock()

    def load(self):
        t0 = time.perf_counter()
        try:
            self._mock_load() if MOCK else self._load()
            self.load_seconds = round(time.perf_counter() - t0, 1)
            self.ready = True
        except Exception as e:  # keep the server alive; /health shows the failure
            self.error = f"{type(e).__name__}: {e}"
        return self

    def status(self) -> dict:
        s = "ok" if self.ready else ("error" if self.error else "loading")
        out = {"status": s, "model": self.model_id, "device": self.device, "load_seconds": self.load_seconds}
        if self.error:
            out["error"] = self.error
        return out

    def run(self, *args, **kwargs):
        with self._lock:
            return self._run(*args, **kwargs)

    # subclasses implement these
    def _load(self): ...
    def _mock_load(self): ...
    def _run(self, *a, **k): ...


# --------------------------------------------------------------------------- #
# 1. Speech-to-text
# --------------------------------------------------------------------------- #
class ASR(Stage):
    name = "asr"

    def __init__(self):
        super().__init__()
        self.model_id = f"faster-whisper/{WHISPER_SIZE}"

    def _load(self):
        from faster_whisper import WhisperModel
        want_gpu = _cuda() and os.getenv("ASR_DEVICE", "auto") != "cpu"
        if want_gpu:
            try:
                self.model = WhisperModel(WHISPER_SIZE, device="cuda", compute_type="float16")
                self.device = "cuda"
                return
            except Exception:
                pass  # usually missing cuDNN/cuBLAS for CTranslate2 -> fall back to CPU
        self.model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
        self.device = "cpu"

    def _mock_load(self):
        self.model = None

    def decode(self, audio_bytes: bytes):
        """Any audio format (wav/mp3/m4a/ogg/webm) -> float32 mono 16 kHz numpy array."""
        if MOCK:
            return None
        from faster_whisper.audio import decode_audio
        return decode_audio(io.BytesIO(audio_bytes), sampling_rate=ASR_SAMPLE_RATE)

    def _run(self, audio) -> str:
        if MOCK:
            return "I like today's weather."
        segments, _ = self.model.transcribe(audio, language="en", beam_size=5, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()


# --------------------------------------------------------------------------- #
# 2 & 3. IndicTrans2 (same wrapper for EN->KN and KN->Dharwad)
# --------------------------------------------------------------------------- #
class IndicTrans2(Stage):
    def __init__(self, name, model_id, src_lang, tgt_lang, revision="main", mock_prefix=""):
        super().__init__()
        self.name, self.model_id, self.revision = name, model_id, revision
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self._mock_prefix = mock_prefix

    def _load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        try:
            from IndicTransToolkit.processor import IndicProcessor
        except ImportError:
            from IndicTransToolkit import IndicProcessor

        self.device = "cuda" if _cuda() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision, trust_remote_code=True)
        self.model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True, torch_dtype=dtype
            ).to(self.device).eval()
        )
        self.ip = IndicProcessor(inference=True)
        self._torch = torch

    def _mock_load(self):
        pass

    def _run(self, texts: list[str]) -> list[str]:
        if MOCK:
            return [f"{self._mock_prefix}{t}" for t in texts]
        out = []
        for i in range(0, len(texts), IT2_BATCH):  # bounded batches keep GPU memory predictable
            out += self._translate_batch(texts[i:i + IT2_BATCH])
        return out

    def _translate_batch(self, texts: list[str]) -> list[str]:
        with self._torch.inference_mode():
            batch = self.ip.preprocess_batch(texts, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
            self.tok.src_lang = self.src_lang
            enc = self.tok(batch, truncation=True, padding="longest", max_length=MAX_LEN,
                           return_tensors="pt").to(self.device)
            out = self.model.generate(**enc, num_beams=NUM_BEAMS, max_length=MAX_LEN, early_stopping=True,
                                      no_repeat_ngram_size=3, use_cache=True)
            dec = self.tok.batch_decode(out.detach().cpu().tolist(), skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
            return self.ip.postprocess_batch(dec, lang=self.tgt_lang)


# --------------------------------------------------------------------------- #
# 4. Text-to-speech
# --------------------------------------------------------------------------- #
class TTS(Stage):
    name = "tts"
    model_id = TTS_MODEL

    def _load(self):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        if _cuda():
            self.device = "cuda"
            # bf16 on GPUs that support it (RTX 30xx/40xx, A-series); fp16 otherwise.
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=dtype).to("cuda")
        else:
            # CPU: the audio codec has no bf16 CPU kernels -> fp32 + int8 dynamic quantization
            self.device = "cpu"
            m = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL, torch_dtype=torch.float32)
            self.model = torch.ao.quantization.quantize_dynamic(m, {torch.nn.Linear}, dtype=torch.qint8)
        self.model.eval()
        self.prompt_tok = AutoTokenizer.from_pretrained(TTS_MODEL)
        self.desc_tok = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        self.sample_rate = self.model.config.sampling_rate
        self._torch = torch

    def _mock_load(self):
        self.sample_rate = 16_000

    def _run(self, text: str, description: str = TTS_DESCRIPTION) -> bytes:
        """Speak sentence by sentence (Parler-TTS is trained on short utterances), then join."""
        sentences = split_sentences(text) or [text]
        if MOCK:
            return _silence_wav(0.5 * len(sentences), self.sample_rate)
        import numpy as np
        pause = np.zeros(int(TTS_PAUSE_SECONDS * self.sample_rate), dtype=np.float32)
        pieces = []
        with self._torch.inference_mode():
            d = self.desc_tok(description, return_tensors="pt").to(self.device)
            for sentence in sentences:
                p = self.prompt_tok(sentence, return_tensors="pt").to(self.device)
                audio = self.model.generate(
                    input_ids=d.input_ids, attention_mask=d.attention_mask,
                    prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                )
                pieces += [audio.to(self._torch.float32).cpu().numpy().squeeze(), pause]
        return wav_bytes(np.concatenate(pieces[:-1]), self.sample_rate)


def build_stages(enable_voice: bool) -> dict:
    """Ordered so the core Dharwad model loads first."""
    stages = {
        "dialect": IndicTrans2("dialect", DIALECT_MODEL, "kan_Knda", "kan_Knda",
                               revision=DIALECT_REVISION, mock_prefix="[dharwad] "),
    }
    if enable_voice:
        stages["en_kn"] = IndicTrans2("en_kn", EN_KN_MODEL, "eng_Latn", "kan_Knda", mock_prefix="[kn] ")
        stages["asr"] = ASR()
        stages["tts"] = TTS()
    return stages
