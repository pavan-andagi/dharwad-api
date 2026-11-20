import base64
import io
import os
import time
import wave

os.environ["MOCK_MODEL"] = "1"
os.environ["API_KEY"] = "test-key"

from fastapi.testclient import TestClient  # noqa: E402
from app.main import STAGES, app  # noqa: E402

AUTH = {"Authorization": "Bearer test-key"}


def wait_ready(client):
    for _ in range(100):
        if client.get("/ping").status_code == 200:
            return
        time.sleep(0.05)
    raise AssertionError("models never became ready")


def fake_wav(seconds=1.0, sr=16_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


def test_health_lists_all_models():
    with TestClient(app) as c:
        wait_ready(c)
        h = c.get("/health").json()
        assert h["status"] == "ok"
        assert set(h["models"]) == {"dialect", "en_kn", "asr", "tts"}


def test_translate_kannada():
    with TestClient(app) as c:
        wait_ready(c)
        r = c.post("/translate", json={"text": "ನಾನು ಶಾಲೆಗೆ ಹೋಗಿದ್ದೆ"}, headers=AUTH)
        assert r.status_code == 200, r.text
        assert r.json()["translations"] == ["[dharwad] ನಾನು ಶಾಲೆಗೆ ಹೋಗಿದ್ದೆ"]
        r = c.post("/translate", json={"texts": ["ಒಂದು", "ಎರಡು"]}, headers=AUTH)
        assert len(r.json()["translations"]) == 2


def test_translate_english():
    with TestClient(app) as c:
        wait_ready(c)
        r = c.post("/translate/en", json={"text": "Why are you so late?"}, headers=AUTH)
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["kannada"] == "[kn] Why are you so late?"
        assert b["dharwad"] == "[dharwad] [kn] Why are you so late?"
        assert set(b["timings_ms"]) == {"en_kn", "dialect"}


def test_tts_returns_wav():
    with TestClient(app) as c:
        wait_ready(c)
        r = c.post("/tts", json={"text": "ನಂಗ ಇಷ್ಟ ಆಗೇತಿ"}, headers=AUTH)
        assert r.status_code == 200
        assert r.headers["content-type"] == "audio/wav"
        assert r.content[:4] == b"RIFF"


def test_speak_full_pipeline():
    with TestClient(app) as c:
        wait_ready(c)
        r = c.post("/speak", files={"audio": ("in.wav", fake_wav(), "audio/wav")}, headers=AUTH)
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["english"] == "I like today's weather."
        assert b["dharwad"].startswith("[dharwad]")
        assert base64.b64decode(b["audio_wav_base64"])[:4] == b"RIFF"
        assert set(b["timings_ms"]) == {"asr", "en_kn", "dialect", "tts"}

        # text only, skip TTS
        r = c.post("/speak", files={"audio": ("in.wav", fake_wav(), "audio/wav")},
                   data={"tts": "false"}, headers=AUTH)
        assert r.json()["audio_wav_base64"] is None
        assert "tts" not in r.json()["timings_ms"]


def test_auth_and_validation():
    with TestClient(app) as c:
        wait_ready(c)
        assert c.post("/translate", json={"text": "x"}).status_code == 401
        assert c.post("/speak", files={"audio": ("a.wav", fake_wav(), "audio/wav")}).status_code == 401
        assert c.post("/translate", json={}, headers=AUTH).status_code == 422
        assert c.post("/translate", json={"text": "a", "texts": ["b"]}, headers=AUTH).status_code == 422
        assert c.post("/translate", json={"text": "x" * 5000}, headers=AUTH).status_code == 422
        assert c.post("/translate/en", json={"text": ""}, headers=AUTH).status_code == 422
        assert c.post("/speak", files={"audio": ("a.wav", b"", "audio/wav")}, headers=AUTH).status_code == 422
        assert c.post("/speak", headers=AUTH).status_code == 422


def test_not_ready_returns_503():
    with TestClient(app) as c:
        wait_ready(c)
        STAGES["tts"].ready = False
        try:
            assert c.get("/ping").status_code == 204
            assert c.post("/tts", json={"text": "x"}, headers=AUTH).status_code == 503
            # /speak without TTS still works while TTS is loading
            r = c.post("/speak", files={"audio": ("a.wav", fake_wav(), "audio/wav")},
                       data={"tts": "false"}, headers=AUTH)
            assert r.status_code == 200
        finally:
            STAGES["tts"].ready = True


def test_metrics():
    with TestClient(app) as c:
        wait_ready(c)
        c.post("/translate", json={"text": "x"}, headers=AUTH)
        m = c.get("/metrics").text
        assert "api_requests_total" in m and "stage_latency_seconds" in m


def test_browser_ui_served():
    with TestClient(app) as c:
        r = c.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "getUserMedia" in r.text and 'fetch("speak"' in r.text
