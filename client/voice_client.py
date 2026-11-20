"""
Thin local client for the Dharwad voice API.

Your PC only does the I/O: record from the mic, send the audio, play the answer.
All four models (Whisper, EN->KN, Dharwad, TTS) run on the server GPU.

Install (no torch, no models needed):
    pip install sounddevice numpy requests

Run:
    export DHARWAD_API_URL=https://<POD_ID>-8000.proxy.runpod.net   # or http://localhost:8000
    export DHARWAD_API_KEY=<your API_KEY>
    python voice_client.py
"""

import base64
import io
import os
import sys
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd

API_URL = os.getenv("DHARWAD_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("DHARWAD_API_KEY", "")
MIC_SR = 16_000
LOG_FILE = Path("translations_log.txt")


def record_until_enter() -> np.ndarray:
    """Push-to-talk: Enter to start, Enter to stop. Returns float32 mono @16 kHz."""
    frames = []

    def cb(indata, n_frames, time_info, status):
        frames.append(indata.copy())

    input("\n[Enter] to start recording...")
    with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32", callback=cb):
        input("Recording... [Enter] to stop.")
    return np.concatenate(frames).squeeze() if frames else np.zeros(0, dtype=np.float32)


def to_wav(samples: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def play_wav(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes)) as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    sd.play(pcm.astype(np.float32) / 32767, sr)
    sd.wait()


def log_turn(r: dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] request_id={r['request_id']}\n")
        f.write(f"English : {r['english']}\nKannada : {r['kannada']}\nDharwad : {r['dharwad']}\n")
        f.write(f"Timings : {r['timings_ms']} total={r['latency_ms']} ms\n" + "-" * 60 + "\n")


def main():
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    try:
        h = requests.get(f"{API_URL}/health", timeout=10).json()
        print(f"Server: {API_URL}  status={h['status']}")
        if h["status"] != "ok":
            print("Models still loading - requests will return 503 until ready.")
    except Exception as e:
        sys.exit(f"Cannot reach {API_URL}: {e}")

    print("Ready. Ctrl+C to quit.")
    while True:
        audio = record_until_enter()
        if audio.size < MIC_SR // 2:
            print("Too short, try again.")
            continue

        print("Sending to server...")
        try:
            resp = requests.post(
                f"{API_URL}/speak", headers=headers, timeout=95,  # RunPod proxy cuts off at ~100 s
                files={"audio": ("speech.wav", to_wav(audio, MIC_SR), "audio/wav")},
            )
        except requests.RequestException as e:
            print(f"Network error: {e}")
            continue
        if resp.status_code != 200:
            print(f"Server error {resp.status_code}: {resp.text[:300]}")
            continue

        r = resp.json()
        print(f"[EN ] {r['english']}\n[KN ] {r['kannada']}\n[DWD] {r['dharwad']}")
        print(f"      stages {r['timings_ms']}  total {r['latency_ms']} ms")
        log_turn(r)
        if r.get("audio_wav_base64"):
            play_wav(base64.b64decode(r["audio_wav_base64"]))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye.")
