# Dharwad Kannada Speech & Translation API

Speak English, hear **Dharwad Kannada**. All four models run on one GPU server, so a client only
needs a microphone and a speaker — no models, no PyTorch, no GPU.

The core is my LoRA-fine-tuned IndicTrans2-1B model
([`pavan007123/dharwad-kannada-indictrans2`](https://huggingface.co/pavan007123/dharwad-kannada-indictrans2)),
which converts standard (bookish) Kannada into the Dharwad dialect.
Training code and thesis: [Bookish_kannada_to_Dharwad_Kannada](https://github.com/pavan-andagi/Bookish_kannada_to_Dharwad_Kannada).

```
 YOUR PC (client)                         GPU SERVER (RunPod) — all models loaded once
 ────────────────                         ─────────────────────────────────────────────
 🎤 record ── WAV ── POST /speak ───────►  1. Whisper small.en      English speech → English text
                                           2. IndicTrans2 200M      English → standard Kannada
                                           3. My fine-tuned 1B      Kannada → Dharwad Kannada
                                           4. Indic Parler-TTS      Dharwad text → speech
 🔊 play  ◄── texts + WAV + timings ─────
```

```
GitHub (code) ──CI: test + build──► Docker Hub (image) ──pull──► RunPod GPU (Pod or Serverless)
                                                                    │ first start
                                     Hugging Face Hub (weights) ◄───┘ download → cached on volume
```

## API

| Method | Path | Input | Output |
|---|---|---|---|
| `POST` | `/speak` | multipart `audio` file (wav/mp3/m4a/ogg/webm, English speech), optional `tts=false` | `english`, `kannada`, `dharwad`, `audio_wav_base64`, `timings_ms` per model |
| `POST` | `/translate/en` | `{"text": "English sentence"}` | `english`, `kannada`, `dharwad`, `timings_ms` |
| `POST` | `/translate` | `{"text": "ಕನ್ನಡ"}` or `{"texts": [...]}` (standard Kannada) | `translations` (Dharwad) |
| `POST` | `/tts` | `{"text": "ಕನ್ನಡ / Dharwad text"}` | `audio/wav` |
| `GET` | `/health` | – | status of each model: `loading` / `ok` / `error`, device, load time |
| `GET` | `/ping` | – | RunPod Serverless probe: `204` loading, `200` ready |
| `GET` | `/metrics` | – | Prometheus: requests, end-to-end and **per-model** latency |
| `GET` | `/docs` | – | Interactive Swagger UI |

```bash
URL=https://<POD_ID>-8000.proxy.runpod.net

# English speech -> Dharwad text + speech
curl -X POST $URL/speak -H "Authorization: Bearer $API_KEY" -F audio=@hello.wav

# English text -> Dharwad text
curl -X POST $URL/translate/en -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" -d '{"text": "Why are you so late?"}'

# Dharwad text -> speech file
curl -X POST $URL/tts -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" -d '{"text": "ನಂಗ ಇಷ್ಟ ಆಗೇತಿ"}' -o out.wav
```

## Local voice client

`client/voice_client.py` is all a user needs: push-to-talk recording, one call to `/speak`,
prints the three texts with per-model timings, plays the Dharwad audio, and appends to
`translations_log.txt`.

```bash
pip install sounddevice numpy requests
export DHARWAD_API_URL=https://<POD_ID>-8000.proxy.runpod.net
export DHARWAD_API_KEY=<your key>
python client/voice_client.py
```

## Design decisions

- **Every model loaded once** at startup in a background thread (Dharwad model first). Health checks answer immediately during the ~6 GB download; each endpoint returns `503` until the models *it* needs are ready (e.g. `/translate` works before TTS has loaded).
- **One lock per model.** Requests overlap across stages (request A in TTS while B is in Whisper) but never run the same model concurrently, so GPU memory stays predictable.
- **GPU precision:** IndicTrans2 in fp16, Parler-TTS in bf16 (fp16 on GPUs without bf16), Whisper fp16 via CTranslate2. CPU fallback: fp32, TTS int8-quantized, Whisper int8.
- **Per-stage timings** in every response, the logs and `/metrics` → shows exactly which model is the bottleneck.
- **Privacy:** logs contain request id, sizes and latencies — **never the text or audio**. Audio is processed in memory and not stored.
- **Paragraphs are processed sentence by sentence.** The translation models were trained on single sentences (≤128 tokens), so text is split at `. ! ?` (and line breaks; overlong sentences at commas), all sentences are translated in one batched GPU pass, and TTS speaks each sentence and joins them with short pauses.
- **Input limits:** 10 MB / 60 s audio, 3000 chars per text, 16 texts per batch, max 15 sentences for speech in one request (TTS takes ~4 s per sentence and the RunPod proxy cuts off at ~100 s).
- **Secrets only via env vars** (`HF_TOKEN`, `API_KEY`).
- **`MOCK_MODEL=1`** replaces every model with a fake, so the test suite and CI run without a GPU.
- **`ENABLE_VOICE=0`** serves only the Dharwad model (text-only, ~3 GB VRAM).

## Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `HF_TOKEN` | **yes** | – | The account must have accepted the terms of the AI4Bharat models used (`indictrans2-en-indic-dist-200M`, `indictrans2-indic-indic-1B` — my checkpoint loads its model code — and `indic-parler-tts` if gated). |
| `API_KEY` | recommended | – | Clients send `Authorization: Bearer <key>`. The RunPod proxy URL is public. |
| `HF_HOME` | recommended | `/models` | `/workspace/hf` (Pod) or `/runpod-volume/hf` (Serverless) to cache weights across restarts. |
| `ENABLE_VOICE` | no | `1` | `0` = text-only server. |
| `WHISPER_SIZE` | no | `small.en` | `base.en` (faster) / `medium.en` (more accurate). |
| `TTS_DESCRIPTION` | no | clear female voice | Parler-TTS voice description; see the model card for named Kannada speakers. |
| `MODEL_ID`, `MODEL_REVISION` | no | my model, `main` | Pin a commit hash in production (the model uses `trust_remote_code`). |
| `NUM_BEAMS` | no | `5` | Lower = faster translation. |
| `IT2_BATCH` | no | `8` | Sentences per GPU batch for translation. |
| `MAX_TTS_SENTENCES` | no | `15` | Longer texts get `413` from `/tts` and `/speak` (use `tts=false`). |
| `TTS_PAUSE_SECONDS` | no | `0.3` | Silence between spoken sentences. |
| `PORT` | no | `8000` | |

## GPU sizing

All four models together need roughly **6–8 GB VRAM**. An RTX A4000 (16 GB), A5000 or 3090/4090 (24 GB) is plenty.
Container disk ≥ 30 GB (image) and a volume ≥ 20 GB (weights cache).

## 1. Build and push the image

GitHub Actions does it on every push to `main` (repo secrets `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`), or manually:

```bash
docker build --platform linux/amd64 -t <dockerhub-user>/dharwad-api:2.0 .
docker push <dockerhub-user>/dharwad-api:2.0
```

## 2a. RunPod Pod (always on)

1. **Pods → Deploy** → GPU (see sizing) → **Edit Template**:
   - Container image `<dockerhub-user>/dharwad-api:2.0`
   - **Expose HTTP Ports** `8000`
   - Volume mount `/workspace`, ≥ 20 GB
   - Env: `HF_TOKEN`, `API_KEY`, `HF_HOME=/workspace/hf`
2. Deploy; watch the logs until all four `"msg": "model_loaded"` lines appear (first start downloads ~6 GB).
3. `https://<POD_ID>-8000.proxy.runpod.net/health` → all models `ok`. Open `/docs` to try it in the browser.

The RunPod proxy cuts requests off at ~100 s — one `/speak` turn takes a few seconds on a GPU.
A Pod bills while running: **stop it when you're not using it.**

## 2b. RunPod Serverless (scale to zero)

Same image. **Serverless → New Endpoint → Load balancing** → custom image; expose port `8000`;
env `PORT=8000`, `PORT_HEALTH=8000`, `HF_TOKEN`, `API_KEY`, `HF_HOME=/runpod-volume/hf`; attach a network volume.
Min workers `0` = pay only when used, but the first request after idle waits for model loading
(for fast cold starts, bake the weights into the image instead).

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q tests     # MOCK_MODEL=1: no GPU, no downloads
```

Run the API locally without models: `MOCK_MODEL=1 uvicorn app.main:app --port 8000` → `http://localhost:8000/docs`.

## Known issues / notes

- `parler-tts` pins `transformers==4.46.1`, which conflicts with IndicTransToolkit's `4.48.0`; it is installed with `--no-deps` and its real dependencies are pinned in `requirements.txt`.
- `av>=16` breaks audio-file decoding in `faster-whisper 1.2.1`; pinned to `av==15.1.0`.
- If Whisper can't find cuDNN/cuBLAS for CTranslate2, it falls back to CPU (visible in `/health` as `"device": "cpu"`). `start.sh` adds PyTorch's bundled CUDA libraries to `LD_LIBRARY_PATH` to prevent this.

## Next steps

- Streaming: send audio back sentence-by-sentence for long inputs
- Micro-batching concurrent requests for higher GPU throughput
- CTranslate2 export of the IndicTrans2 models for lower latency
- Request tracing (OpenTelemetry/Langfuse) and a Grafana dashboard on `/metrics`
- A small web page with a mic button as a client
