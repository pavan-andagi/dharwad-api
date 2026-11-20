# GPU image for the Dharwad Kannada speech & translation API.
# Base contains CUDA 12.1 + cuDNN 9 + torch/torchaudio 2.4.1. torch/torchaudio are re-pinned in pip
# below as a guard, so no dependency can silently upgrade them.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=8000

# git: for the pinned git dependencies; build-essential: in case a package has no wheel;
# libsndfile1: soundfile backend used by the TTS audio codec
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt hf_transfer "torch==2.4.1" "torchaudio==2.4.1" \
 && pip install --no-deps "parler_tts @ git+https://github.com/huggingface/parler-tts.git@d108732cd57788ec86bc857d99a6cabd66663d68"

COPY app ./app
COPY start.sh ./start.sh
RUN chmod +x start.sh

# Model cache. On RunPod, set HF_HOME to the mounted volume (/workspace/hf on a Pod,
# /runpod-volume/hf on Serverless) so ~6 GB of weights survive restarts.
# Runs as root because RunPod volumes are root-owned.
ENV HF_HOME=/models

EXPOSE 8000
# CMD (not ENTRYPOINT) so RunPod's start-command override still works.
CMD ["/app/start.sh"]
