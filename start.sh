#!/bin/sh
# faster-whisper (CTranslate2) looks for cuBLAS / cuDNN shared libraries on LD_LIBRARY_PATH.
# PyTorch ships them as pip packages under site-packages/nvidia/*/lib, which is not on that
# path by default - add them so Whisper can use the GPU (otherwise it falls back to CPU).
NV_LIBS=$(python - <<'EOF'
import glob, os, site
dirs = []
for sp in site.getsitepackages():
    dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
print(":".join(dirs))
EOF
)
if [ -n "$NV_LIBS" ]; then
  export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
