#!/usr/bin/env bash
# OpenArm adaptation: private Python/dependency bootstrap in a registered CPU job.
set -euo pipefail
: "${RLDX_SCRATCH:?private scratch required}"
[[ ${MODEL_LAB_MAX_STEPS:?} == 0 && ${MODEL_LAB_STAGE_KIND:?} == none ]]
export UV_CACHE_DIR="$RLDX_SCRATCH/uv-cache"
export UV_PYTHON_INSTALL_DIR="$RLDX_SCRATCH/python"
export UV_PROJECT_ENVIRONMENT="$RLDX_SCRATCH/env"
export UV_PYTHON_PREFERENCE=only-managed
export UV_PYTHON_BIN_DIR="$RLDX_SCRATCH/bin"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
export UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_INSTALLS=2
export DS_BUILD_OPS=0
mkdir -p "$RLDX_SCRATCH/bootstrap" "$RLDX_SCRATCH/evidence"
python3 -m pip install --no-cache-dir --target "$RLDX_SCRATCH/bootstrap" 'uv==0.8.24'
uvbin="$RLDX_SCRATCH/bootstrap/bin/uv"
"$uvbin" python install 3.10
"$uvbin" sync --frozen --python 3.10 --no-install-package flash-attn
"$uvbin" pip freeze --python "$RLDX_SCRATCH/env/bin/python" > "$RLDX_SCRATCH/evidence/requirements-bootstrap.txt"
"$RLDX_SCRATCH/env/bin/python" - <<'PY'
import sys, os, json
import torch, pandas, pyarrow, av
from pathlib import Path
p=Path(os.environ['RLDX_SCRATCH'])/'evidence'/'bootstrap.json'
d=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,affinity=sorted(os.sched_getaffinity(0)),av=av.__version__,pandas=pandas.__version__,pyarrow=pyarrow.__version__)
p.write_text(json.dumps(d,indent=2)+'\n');print(d)
PY
