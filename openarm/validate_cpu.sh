#!/usr/bin/env bash
set -euo pipefail
: "${RLDX_SCRATCH:?}" "${RLDX_OUTPUT:?}"
python="$RLDX_SCRATCH/env/bin/python"
"$python" -m openarm.check_dataloader --base "$RLDX_SCRATCH/assets/base" --backbone "$RLDX_SCRATCH/assets/vlm" --overlay "$RLDX_SCRATCH/data/pillow0702" --output "$RLDX_OUTPUT/loader-check"
"$python" -m unittest openarm.test_backend openarm.test_local_assets openarm.test_rtc -v
export PATH="$RLDX_SCRATCH/env/bin:$RLDX_SCRATCH/bootstrap/bin:$PATH"
export UV_CACHE_DIR="$RLDX_SCRATCH/uv-cache" UV_PYTHON_INSTALL_DIR="$RLDX_SCRATCH/python"
bash openarm/setup_inference.sh "$RLDX_SCRATCH/deployment-env" /workspace/openarm-eval-workspace
bash /workspace/openarm-eval-workspace/launch_inference.sh --dry-run /workspace/openarm-eval-workspace/exp/RLDX-1/pillow500.yaml
