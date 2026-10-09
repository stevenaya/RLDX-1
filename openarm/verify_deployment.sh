#!/usr/bin/env bash
# Final isolated canonical-source check; never installs in the controller.
set -euo pipefail
: "${RLDX_SCRATCH:?}"
[[ ${MODEL_LAB_MAX_STEPS:?} == 0 && ${MODEL_LAB_STAGE_KIND:?} == none && ${MODEL_LAB_GPUS:?} == 0 ]]
unset PYTHONPATH PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1
export UV_CACHE_DIR="$RLDX_SCRATCH/uv-cache" UV_PYTHON_INSTALL_DIR="$RLDX_SCRATCH/python"
export PATH="$RLDX_SCRATCH/bootstrap/bin:$PATH"
canonical=/workspace/openarm-eval-workspace/models/RLDX-1
environment="$RLDX_SCRATCH/canonical-deployment-env"
[[ ! -e "$environment" ]]
mkdir -p "$RLDX_SCRATCH/deployment-check"
cd "$RLDX_SCRATCH/deployment-check"
bash "$canonical/openarm/setup_inference.sh" "$environment" /workspace/openarm-eval-workspace
export PATH="$environment/bin:$PATH"
"$environment/bin/python" -I -B - <<'PY'
import torch, unittest
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
suite = unittest.defaultTestLoader.loadTestsFromName('openarm.test_local_assets')
if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
    raise SystemExit(1)
PY
"$environment/bin/python" -I -B "$canonical/openarm/serve.py" --help
bash /workspace/openarm-eval-workspace/launch_inference.sh --dry-run /workspace/openarm-eval-workspace/exp/RLDX-1/pillow500.yaml
