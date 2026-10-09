#!/usr/bin/env bash
# Rebuild a private inference environment using the repository's pinned uv lock.
set -euo pipefail
if [[ $# != 2 ]]; then
  echo 'Usage: bash openarm/setup_inference.sh /absolute/new/environment /absolute/eval-workspace' >&2
  exit 2
fi
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UV_PROJECT_ENVIRONMENT="$1"
eval_workspace="$2"
[[ "$UV_PROJECT_ENVIRONMENT" == /* && "$eval_workspace" == /* ]]
[[ -f "$eval_workspace/packages/openarm-policy-runtime/pyproject.toml" ]]
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 TORCHINDUCTOR_COMPILE_THREADS=2
export UV_CONCURRENT_BUILDS=1 UV_CONCURRENT_DOWNLOADS=2 UV_CONCURRENT_INSTALLS=2
export DS_BUILD_OPS=0 RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1
# Upstream lock's project version is stale (1.0.0 vs1.0.1); frozen preserves its
# exact package versions without resolving unrequested optional acceleration stacks.
uv sync --project "$repo" --frozen --python 3.10.18 \
  --no-install-project --no-install-package flash-attn
"$UV_PROJECT_ENVIRONMENT/bin/python" -I -B - "$repo" "$eval_workspace" <<'PY'
from pathlib import Path
import site, sys
# Bind pinned sources directly; neither read-only checkout needs to build a wheel.
model = Path(sys.argv[1]).resolve(strict=True)
runtime = (Path(sys.argv[2]) / 'packages/openarm-policy-runtime/src').resolve(strict=True)
(Path(site.getsitepackages()[0]) / 'openarm_model_runtime.pth').write_text(
    str(model) + '\n' + str(runtime) + '\n'
)
PY
"$UV_PROJECT_ENVIRONMENT/bin/python" -I -B - "$repo" "$eval_workspace" <<'PY'
from pathlib import Path
import json, sys
import torch
import rldx
import openarm.backend
import openarm_policy_runtime
model = Path(sys.argv[1]).resolve(strict=True)
runtime = (Path(sys.argv[2]) / 'packages/openarm-policy-runtime/src').resolve(strict=True)
sources = {}
for module, expected in ((rldx, model / 'rldx'),
                         (openarm.backend, model / 'openarm'),
                         (openarm_policy_runtime, runtime / 'openarm_policy_runtime')):
    actual = Path(module.__file__).resolve(strict=True)
    if not actual.is_relative_to(expected):
        raise RuntimeError(f'{module.__name__} resolved outside pinned source: {actual}')
    sources[module.__name__] = str(actual)
print(json.dumps({'python': sys.version, 'torch': torch.__version__,
                  'rldx': rldx.__version__, 'source_paths': sources}, indent=2))
PY
