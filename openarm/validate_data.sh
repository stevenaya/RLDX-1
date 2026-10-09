#!/usr/bin/env bash
# OpenArm adaptation: CPU-only overlay, normalization and adapter validation.
set -euo pipefail
: "${RLDX_SCRATCH:?}" "${RLDX_OUTPUT:?}"
python="$RLDX_SCRATCH/env/bin/python"
overlay="$RLDX_SCRATCH/data/pillow0702"
"$python" -m openarm.data prepare --source /data/pillow-0702-filtered --modality /data/openarm-modality.json --overlay "$overlay"
"$python" -m openarm.data stats --overlay "$overlay" --scratch "$RLDX_SCRATCH/stats-tmp"
"$python" -m unittest openarm.test_data openarm.test_backend openarm.test_local_assets -v
"$python" -m openarm.data warmup --overlay "$overlay" --output "$RLDX_OUTPUT/warmup-validation-private.npz"
"$python" -m openarm.serve --help
