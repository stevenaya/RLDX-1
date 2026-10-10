#!/usr/bin/env bash
# Separate ordinary, inpaint and guided server contexts in one bounded job.
# Recorded inputs only; no robot graph. The existing harness owns checks/cleanup.
set -euo pipefail
: "${RLDX_OUTPUT:?}" "${RLDX_SCRATCH:?}"
suffix=${1:?Supply a fresh evidence suffix}
[[ "$suffix" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
for mode in none inpaint guided; do
  ramp=0
  [[ "$mode" != inpaint ]] || ramp=12
  "$RLDX_SCRATCH/env/bin/python" -m openarm.check_inference \
    --checkpoint-dir "$RLDX_OUTPUT/formal-1/checkpoint-500" \
    --warmup-sample "$RLDX_OUTPUT/warmup-validation-private.npz" \
    --eval-workspace /workspace/openarm-eval-workspace \
    --output "$RLDX_OUTPUT/rtc-modes20261010/shm-$mode-$suffix.json" \
    --rtc-mode "$mode" --rtc-inpaint-ramp-steps "$ramp" \
    --warmup-steps 2 --rtc-margin-ms 10 --timeout-seconds 330 \
    --readiness-seconds 240 --request-seconds 120
done
