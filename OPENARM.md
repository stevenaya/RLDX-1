# OpenArm Pillow acceptance adapter

This model-local adapter targets RLDX-1 single-frame Pillow checkpoints produced
by this adaptation. Its ordinary inference implementation uses the pinned
OpenArm runtime's `ModelSession` and SHM socket service. It is not a robot-quality
validation. Consult the run report for completed training, reload and GPU tests.

Acceptance status on 2026-10-09: **formal training completed exactly 500 optimizer
updates and strict checkpoint reload passed**. Fresh official-base run
Slurm job 12531 recorded consecutive finite loss/gradient updates on both GPUs,
saved checkpoint-500 with complete/formal=true evidence, and exited with code 0.
Slurm job 12532 then strictly reloaded model weights, each rank's actual AdamW
moment states at step 500, and the separately saved HF scheduler without
additional updates, and exited with code 0.
RNG was not restored; this does not claim bitwise continuation. Ordinary
real-checkpoint SHM inference passed in Slurm job 12533. Guided sampling and its
offline numerical checks ran successfully. The original guided handoff missed
its deadline with the default 10 ms margin; one authorized follow-up passed
with an explicit 150 ms margin and unchanged 50 ms lateness allowance. Guided
mode remains experimental; these short probes do not establish executor
readiness or sustained timing. Serving defaults remain mode `none`, margin
10 ms and lateness 50 ms. The original evidence is under
model-lab `runs/rldx1-pillow500-20261009/`, including
`formal-1-training-evidence.json`, `formal-1-checkpoint-500-save-completion.json`
and `formal-reload-evidence.json`.

The 2026-10-10 follow-up adds separate inpaint sampling, guidance-step selection
and active-prior warmup with mode-specific timing calibration. These changes
reuse the same checkpoint. CPU regressions, native GPU comparisons and final
ordinary/inpaint/guided SHM checks passed, including one calibrated receipt per
RTC mode at the unchanged 10 ms margin and 50 ms lateness. The support matrix
below separates these short probes from the historical acceptance evidence.
Follow-up evidence belongs in model-lab `runs/rldx1-rtc-modes-20261010/`.

The scoped SDPA fix preserves the original 4D mask for single-frame identity
token selection and passed CPU numerical tests plus real GPU training. Genuine
4D-mask video/motion compression remains unsupported. Effective batch 4 is a
verified feasible choice; larger microbatches were not compared, so it is not
claimed optimal. This short acceptance does not establish task or robot quality.

The checkpoint directory must contain the native HF model/config, its own
`processor/` directory, `backbone/` config/tokenizer assets and producer's
`openarm.json`. No independent stats
override is accepted. The loader uses `GENERAL_EMBODIMENT` and validates the
loaded processor's four cameras, joint groups, dimensions and representations.
It rejects history, memory, motion and physics. Ordinary inference is the
default; optional guided and inpaint RTC have the experimental scope below.

Inputs select `Observation.current_index()`, then pass one RGB uint8 frame per
camera to the native processor in this order: wrist_right, wrist_left, ceiling,
head_left. Runtime field names have the `camera_` prefix. State order is right arm
7, right gripper 1, left arm 7, left gripper 1. Native checkpoint decoding returns
absolute targets from the relative-arm/absolute-gripper training representation.
The backend returns all 32 finite targets at 30 Hz action spacing; it does not
crop windows or change units. The verified collector contract uses joint radians,
including signed gripper motor positions; these are not normalized gripper widths. The caller controls inference rate, selection,
chunk identity, attempt boundaries and logging. Actual units and sample evidence
belong in the producer's metadata and run report.

## Deployment and startup

On Linux x86_64 with a compatible NVIDIA driver for PyTorch CUDA12.6, use
uv0.8.24 to rebuild the repository's pinned Python3.10.18 environment:

```bash
# From eval's canonical models/RLDX-1 checkout. Supply absolute paths.
bash openarm/setup_inference.sh "$PWD/.venv" "$(realpath ../..)"
export RLDX_ATTN_IMPL=sdpa NO_ALBUMENTATIONS_UPDATE=1
```

The setup uses the existing exact uv lock with `--frozen` because upstream's
lock project version is1.0.0 while source metadata is1.0.1. It skips flash-attn
and uses upstream's supported SDPA route. `--no-install-project` avoids an
editable build writing into the model checkout. An environment-local `.pth`
binds the model checkout and supplied eval runtime directly, including the
model's `openarm` package. RLDX reports its version from `rldx.__version__`;
no installed RLDX distribution metadata is required by this serving entrypoint.
The setup's `python -I` import check ignores inherited PYTHONPATH/current-directory
imports and asserts all three module paths resolve inside those exact sources.
CUDA toolkit compilation and compiled
inference are not required or claimed. The CPU job verified this dependency
set's imports; consult run evidence for GPU and fresh deployment setup checks.
Never copy a training virtualenv or depend on an unmounted training cache. The
setup binds the supplied eval checkout's pinned runtime to this environment;
the workspace launcher also supplies that runtime's source path. This adapter supports socket SHM only. Direct
Dora, input borrowing and compilation are not exposed.

Transfer the actual HF export, its entire required processor/config tree,
`backbone/` tokenizer/image-processor/config files and `openarm.json` through the
eval download manifest. The explicit local `backbone_path` loader option binds
both model and processor to those assets with `local_files_only=True`; saved
training paths or Hub names are not used for deployment. The full RLDX export
already supplies backbone weights, so a second full VLM weight copy is unnecessary.
The explicit local loader rejects any reported missing, unexpected or mismatched
weight keys and loading errors, and records its diagnostic summary in the server
log. Serving-load diagnostics are separate from the completed formal
model/optimizer/scheduler reload. Code is Apache-2.0. The official checkpoint and
its fine-tuned derivatives use the **RLWRLD Model License v1.0, restricted to
noncommercial use**; the export preserves that license as `checkpoint/LICENSE.md`
and retains the backbone license. Do not publish weights or private data. This
code adaptation grants no commercial model rights.

## Checkpoint download and private warmup

The eval manifest is
`tools/download_models/manifests/RLDX-1/rldx1-pillow500-20261009.yaml`. Its step500
selection matches the actual producer inventory: **22 files, 13,830,985,921 bytes**
(about 12.88 GiB), including the completion marker. These are three HF weight
shards and their index, model config, `openarm.json`, licenses, the checkpoint's
three processor files, and eleven backbone config/tokenizer/license files.
The explicit selection excludes DeepSpeed model/optimizer partitions, scheduler,
RNG, trainer state and other training-resume artifacts. It is an inference bundle,
not a training-resume checkpoint. `formal-manifest-inventory.json` records the
selection evidence; no downloader transfer receipt exists yet.

The original eval-main baseline `7cbe6d22c726a15c8a278b80b7e2e66246236750` lacked
the downloader engine. The user's separately authorized download-only merge
has since brought it into the maintained integration checkout. The local
`--latest --dry-run` verified step 500, its marker and all 22 selected files on
2026-10-10 (Asia/Tokyo); actual weight transfer remains untested. Run the commands
below from that eval checkout on the inference machine, using Python 3.10+ with
PyYAML, Bash, SSH and rsync with `--protect-args` support:

```bash
# Offline catalog, then remote candidates for this run.
bash tools/download_models/download.sh list
bash tools/download_models/download.sh list RLDX-1/rldx1-pillow500-20261009

# Inspect the exact step500 selection before transferring weights.
bash tools/download_models/download.sh RLDX-1/rldx1-pillow500-20261009 \
  --step 500 --target /hdd_data/models --dry-run

# Transfer through eval's existing engine; bandwidth limit is KiB/s.
bash tools/download_models/download.sh RLDX-1/rldx1-pillow500-20261009 \
  --step 500 --target /hdd_data/models --bwlimit 51200
```

Configure the private `KDM00` SSH alias on the inference machine so the manifest's
NAS source path is visible. Keep credentials in SSH configuration and preserve
host-key verification. The engine's `--host ALIAS` can select another verified
source namespace; it does not rewrite paths. The expected destination is
`/hdd_data/models/RLDX-1/rldx1-pillow500-20261009/500/`. Use its printed bindings
to set YAML `assets_dir` to that bundle and `checkpoint_dir` to its `checkpoint/`
subdirectory. The engine validates the selected file set and writes
`.download-receipt.json`; it does not edit experiment YAML or prove GPU inference.
Source inventory/dry-run evidence is separate from a destination transfer
receipt. No actual checkpoint transfer is claimed here.

Warmup uses a local recorded NPZ, not robot execution. It contains `qpos` float32
`[16]`, four `camera_<key>` uint8 RGB `[height,width,3]` arrays, and optionally a
scalar Unicode `prompt`. Match deployment camera sizes, state and task prompt;
prompt/token-length changes can change runtime even when camera sizes match.
The NPZ prompt takes precedence; `--prompt` is only a fallback when the NPZ has
no prompt. Keep this private sample out of Git and prepare/transfer it explicitly
on the inference machine. Store it **outside the validated download
bundle**: adding an undeclared file there would change the engine's validated
file set. For the acceptance's existing validation sample, this example transfers
exactly one private file through the verified SSH alias; it does not transfer
weights or copy dataset directories:

```bash
umask 077
warmup_dir=/hdd_data/openarm-private/rldx1-pillow500
mkdir -p "$warmup_dir"
rsync --protect-args --partial --bwlimit=51200 \
  -e 'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes' \
  KDM00:/mnt/syno127/volume1/stevenaya/model-lab/rldx1-pillow500-20261009/warmup-validation-private.npz \
  "$warmup_dir/warmup-validation-private.npz"
chmod 600 "$warmup_dir/warmup-validation-private.npz"
```

Set YAML `warmup_sample` to
`/hdd_data/openarm-private/rldx1-pillow500/warmup-validation-private.npz` explicitly.
This one-file transfer is guidance, not an executed transfer receipt. The sample
itself is never published. All warmup output is discarded and the native policy
registry plus synthetic plan/episode identities are cleared before socket
readiness. Mode-specific latency calibration survives that cleanup. For RTC,
each warmup round includes both bootstrap and an active simulated prior;
`--warmup-steps 2` therefore executes four sampler calls. Ordinary mode executes
one call per round. Warmup timing uses the latter half of each path's samples,
excluding its first when there is more than one; two rounds provide only one
calibration sample per path. A single round is labeled a single cold sample.
`--warmup-steps 0` explicitly skips warmup; readiness then guarantees load only,
with no active-path calibration. Ordinary bootstrap remains available, but an
adopted-prior RTC request is skipped with `latency_uncalibrated` until its matching
active path is calibrated. Changed camera shapes can cause the same explicit skip;
restart with a representative warmup sample. Only discarded local warmup calls
use the internal `calibration_only` path; socket metadata cannot enable it.
The first request can incur additional initialization latency. These small warmup samples do not establish sustained
latency percentiles.

```bash
# In the already provisioned model environment, with eval runtime importable:
python openarm/serve.py --checkpoint-dir /models/rldx1/checkpoint \
  --warmup-sample /private/rldx1/warmup.npz --warmup-steps 2 --dry-run

# Starts only a model socket, not a robot graph:
python openarm/serve.py --checkpoint-dir /models/rldx1/checkpoint \
  --warmup-sample /private/rldx1/warmup.npz --warmup-steps 2 \
  --socket-path /dev/shm/rldx1-pillow500.socket --torch-num-threads 2

# CPU/mock tests run in the isolated model environment/job:
python -m unittest openarm.test_backend openarm.test_local_assets openarm.test_rtc \
  openarm.test_rtc_sampler_modes openarm.test_check_rtc_modes -v
```

`--checkpoint-dir` must be a local export; Hub download/credential resolution is
not a serving option. The native denoising schedule and count remain those of the
checkpoint. `--dry-run` inspects metadata only; it does not prove weights load,
processor compatibility or inference. Warmup and model load finish before the
shared runtime answers its readiness ping. Explicit reset/disconnect clears the
native session registry; close clears it and releases the policy reference.
`--torch-num-threads` accepts 1 or 2 (default 2) and caps Torch intra-op/inter-op,
Arrow CPU/I/O and OpenCV pools in the actual server process. Startup logs their
measured API counts and process CPU affinity. Keep the wrapper/YAML's library,
decoder and compiler environment caps when launching.

Ordinary checkpoint inference in Slurm job 12533 produced four full finite
`[32,16]` absolute-action responses over SHM. Strict local loading reported zero
missing, unexpected or mismatched keys and zero loading errors. Startup plus
one discarded recorded-sample warmup took 36.274 seconds; the four local call
times were 151.028–167.370 ms. Input acknowledgement/release, caller-selected
windows, explicit reset, disconnect/reconnect, clean shutdown and socket removal
passed. These smoke timings are not a sustained deployment benchmark. Native
RNG was uncontrolled for those calls; repeated outputs need not match. See
`inference-none-1.json` and its server log in the run evidence directory.

## RTC assessment

The 2026-10-10 implementation has the following scope. CPU job 12546 passed all
59 tests, including missing/changed-shape calibration and actual processor checks
without weights. GPU job 12544 passed 37 calls on the existing checkpoint-500;
12545 passed ordinary/inpaint SHM. Final job 12547 passed ordinary/inpaint/guided
serving after the calibration-miss guard. Historical receipts below retain their
original scope. Training and the checkpoint are unchanged.

| Capability | Status and limits |
| --- | --- |
| `none` | Default ordinary native sampling. Full server context and four finite `[32,16]` SHM responses passed in each of 12545/12547. |
| `guided` | Actual 1/3/4 VJPs out of four denoise steps verified in 12544; omitted count matched native 3 exactly. One calibrated guided3 SHM deadline passed in 12547. Prefix remains soft. |
| `inpaint` | R0/R12 numerics and active warmup passed in 12544. R12 prediction-derived priors and recorded deadlines passed in 12545/12547; inference-only conditioning. |
| Active-prior warmup | Actual branch/VJP traces verified, including the final guard revision. Synthetic state clears while separate calibration survives. |
| Mode-specific timing | Separate estimates verified; missing/changed-shape calibration rejects in CPU tests. No sustained deadline or real executor-adoption claim. |
| `trained`, `executor` | Rejected. This checkpoint has no positive training RTC delay; executor is a feedback source, not a sampler. |
| Dora graph, robot/task quality | Untested. The recorded SHM harness bypasses outer bridge pending-ACK/admission. |

Both RTC modes need `actions-executor/execution_plan` wired to
`policy-server/execution_plan` and the shared caller's acknowledgement protocol.
The helper samples adopted absolute commands on the current observation's 30 Hz
grid, rebases arms against current raw state, and applies checkpoint statistics.
Short plans hold their last target. The scoped prior encoder disables clipping
for conditioning; training/state normalization and native output decoding are
unchanged. An explicit target and mask cover the 16 real command channels;
padded channels contribute no direct guidance residual or inpaint constraint.
Missing plans bootstrap without cached-model-chunk fallback. The backend always
returns the full trajectory; caller windows and executor adoption remain outside it.

`guided` advances the native tau 0→1 schedule and applies soft attraction with
weight 1 on `[0,F)`, exponential taper on `[F,16)`, and zero on `[16,32)`.
It does not hard-freeze the prefix. `--rtc-guidance-steps` selects 1..4 VJP steps
out of this checkpoint's four denoising steps. Omitting it preserves the native
checkpoint setting, currently 3. This option is accepted only with guided mode.
More guidance work or smaller prior error does not establish better task quality.

`inpaint` injects the normalized prior into masked prefix and ramp positions.
Euler update strength is 0 on `[0,F)`, `j/R` at ramp offset `j=0..R-1`, and 1
afterward and on padded channels. Thus the first ramp point is also frozen.
`--rtc-inpaint-ramp-steps R` defaults to 0 and accepts 0..31 only with inpaint mode;
the current prefix plus ramp must fit horizon 32. This preserves the native
token times; it does not force clean-prefix `t=1` or claim trained RTC support.
The normalized frozen target is BF16-quantized. Exact FP32 absolute-prefix
equality is not promised: checkpoint scaling can amplify quantization error.
In 12544, F4 frozen normalized values and every Euler update matched exactly;
the decoded prefix differed from its original FP32 absolute prior by at most
0.000283063 rad. R12 ramp/free output clipping changed values by up to 0.095555/
0.106172 rad relative to the unclipped inverse. A synthetic +10 rad rebase was
rejected; native clipping would have moved its frozen target by up to 9.897142 rad.

Native output decoding still clips min-max outputs. Inpaint rejects a frozen
prefix that cannot survive that decoder, including the first zero-strength
ramp point, and returns no commands for that request. Out-of-range rebased
targets are not silently accepted as an exact absolute prefix. Decoder ranges
are normalization statistics, not actuator safety bounds. Guided is soft and
may produce different decoded prefix targets even with lossless prior encoding.

The scheduling prefix F includes observation age, the matching active-path
latency estimate and `--rtc-margin-ms` (default 10 ms). Guided requires F<=16;
inpaint requires F<=min(31,32-R). Unsupported delays skip without shortening the
estimate. `--rtc-max-lateness-ms` stays 50 ms by default; increasing scheduling
margin moves takeover later and can lengthen the constrained prefix, without
accelerating the model or relaxing lateness. Automatic recovery probes are not
implemented.

Warmup runs representative recorded images/state/prompt through bootstrap and
the selected active-prior path, using a local simulated normalized-center plan.
The server logs `Warmup paths` and `Mode latency calibration ms`; requested mode
alone is not proof the path ran. It records actual denoising/VJP counts and clears
synthetic plans, context and identities before readiness. Calibration remains
separate by effective mode, denoising/guidance/ramp settings, device/native
configuration and camera shapes. The outer configuration label in a key is not
an independently measured kernel identity. **Prompt/token length is not part of the calibration
key**: use a representative matching task prompt, and remeasure for changed
prompt workloads. Camera-shape equality alone does not establish timing coverage.

Fresh valid acknowledgements update the path that generated the identified
sample; bootstrap feedback cannot overwrite active guided/inpaint calibration.
The bounded identity records reject stale, untracked or mismatched feedback.
Prompt/trial/attempt changes and reset/disconnect clear episode state and native
caches while retaining path-specific latency calibration. A new process or
internal `reset(clear_calibration=True)` clears calibration. No estimate guarantees
the next receipt will meet its deadline.

In an independently provisioned model environment on an authorized inference
machine, these commands start only a socket server. They are experimental mode
examples, not recorded test results:

```bash
# Guided: explicit 3 matches this checkpoint's default; compare 1/3/4 before choosing.
python openarm/serve.py --checkpoint-dir /models/rldx1/checkpoint \
  --warmup-sample /private/rldx1/warmup.npz --warmup-steps 2 \
  --rtc-mode guided --rtc-guidance-steps 3 --rtc-margin-ms 10 \
  --rtc-max-lateness-ms 50 --socket-path /dev/shm/rldx1-guided.socket

# Inpaint: optional 12-step ramp; omit the flag for default 0.
python openarm/serve.py --checkpoint-dir /models/rldx1/checkpoint \
  --warmup-sample /private/rldx1/warmup.npz --warmup-steps 2 \
  --rtc-mode inpaint --rtc-inpaint-ramp-steps 12 --rtc-margin-ms 10 \
  --rtc-max-lateness-ms 50 --socket-path /dev/shm/rldx1-inpaint.socket
```

The one-load `openarm.check_rtc_modes` harness compares ordinary, inpaint with
and without a ramp, and guided 1/3/4 at matched inputs/RNG and total 4 denoising
steps. It records per-step normalized and decoded errors, real/padded-mask
behavior, clipping rejection, VJP traces, timing/memory and production warmup.
Instrumentation and earlier settings can warm shared kernels; its first/repeat
samples are not a sustained benchmark. Execute it only in a registered bounded
model job. Its ordinary control uses a guided-configured policy with no prior
and `no_grad`; the full ordinary server's `inference_mode` path was checked
separately over SHM. Original logs, exact timestamps and measured limits are in
model-lab `runs/rldx1-rtc-modes-20261010/`.

The matched-input/RNG instrumented pipeline repeats measured 154.648 ms for
inpaint R0, 164.273 ms for R12, and 191.544/239.175/260.938 ms for guided 1/3/4.
There were only two samples per setting, in fixed order; earlier calls could
warm shared kernels. The first ordinary control took 4982.050 ms, including
680.489 ms preprocessing and 4300.638 ms native work. These are whole-path
measurements, not isolated backward-kernel costs. The fourth VJP at tau 0.75
was nonzero (maximum 0.0966796875); three guidance steps are a retained native
baseline, not a universal optimum. No preset is preferred from these timings
or its error to a single prior.

Final job 12547 used two warmup rounds, 10 ms margin and 50 ms lateness:

| Actual server path | Recorded call wall | Active estimate | Prefix / ramp or taper | Receipt before deadline |
| --- | ---: | ---: | --- | ---: |
| Ordinary | 154.095–155.862 ms, four calls | Not used | None | Not timed RTC |
| Inpaint R12 | 157.473 ms, one active call | 152.198 ms | F6 / ramp12 | 85.414 ms |
| Guided native3 | 239.347 ms, one active call | 233.376 ms | F8 / taper8 | 70.519 ms |

The guided backend-start-to-receipt interval was 237.501 ms, only 4.125 ms
above its active warmup estimate. Ordinary feedback did not replace that
estimate. This addresses the previous ordinary-to-guided underestimate below;
VJP work still costs time after warmup. Both active priors came from actual
preceding caller-selected predictions and were rebased to changed current state.
Missing/withdrawn plans, stale rejection, reset/reconnect, SHM release and clean
shutdown passed. These short offline probes do not establish estimator
convergence, sustained deadlines, bridge admission or real executor adoption.

### Historical guided evidence before the warmup/calibration rework

Guided checkpoint SHM testing in Slurm job 12534 exercised the native VJP path
with a rebased adopted-plan target, prefix F=6 and horizon 32, and returned finite
full `[32,16]` actions. Missing/withdrawn plans, stale-plan rejection and
reset/disconnect checks passed. However, the guided call took 321.298 ms and
arrived **130.057 ms after its takeover timestamp**, exceeding the allowed
50 ms lateness. Its protocol/lifecycle completion does not mean RTC timing
passed: `inference-guided-1.json` records `timing_feasible=false`. This original
failure remains part of the evidence.

The single authorized follow-up, Slurm job 12539, used the same checkpoint and
input hashes with explicit `--rtc-margin-ms 150` and unchanged
`--rtc-max-lateness-ms 50`. Its first guided call took 270.220 ms and arrived
**56.084627 ms before takeover**, leaving **106.084627 ms before the deadline**.
The 170.834542 ms bootstrap estimate, 8.990174 ms observation age and 150 ms
margin produced F=10, six taper steps and 16 free steps. It returned finite full
`[32,16]` actions; missing/withdrawn plans, stale-plan rejection, reset/reconnect
and clean shutdown checks also passed. See `inference-guided-margin150-1.json`
and `rtc-margin-comparison.json` in the run evidence directory.

In that historical implementation, warmup had no effective prior and exercised
no guided VJP. Its delay estimate came from an unguided bootstrap, while guided
sampling added three VJPs. It still underestimated the follow-up guided
backend-start-to-receipt duration by 97.423987 ms; the explicit larger margin
provided enough lead time for this call. The new call also ran faster, native
RNG was uncontrolled and prefix length differed, so these receipts do not isolate a
latency improvement caused by margin. Ordinary mode, 10 ms margin and 50 ms
lateness remain the defaults. Guided mode remains experimental: one short
timing pass does not establish sustained deadlines, estimator convergence or
real executor adoption.

The separate Slurm job 12535 numerical probe completed six native-sampler calls,
including one discarded warmup, with matched RNG states and identical processed
inputs for the paired comparisons. Baseline repeats, zero-mask versus baseline,
and guided repeats had exactly zero output differences in this probe. Real-channel
guidance changed normalized actions by at most 0.0234375, above the measured zero
repeatability floor. Each guided call exercised three finite VJPs; padded-channel
residuals were zero. All calls produced finite normalized `[1,32,64]` and decoded
`[32,16]` outputs. Peak allocated GPU memory was 17,127,150,080 bytes (about 15.95 GiB).
See `rtc-numerics-1.json` for the recorded errors and traces. This was offline
F=1 conditioning, without executor timestamps; it does not validate handoff
timing, prove general determinism, or guarantee a frozen prefix or improved
task quality. The two SHM timing receipts above are separate evidence.

The SHM harness uses the caller's Session and SHM helpers; its direct requests
bypass the outer bridge's pending-ACK and `requires_plan` admission. No complete
Dora graph or physical robot ran. CPU checks, real-checkpoint ordinary inference,
guided sampler numerics and executor timing are separate evidence scopes.
Physical robot and task quality remain untested; trained RTC remains unsupported.

## Final canonical-source deployment check

Consult the final integration report at model-lab
`runs/rldx1-pillow500-20261009/report.md` for the canonical-source verification
result and published model/eval commits. After the fork commit is pinned into
eval, run a registered zero-GPU, zero-update
job with the eval workspace mounted read-only, private scratch/output paths and
the same experiment budget ID. The wrapper `openarm/verify_deployment.sh` requires
the registered stage's zero-GPU/zero-update declarations, unsets inherited
`PYTHONPATH` and `PYTHONHOME`, and refuses an existing verification environment.
It invokes setup from private scratch, checks exact canonical import paths using
`python -I`, runs the local-asset CPU tests, checks serving CLI help and renders
the eval launcher. Invoke it only inside that registered model job, with its
private `RLDX_SCRATCH` already set:

```bash
unset PYTHONPATH PYTHONHOME
bash /workspace/openarm-eval-workspace/models/RLDX-1/openarm/verify_deployment.sh
```

Capture setup's asserted module paths and test/render output in the run evidence.
Do not execute model setup or tests in the authenticated controller. The private
`$RLDX_SCRATCH/canonical-deployment-env` is for this isolated verification job;
an inference machine should build its own environment at the YAML's configured
`model_environment`. This zero-GPU check does not load checkpoint tensors or
execute the YAML's GPU command. It establishes neither checkpoint transfer nor
GPU inference, executor RTC timing or physical-robot quality.
