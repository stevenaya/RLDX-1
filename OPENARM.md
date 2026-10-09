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
default; optional executor-guided RTC has the experimental scope below.

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

The recorded eval-main baseline `7cbe6d22c726a15c8a278b80b7e2e66246236750` lacks
the downloader engine. The commands below follow its reviewed
`b70754080df8736f3c150359c302531305138d78` README and remain pending until the
user brings that engine into an eval checkout containing this manifest. This
adaptation did not merge or copy the engine. Run these from that eval checkout
on the inference machine, using an environment with Python 3.10+ and PyYAML,
plus Bash, SSH and rsync with `--protect-args` support:

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
Catalog execution, engine dry-run and actual checkpoint transfer are untested here.

Warmup uses a local recorded NPZ, not robot execution. It contains `qpos` float32
`[16]`, four `camera_<key>` uint8 RGB `[height,width,3]` arrays, and optionally a
scalar Unicode `prompt`. Keep this private sample out of Git and prepare/transfer
it explicitly on the inference machine. Store it **outside the validated download
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
itself is never published. All warmup output is discarded and the
native policy session registry is reset before socket readiness is available.
`--warmup-steps 0` explicitly skips warmup; readiness then guarantees load only,
and the first request can incur additional initialization latency.

```bash
# In the already provisioned model environment, with eval runtime importable:
python openarm/serve.py --checkpoint-dir /models/rldx1/checkpoint \
  --warmup-sample /private/rldx1/warmup.npz --warmup-steps 2 --dry-run

# Starts only a model socket, not a robot graph:
python openarm/serve.py --checkpoint-dir /models/rldx1/checkpoint \
  --warmup-sample /private/rldx1/warmup.npz --warmup-steps 2 \
  --socket-path /dev/shm/rldx1-pillow500.socket --torch-num-threads 2

# CPU/mock tests run in the isolated model environment/job:
python -m unittest openarm.test_backend openarm.test_local_assets openarm.test_rtc -v
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

`--rtc-mode none` preserves ordinary native sampling and ignores plan feedback.
`--rtc-mode guided` opts into model-local executor-aligned Jacobian guidance.
It needs `actions-executor/execution_plan` wired to `policy-server/execution_plan`
and the current shared caller's pending-plan acknowledgement protocol. It returns
full trajectories with timing metadata; the caller still selects windows.
`trained` and `executor` mode values are rejected: this acceptance has no RTC
training signal, and executor names a feedback source rather than a sampler.

The helper snapshots adopted absolute command plans, samples them on the current
observation's 30 Hz grid, rebases arms against current raw state and uses the
checkpoint's native normalization parameters. Its copied prior encoder disables
clipping only for conditioning; training/state normalization is unchanged. The
new explicit native `executor_rtc` option injects full normalized targets and
a boolean mask for 16 real channels. It bypasses legacy model-chunk caches,
including an explicit `None` bootstrap. Padded channels contribute no direct
guidance residual. This uses the existing native sampler with one optional mask.

Native output decoding still clips min-max outputs. Guided sampling attracts
the prefix through a VJP; it does not hard-inpaint it. Thus even a lossless prior
encoding does not promise identical output prefix targets or handoff continuity.
This differs from upstream inference documentation's “hard-fixing” description.

The native convention advances tau from 0 to 1. With horizon 32 and fixed native
execution horizon 16, guidance weight is 1 on `[0,F)`, tapers exponentially on
`[F,16)`, and is zero on `[16,32)`. Here F is derived from observation age plus
receipt-latency EWMA and `--rtc-margin-ms` (default 10 ms). `--rtc-max-lateness-ms`
(default 50 ms) controls timed-handoff tolerance. Both values must be finite and
nonnegative. There is no configurable ramp or compilation option.

F greater than 16 skips inference explicitly; the implementation does not
shorten measured latency to force a result. An excessive EWMA may keep skipping;
a fresh valid sample acknowledgement can update it, and an explicit
reset/restart clears it immediately. Automatic probe-based recovery is not
implemented. Short adopted plans hold their last target, matching executor
behavior. Missing feedback cold-bootstraps without model-cache fallback; stale
or mismatched-attempt feedback is skipped. Prompt/trial/attempt changes and
reset/disconnect clear both native and helper state. Warmup uses no executor
plan and clears helper state afterward; its readiness does not prove the first
guided VJP has already been exercised.

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

The delay estimate initially comes from an unguided bootstrap, while guided
sampling adds three VJPs. It still underestimated the new guided
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
