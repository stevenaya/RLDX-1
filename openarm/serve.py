"""Serve the Pillow acceptance checkpoint through OpenArm's shared SHM socket API."""

import argparse
import json
import logging
import os
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RLDX_ATTN_IMPL", "sdpa")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--transport", choices=("socket",), default="socket")
    parser.add_argument("--socket-path", default="/dev/shm/rldx1-openarm.socket")
    parser.add_argument("--local-server", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--torch-num-threads", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warmup-sample", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--rtc-mode", choices=("none", "guided", "inpaint", "trained", "executor"), default="none")
    parser.add_argument("--rtc-guidance-steps", type=int, choices=range(1, 5), default=None,
                        help="Guided VJP steps out of this checkpoint's four denoising steps; default uses checkpoint setting")
    parser.add_argument("--rtc-inpaint-ramp-steps", type=int, choices=range(32), default=0,
                        help="Inpaint linear update-strength ramp after frozen prefix; first ramp step is also frozen")
    parser.add_argument("--rtc-margin-ms", type=float, default=10)
    parser.add_argument("--rtc-max-lateness-ms", type=float, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Check local metadata only; no model load")
    args = parser.parse_args(argv)
    if args.rtc_mode not in ("none", "guided", "inpaint"):
        parser.error("Only none, guided and inpaint are supported; trained/executor modes are unsupported")
    if args.rtc_guidance_steps is not None and args.rtc_mode != "guided":
        parser.error("--rtc-guidance-steps requires guided mode")
    if args.rtc_inpaint_ramp_steps and args.rtc_mode != "inpaint":
        parser.error("--rtc-inpaint-ramp-steps requires inpaint mode")
    if any(not math.isfinite(value) or value < 0 for value in (args.rtc_margin_ms, args.rtc_max_lateness_ms)):
        parser.error("RTC margin and maximum lateness must be finite nonnegative milliseconds")
    if args.warmup_steps < 0 or (args.warmup_steps and args.warmup_sample is None):
        parser.error("positive --warmup-steps requires --warmup-sample (recorded NPZ)")
    return args


def inspect_checkpoint(path):
    from openarm.backend import validate_manifest

    path = path.resolve(strict=True)
    if not (path / "processor").is_dir():
        raise ValueError("Expected the checkpoint's own processor/ directory")
    if not (path / "backbone" / "config.json").is_file():
        raise ValueError("Expected local checkpoint/backbone config and tokenizer assets")
    config = json.loads((path / "config.json").read_text())
    manifest = json.loads((path / "openarm.json").read_text())
    validate_manifest(manifest)
    if config.get("action_horizon") != 32 or config.get("video_length") != 1:
        raise ValueError("Expected action_horizon=32 and video_length=1 in config.json")
    if any(config.get(key) for key in ("use_memory", "use_motion", "use_physics", "physics_keys")):
        raise ValueError("Memory, motion and physics addons are unsupported")
    if config.get("rtc_inference_mode", "none") != "none":
        raise ValueError("Expected the acceptance checkpoint's saved rtc_inference_mode=none")
    return path, manifest


def main(argv=None):
    args = parse_args(argv)
    path, manifest = inspect_checkpoint(args.checkpoint_dir)
    if args.warmup_sample is not None and not args.warmup_sample.is_file():
        raise FileNotFoundError(args.warmup_sample)
    logging.basicConfig(level=logging.INFO)
    if args.dry_run:
        print(json.dumps({"checkpoint_dir": str(path), "contract": manifest,
                          "transport": "socket", "rtc_mode": args.rtc_mode,
                          "checkpoint_inference": "not performed"}, indent=2))
        return

    import cv2
    import pyarrow as pa
    import torch
    torch.set_num_threads(args.torch_num_threads)
    torch.set_num_interop_threads(args.torch_num_threads)
    cv2.setNumThreads(args.torch_num_threads)
    pa.set_cpu_count(args.torch_num_threads)
    pa.set_io_thread_count(args.torch_num_threads)
    logging.info("Inference thread pools: %s", json.dumps({
        "torch_intraop": torch.get_num_threads(),
        "torch_interop": torch.get_num_interop_threads(),
        "opencv": cv2.getNumThreads(), "arrow_cpu": pa.cpu_count(),
        "arrow_io": pa.io_thread_count(), "affinity": sorted(os.sched_getaffinity(0)),
    }, sort_keys=True))
    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.policy.rldx_policy import RLDXPolicy
    from openarm.backend import OpenArmBackend
    from openarm_policy_runtime import ModelSession, serve

    # The native policy enables executor-prior plumbing under guided. The
    # per-request option selects the actual guided or gradient-free sampler.
    rtc_options = {"rtc_inference_mode": "none" if args.rtc_mode == "none" else "guided"}
    if args.rtc_mode != "none":
        rtc_options.update(rtc_inference_delay=1, rtc_inference_exec_horizon=16)
    policy = RLDXPolicy(
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        model_path=str(path), device=args.device, strict=True,
        backbone_path=str(path / "backbone"),
        **rtc_options,
    )
    rtc = None
    if args.rtc_mode != "none":
        from openarm.rtc import ExecutionRTC
        rtc = ExecutionRTC(policy, mode=args.rtc_mode, guidance_steps=args.rtc_guidance_steps,
                           inpaint_ramp_steps=args.rtc_inpaint_ramp_steps, margin_ms=args.rtc_margin_ms,
                           max_lateness_ms=args.rtc_max_lateness_ms)
    backend = OpenArmBackend(policy, manifest, warmup_sample=args.warmup_sample,
                             warmup_steps=args.warmup_steps, prompt=args.prompt, rtc=rtc)
    try:
        backend.warmup()  # Ping readiness becomes available only after this completes.
        logging.info("Warmup paths: %s", json.dumps(backend.warmup_trace))
        if rtc is not None:
            logging.info("Mode latency calibration ms: %s", json.dumps(rtc.latency_snapshot()))
        serve(args.socket_path, lambda: ModelSession(backend), listen=args.local_server)
    finally:
        backend.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("RLDX-1 OpenArm server stopped")
