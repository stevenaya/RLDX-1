"""Bounded real-checkpoint SHM acceptance; recorded input only, no robot execution.

The pinned runtime has no public socket client. This harness reuses the caller's
SharedMemoryRingWriter and Session; only JSON-lines request framing is local.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

from openarm.data import CAMERAS, job_guard, write_json


def completed_formal_checkpoint(checkpoint):
    completion = json.loads((checkpoint / "save-completion.json").read_text())
    if (completion.get("complete") is not True or completion.get("formal") is not True
            or completion.get("optimizer_updates") != 500):
        raise ValueError("Inference acceptance requires a complete formal 500-update checkpoint")
    return completion


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint-dir", "warmup-sample", "eval-workspace", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--rtc-mode", choices=("none", "guided", "inpaint"), default="none")
    parser.add_argument("--rtc-margin-ms", type=float, default=10)
    parser.add_argument("--rtc-guidance-steps", type=int, choices=range(1, 5), default=None)
    parser.add_argument("--rtc-inpaint-ramp-steps", type=int, choices=range(32), default=0)
    parser.add_argument("--warmup-steps", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--repeats", type=int, choices=range(2, 5), default=2)
    parser.add_argument("--timeout-seconds", type=int, default=1100)
    parser.add_argument("--readiness-seconds", type=int, default=360)
    parser.add_argument("--request-seconds", type=int, default=180)
    args = parser.parse_args(argv)
    if not math.isfinite(args.rtc_margin_ms) or args.rtc_margin_ms < 0:
        parser.error("RTC margin must be finite nonnegative milliseconds")
    if args.rtc_guidance_steps is not None and args.rtc_mode != "guided":
        parser.error("--rtc-guidance-steps requires guided mode")
    if args.rtc_inpaint_ramp_steps and args.rtc_mode != "inpaint":
        parser.error("--rtc-inpaint-ramp-steps requires inpaint mode")
    job_guard()
    if os.environ.get("MODEL_LAB_STAGE_KIND") != "none" or os.environ.get("MODEL_LAB_MAX_STEPS") != "0":
        parser.error("Inference verification requires a registered zero-update stage")
    if not 0 < args.timeout_seconds <= 1100 or not 0 < args.readiness_seconds <= args.timeout_seconds:
        parser.error("Invalid bounded readiness/total timeout")
    if not 0 < args.request_seconds <= args.timeout_seconds:
        parser.error("Invalid bounded request timeout")
    output_root = Path(os.environ["RLDX_OUTPUT"]).resolve()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(output_root) or args.output.exists():
        parser.error("Evidence must be a fresh file under private RLDX_OUTPUT")
    args.checkpoint_dir = args.checkpoint_dir.resolve(strict=True)
    args.warmup_sample = args.warmup_sample.resolve(strict=True)
    if not args.checkpoint_dir.is_relative_to(output_root) or not args.warmup_sample.is_relative_to(output_root):
        parser.error("Checkpoint and recorded warmup must be under private RLDX_OUTPUT")
    if not (args.checkpoint_dir / "save-completion.json").is_file():
        parser.error("Checkpoint lacks its save-completion marker")
    args.checkpoint_completion = completed_formal_checkpoint(args.checkpoint_dir)
    return args


def server_command(args, socket_path):
    """Forward the candidate margin without changing serving or lateness defaults."""
    command = [sys.executable, "-m", "openarm.serve", "--checkpoint-dir", str(args.checkpoint_dir),
            "--socket-path", str(socket_path), "--warmup-sample", str(args.warmup_sample),
            "--warmup-steps", str(args.warmup_steps), "--rtc-mode", args.rtc_mode,
            "--rtc-margin-ms", str(args.rtc_margin_ms), "--rtc-max-lateness-ms", "50"]
    if args.rtc_guidance_steps is not None:
        command += ["--rtc-guidance-steps", str(args.rtc_guidance_steps)]
    if args.rtc_mode == "inpaint":
        command += ["--rtc-inpaint-ramp-steps", str(args.rtc_inpaint_ramp_steps)]
    return command


def arrow_input(sample, current=None):
    import numpy as np
    import pyarrow as pa
    position = sample["qpos"] if current is None else current
    arrays = {"position": np.asarray(position, dtype=np.float32)[None]}
    metadata = {"history_delta_indices": "0", "inference_trial_id": "checkpoint-smoke",
                "episode_attempt_id": "attempt-1", "action_window_start": 3, "action_window_size": 2}
    arrays.update({"camera_" + key: sample["camera_" + key][None] for key in CAMERAS})
    fields, names = [], []
    for key, value in arrays.items():
        value = np.ascontiguousarray(value)
        fields.append(pa.ListArray.from_arrays(pa.array([0, value.size], type=pa.int32()), pa.array(value.reshape(-1))))
        names.append(key)
        if key.startswith("camera_"):
            metadata[key + ".height"], metadata[key + ".width"] = value.shape[1:3]
    fields.append(pa.array([str(sample["prompt"].item())]))
    names.append("task_prompt")
    return pa.StructArray.from_arrays(fields, names=names), metadata


def exchange(io, request):
    from openarm_policy_runtime.logging import json_default
    io.write(json.dumps(request, default=json_default, allow_nan=False) + "\n")
    io.flush()
    line = io.readline()
    received_ns = time.time_ns()
    if not line:
        raise ConnectionError("Server disconnected before completing its response")
    response = json.loads(line)
    if response.get("error"):
        raise RuntimeError(response["error"])
    return response, received_ns


@contextmanager
def ready_connection(path, process, deadline, request_timeout, readiness_limit):
    startup_deadline = min(deadline, time.monotonic() + readiness_limit)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Server exited during startup: {process.returncode}")
            remaining = startup_deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Server readiness timed out")
            sock.settimeout(remaining)
            try:
                sock.connect(str(path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                time.sleep(min(.1, remaining))
        with sock.makefile("rw", encoding="utf-8") as io:
            reply, _ = exchange(io, {"ping": True})
            if reply.get("ready") is not True or reply.get("input_mode") != "copy":
                raise RuntimeError(f"Unexpected readiness response: {reply}")
            if reply.get("backend") != "rldx1-openarm" or reply.get("shm_slot_count") != 3:
                raise RuntimeError(f"Unexpected backend/slot contract: {reply}")
            sock.settimeout(min(request_timeout, max(.001, deadline - time.monotonic())))
            yield sock, io, reply


def summary(response):
    import numpy as np
    positions = np.asarray(response["positions"], dtype=np.float32)
    if response.get("prefill"):
        if positions.size:
            raise AssertionError("Skipped inference must not return commands")
    elif positions.shape != (32, 16) or not np.isfinite(positions).all():
        raise AssertionError(f"Expected full finite [32,16], received {positions.shape}")
    if response["interval"] != 33_333_333:
        raise AssertionError("Decoded action interval differs from the 30 Hz contract")
    return {"positions_shape": list(positions.shape), "finite": bool(np.isfinite(positions).all()),
            "positions_sha256": hashlib.sha256(positions.tobytes()).hexdigest(),
            "prefill": bool(response.get("prefill")), "reset_applied": response.get("reset_applied"),
            "interval_ns": response["interval"], "timing": response.get("timing"),
            "execution": response.get("execution"), "log_data": response.get("log_data")}


def receipt_summary(response, received_ns):
    """Observe actual receipt timing; this does not emulate bridge admission."""
    result = summary(response)
    result.update(client_received_timestamp_ns=int(received_ns),
                  generated_timestamp_ns=response.get("generated_timestamp_ns"))
    execution = response.get("execution") or {}
    takeover = int(execution.get("takeover_timestamp_ns") or 0)
    used = response.get("log_data", {}).get("rtc", {}).get("used") is True
    if used and not response.get("prefill") and takeover > 0:
        tolerance = int(execution["max_lateness_ns"])
        deadline = takeover + tolerance
        result["rtc_receipt_timing"] = {
            "scope": "timing_only; excludes outer bridge pending-ACK/admission and executor adoption",
            "receipt_minus_takeover_ms": (received_ns - takeover) / 1e6,
            "takeover_timestamp_ns": takeover,
            "max_lateness_ns": tolerance,
            "deadline_timestamp_ns": deadline,
            "observation_age_ms": (execution["inference_started_timestamp_ns"]
                                   - execution["action_origin_timestamp_ns"]) / 1e6,
            "inference_start_to_receipt_ms": (received_ns
                                              - execution["inference_started_timestamp_ns"]) / 1e6,
            "deadline_slack_ms": (deadline - received_ns) / 1e6,
            "received_by_deadline": received_ns <= deadline,
            "timing_feasible": received_ns <= deadline,
        }
        if response.get("log_data", {}).get("rtc", {}).get("mode", "guided") == "guided":
            result["guided_receipt_timing"] = result["rtc_receipt_timing"]
    # Lateness is evidence, not a failure of finite-output/VJP validation.
    return result


def acknowledge_input(response, written):
    sequence = written.descriptor["sequence"]
    if response.get("input_sequence") != sequence or sequence not in response.get("released_input_sequences", []):
        raise AssertionError("Shared input was not acknowledged and released")


def require_bootstrap(response, label, *, guided):
    if response.get("prefill"):
        raise AssertionError(f"{label} must return a full prediction")
    if guided:
        rtc = response.get("log_data", {}).get("rtc", {})
        execution = response.get("execution")
        if (rtc.get("used") is not False or not isinstance(execution, dict)
                or execution.get("based_on_chunk_id") != ""):
            raise AssertionError(f"{label} must not retain an RTC prior")


def main():
    args = parse_args()
    caller_source = args.eval_workspace.resolve(strict=True) / "nodes/dora-openarm-local-policy-server/src"
    if not caller_source.is_dir():
        raise FileNotFoundError(caller_source)
    sys.path.insert(0, str(caller_source))
    import numpy as np
    from dora_openarm_local_policy_server.shm_ring import SharedMemoryRingWriter
    from dora_openarm_local_policy_server.session import Session

    with np.load(args.warmup_sample, allow_pickle=False) as stored:
        sample = {key: stored[key].copy() for key in stored.files}
    if sample["qpos"].shape != (16,) or not np.isfinite(sample["qpos"]).all():
        raise ValueError("Warmup state must be finite [16]")
    for key in CAMERAS:
        image = sample["camera_" + key]
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError("Warmup cameras must be RGB uint8 HWC")
    observation, metadata = arrow_input(sample)
    evidence = {"complete": False, "checkpoint": str(args.checkpoint_dir), "rtc_mode": args.rtc_mode,
                "rtc_margin_ms": args.rtc_margin_ms, "rtc_max_lateness_ms": 50,
                "rtc_guidance_steps": args.rtc_guidance_steps,
                "rtc_inpaint_ramp_steps": args.rtc_inpaint_ramp_steps, "warmup_steps": args.warmup_steps,
                "robot_execution": False, "dora_graph_execution": False, "seed_controlled": False,
                "caller_scope": "SHM and Session helpers; direct requests bypass bridge pending-ACK and requires_plan admission",
                "checks": [], "input_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in sample.items()},
                "input_shapes": {key: list(value.shape) for key, value in sample.items()},
                "checkpoint_save": args.checkpoint_completion["optimizer_updates"],
                "checkpoint_save_complete": True, "checkpoint_formal": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix(".server.log")
    if log_path.exists():
        raise FileExistsError(log_path)
    started = time.monotonic()
    deadline = started + args.timeout_seconds
    error = None
    with tempfile.TemporaryDirectory(prefix="rldx-check-", dir="/dev/shm") as directory, log_path.open("x") as log:
        socket_path = Path(directory) / "policy.socket"
        command = server_command(args, socket_path)
        evidence["server_command"] = command
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        writer = None
        try:
            with ready_connection(socket_path, process, deadline, args.request_seconds, args.readiness_seconds) as (sock, io, ready):
                evidence["ready"] = ready
                evidence["startup_and_warmup_s"] = time.monotonic() - started
                writer = SharedMemoryRingWriter(directory, slot_count=ready["shm_slot_count"])
                session = Session(prompt=str(sample["prompt"].item()), reset_gap=None, action_window_start=3, action_window_size=2)

                def predict(label, *, plan=None, trial="checkpoint-smoke", reset=None, current=None):
                    local_observation, local_metadata = (observation, dict(metadata)) if current is None else arrow_input(sample, current)
                    local_metadata.update(timestamp=time.time_ns(), inference_trial_id=trial)
                    written = writer.write(local_observation, local_metadata)
                    request = session.prepare(1, local_metadata, str(sample["prompt"].item()), plan)
                    request.update(transport="shm_ring_v1", shm=written.descriptor, record_log=True)
                    if reset:
                        request["reset_reason"] = reset
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Inference smoke exhausted its time budget")
                    sock.settimeout(min(args.request_seconds, remaining))
                    call_start = time.monotonic()
                    response, received = exchange(io, request)
                    acknowledge_input(response, written)
                    row = dict(receipt_summary(response, received), label=label,
                               observation_timestamp_ns=int(local_metadata["timestamp"]),
                               wall_ms=(time.monotonic() - call_start) * 1000)
                    evidence["checks"].append(row)
                    selected = session.complete(request, response)
                    # Simulate the caller consuming a window; no robot/Dora output.
                    session.sent(selected)
                    row.update(caller_selected_steps=len(selected["positions"]),
                               caller_window_start=selected["metadata"]["action_window_start"])
                    return request, response, received, selected

                repeated = []
                first_request = first_response = first_received = first_selected = None
                for index in range(args.repeats):
                    request, response, received, selected = predict(f"ordinary_{index}")
                    if response.get("prefill"):
                        raise AssertionError("An ordinary/missing-plan call must return a full prediction")
                    repeated.append(np.asarray(response["positions"], dtype=np.float32))
                    if first_response is None:
                        first_request, first_response, first_received, first_selected = request, response, received, selected
                evidence["repeat_max_absolute_differences"] = [float(np.max(np.abs(value - repeated[0]))) for value in repeated[1:]]
                evidence["repeat_interpretation"] = "identical recorded state/images/prompt; native RNG uncontrolled, differences are not a determinism failure"

                if args.rtc_mode != "none":
                    inferred_start = first_response["execution"]["inference_started_timestamp_ns"]
                    plan = {"chunk_id": "simulated-adopted-" + first_request["chunk_id"],
                            "sample_chunk_id": first_request["chunk_id"], "start_timestamp_ns": first_received,
                            "interval_ns": first_response["interval"], "positions": first_selected["positions"],
                            "received_timestamp_ns": first_received, "inference_started_timestamp_ns": inferred_start,
                            "episode_attempt_id": "attempt-1"}
                    # Small explicit current-state change exercises prior rebase; no command leaves this harness.
                    current = sample["qpos"].copy()
                    current[:7] += .001
                    current[8:15] -= .001
                    _, guided, _, _ = predict(args.rtc_mode + "_adopted_plan_current_state_rebase", plan=plan, current=current)
                    rtc = guided.get("log_data", {}).get("rtc", {})
                    active = bool(rtc.get("used")) and not guided.get("prefill")
                    sampler = rtc.get("sampler", {})
                    if active and (sampler.get("effective_mode") != args.rtc_mode
                                   or (sampler.get("vjp_calls", 0) > 0) != (args.rtc_mode == "guided")):
                        raise AssertionError("Actual sampler trace does not match requested active RTC path")
                    evidence["active_model_exercised"] = active
                    evidence[args.rtc_mode + "_model_exercised"] = active
                    evidence["executor_plan_scope"] = "simulated offline adoption of caller-selected two-step window, measured inference/receipt timestamps; endpoint hold follows executor semantics"
                    if rtc.get("used") and guided["execution"]["based_on_chunk_id"] != plan["chunk_id"]:
                        raise AssertionError("Guidance is not based on the supplied adopted plan")
                    _, missing, _, _ = predict(args.rtc_mode + "_missing_plan_no_cache")
                    require_bootstrap(missing, "Missing executor feedback", guided=True)
                    _, withdrawn, _, _ = predict(args.rtc_mode + "_withdrawn_plan_no_cache",
                        plan={"chunk_id": "", "positions": []})
                    require_bootstrap(withdrawn, "Withdrawn executor feedback", guided=True)
                    if withdrawn["log_data"]["rtc"].get("reason") != "no_adopted_plan":
                        raise AssertionError("Withdrawn executor feedback was not recognized")
                    stale = dict(plan, inference_started_timestamp_ns=inferred_start - 1)
                    _, rejected, _, _ = predict(args.rtc_mode + "_stale_plan_rejected", plan=stale)
                    if not rejected.get("prefill") or rejected.get("execution") is not None:
                        raise AssertionError("Stale executor plan was not rejected without commands")

                _, reset_response, _, _ = predict("explicit_trial_reset", trial="checkpoint-smoke-reset", reset="trial")
                if reset_response.get("reset_applied") is not True:
                    raise AssertionError("Explicit reset was not acknowledged")
                require_bootstrap(reset_response, "Explicit reset", guided=args.rtc_mode != "none")
                session.close()
            # Disconnect must reset backend state while leaving the listener reusable.
            writer.close()
            writer = None
            with ready_connection(socket_path, process, deadline, args.request_seconds, min(30, args.readiness_seconds)) as (sock, io, ready):
                writer = SharedMemoryRingWriter(directory, slot_count=ready["shm_slot_count"])
                # Retain the latest context and supply its old pre-reset plan.
                # Without the disconnect reset, guided mode must reject that
                # plan as stale; a new context alone must not make this pass.
                reconnect_metadata = dict(metadata, timestamp=time.time_ns(),
                                          inference_trial_id="checkpoint-smoke-reset")
                written = writer.write(observation, reconnect_metadata)
                request = dict(transport="shm_ring_v1", shm=written.descriptor, metadata=reconnect_metadata,
                               chunk_id="reconnect", record_log=True, prompt=str(sample["prompt"].item()))
                if args.rtc_mode != "none":
                    request["execution_plan"] = plan
                call_start = time.monotonic()
                response, received = exchange(io, request)
                acknowledge_input(response, written)
                require_bootstrap(response, "Disconnect/reconnect", guided=args.rtc_mode != "none")
                evidence["checks"].append(dict(receipt_summary(response, received), label="disconnect_reconnect",
                    observation_timestamp_ns=int(reconnect_metadata["timestamp"]),
                    wall_ms=(time.monotonic() - call_start) * 1000))
        except Exception as exc:
            error = exc
            evidence["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            shutdown_started = time.monotonic()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    evidence["shutdown_escalated"] = True
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            if writer is not None:
                writer.close()
            evidence.update(server_exit_code=process.returncode, shutdown_s=time.monotonic() - shutdown_started,
                            socket_removed=not socket_path.exists(), elapsed_s=time.monotonic() - started,
                            server_log=str(log_path))
            evidence["complete"] = (error is None and process.returncode == 0
                                    and not socket_path.exists() and not evidence.get("shutdown_escalated"))
            write_json(args.output, evidence)
    if error is not None:
        raise error
    if not evidence["complete"]:
        raise RuntimeError("Server shutdown did not complete cleanly; see evidence")


if __name__ == "__main__":
    main()
