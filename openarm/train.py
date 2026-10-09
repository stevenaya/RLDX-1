"""Bounded OpenArm acceptance using the native RLDX model, processor and trainer."""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import time
import resource

from openarm.data import BASE_REVISION, CAMERAS, GROUPS, HORIZON, PillowDataset, job_guard, modality_config, write_json


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["tune", "formal", "reload"])
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-batch", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--tune-steps", type=int, default=12)
    parser.add_argument("--top-llm-layers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--decoder", choices=["pyav", "torchcodec"], default="pyav")
    return parser.parse_args()


def make_processor(base, backbone, dataset):
    from rldx.model.core.processing_rldx import RLDXProcessor
    kwargs = json.loads((base / "processor/processor_config.json").read_text())["processor_kwargs"]
    kwargs.update(model_name=str(backbone), modality_configs={"general_embodiment": modality_config()},
                  statistics=dataset.get_dataset_statistics(), max_action_horizon=HORIZON,
                  memory_length=1, general_embodiment_train_ratio=0, use_relative_action=True,
                  apply_sincos_state_encoding=False, physics_keys=[], physics_dims=[])
    kwargs["embodiment_id_mapping"] = json.loads((base / "processor/embodiment_id.json").read_text())
    return RLDXProcessor(**kwargs, transformers_loading_kwargs={"local_files_only": True})


def export_metadata(checkpoint, processor, backbone, base, evidence):
    processor.save_pretrained(checkpoint / "processor")
    shutil.copyfile(base / "LICENSE.md", checkpoint / "LICENSE.md")
    experiment = checkpoint.parent / "experiment_cfg"
    if experiment.is_dir():
        shutil.copytree(experiment, checkpoint / "experiment_cfg", dirs_exist_ok=True)
    assets = checkpoint / "backbone"
    assets.mkdir(exist_ok=True)
    for source in backbone.iterdir():
        if (source.is_file() and source.suffix in (".json", ".txt", ".jinja", ".md")
                and source.name != "model.safetensors.index.json"):
            shutil.copyfile(source, assets / source.name)
    # Saved references are public provenance, never a silent dependency on training paths.
    # The OpenArm backend resolves these explicitly to checkpoint/backbone at load time.
    config = json.loads((checkpoint / "config.json").read_text())
    config["model_name"] = "RLWRLD/RLDX-1-VLM"
    write_json(checkpoint / "config.json", config)
    proc = json.loads((checkpoint / "processor/processor_config.json").read_text())
    proc["processor_kwargs"]["model_name"] = "RLWRLD/RLDX-1-VLM"
    write_json(checkpoint / "processor/processor_config.json", proc)
    contract = {"format_version": 1, "horizon": HORIZON, "action_hz": 30,
                "video_keys": CAMERAS, "state_action_keys": list(GROUPS),
                "group_dims": [end - start for start, end in GROUPS.values()], "absolute_command": True,
                "training_semantics": "current-state-relative arms and absolute grippers",
                "units": "joint radians, including signed gripper motor angles",
                "units_evidence": "current collector/driver contract and sampled converted data; historical converter revision unavailable",
                "base_model": "RLWRLD/RLDX-1-PT-IMG", "base_revision": BASE_REVISION,
                "backbone_revision": "4b9f870d1287e0d38d7eb1445e6d8c60afe66dd7",
                "rtc": "ordinary default; optional guided adapter experimental; trained RTC unsupported (training max delay zero)",
                "optimizer_updates": evidence["optimizer_updates"]}
    write_json(checkpoint / "openarm.json", contract)
    files = [{"path": str(path.relative_to(checkpoint)), "bytes": path.stat().st_size}
             for path in sorted(checkpoint.rglob("*")) if path.is_file()]
    if not any(item["path"].endswith(".safetensors") for item in files):
        raise RuntimeError("No model safetensors export was saved")
    write_json(checkpoint / "save-completion.json", dict(evidence, files=files, complete=True))


def build(args):
    import torch
    import rldx  # Register the native HF model and processor.
    from transformers import TrainingArguments, set_seed
    from rldx.configs.base_config import Config
    from rldx.configs.model.rldx import RLDXConfig
    from rldx.model.core.setup import RLDXPipeline
    from rldx.experiment.trainer import RLDXTrainer
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if not 0 <= args.workers * world <= 2 or args.accumulation < 1 or args.effective_batch % (world * args.accumulation):
        raise ValueError("Total workers must be <=2; effective batch must divide world_size × accumulation")
    if not 1 <= args.tune_steps <= 12:
        raise ValueError("Tuning is bounded to 1..12 optimizer updates")
    steps = 500 if args.command in ("formal", "reload") else args.tune_steps
    cfg = Config()
    cfg.model = RLDXConfig.from_pretrained(args.base, local_files_only=True)
    cfg.model.model_name = str(args.backbone)
    cfg.model.model_revision = None
    cfg.model.action_horizon = HORIZON
    cfg.model.video_length = 1
    cfg.model.use_memory = cfg.model.use_motion = cfg.model.use_physics = False
    cfg.model.use_relative_action = True
    cfg.model.general_embodiment_train_ratio = 0
    cfg.model.rtc_training_max_delay = 0
    cfg.model.rtc_inference_mode = "none"
    cfg.model.tune_top_llm_layers = args.top_llm_layers
    cfg.model.tune_llm = cfg.model.tune_visual = False
    cfg.model.tune_diffusion_model = cfg.model.tune_projector = True
    # Preserve IMG's state_dropout_prob=0: enabling it adds a fresh mask_token.
    cfg.training.start_from_checkpoint = str(args.base)
    cfg.training.transformers_local_files_only = True
    cfg.training.num_gpus = world
    cfg.data.modality_configs = {"general_embodiment": modality_config()}
    deepspeed = cfg.get_deepspeed_config() if world > 1 else None
    training_args = TrainingArguments(
        output_dir=str(args.output), max_steps=steps,
        per_device_train_batch_size=args.effective_batch // (world * args.accumulation),
        gradient_accumulation_steps=args.accumulation, learning_rate=1e-4,
        lr_scheduler_type="cosine", warmup_steps=25 if steps == 500 else min(2, steps),
        weight_decay=1e-5, max_grad_norm=1.0, optim="adamw_torch_fused",
        bf16=True, tf32=True, logging_steps=1, save_steps=steps, save_total_limit=1,
        dataloader_num_workers=args.workers, remove_unused_columns=False,
        report_to="none", deepspeed=deepspeed, ddp_find_unused_parameters=False, seed=42,
        logging_nan_inf_filter=False, save_safetensors=True,
    )
    set_seed(42)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    model_config_dir = args.output / "experiment_cfg"
    model_config_dir.mkdir(exist_ok=True)
    model = RLDXPipeline(cfg, model_config_dir)._create_model()
    if getattr(model, "_new_param_names", set()):
        raise RuntimeError(f"Unexpected fresh parameters loading IMG base: {sorted(model._new_param_names)}")
    dataset = PillowDataset(args.overlay / "train", decoder=args.decoder)
    processor = make_processor(args.base, args.backbone, dataset)
    processor.train()
    dataset.processor = processor

    class FiniteTrainer(RLDXTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
            loss = result[0] if return_outputs else result
            if not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite loss before optimizer update {self.state.global_step + 1}")
            return result

    trainer = FiniteTrainer(model=model, args=training_args, train_dataset=dataset,
                            data_collator=processor.collator, multiprocessing_context="spawn")
    return trainer, processor, cfg, steps


def reload_checkpoint(args):
    """Strict model/optimizer/scheduler reload without executing an optimizer update."""
    import torch
    from transformers import AutoConfig
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    checkpoint = args.checkpoint
    if checkpoint is None:
        raise ValueError("reload requires --checkpoint")
    completion = json.loads((checkpoint / "save-completion.json").read_text())
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    if not completion["complete"] or state["global_step"] != completion["optimizer_updates"]:
        raise ValueError("Saved completion evidence disagrees with trainer state")
    if completion.get("formal") and state["global_step"] != 500:
        raise ValueError("Formal acceptance checkpoint must contain exactly 500 updates")
    # Native HF load reports all key discrepancies; no permissive fallback is allowed.
    import rldx
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config.model_name = str(args.backbone)
    from rldx.model.core.rldx import RLDX
    from transformers.modeling_utils import load_sharded_checkpoint
    # Native training construction preserves FP32 trainable parameters and BF16
    # frozen parameters. Forcing every parameter to BF16 would change optimizer reload.
    model = RLDX(config, transformers_loading_kwargs={"skip_pretrained_weights": True, "local_files_only": True})
    if (checkpoint / "model.safetensors.index.json").exists():
        load_sharded_checkpoint(model, str(checkpoint), strict=True, prefer_safe=True)
    else:
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(checkpoint / "model.safetensors")), strict=True)
    # Build the same native HF optimizer grouping/scheduler without starting its train loop.
    from rldx.experiment.trainer import RLDXTrainer
    from transformers import TrainingArguments
    saved = json.loads((checkpoint.parent / "acceptance-args.json").read_text())
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world != saved["world_size"]:
        raise ValueError("Optimizer reload requires the same GPU count as the saved run")
    train_args = TrainingArguments(output_dir=str(args.output), learning_rate=1e-4, weight_decay=1e-5,
        optim="adamw_torch_fused", lr_scheduler_type="cosine", warmup_steps=saved["warmup_steps"],
        bf16=True, report_to="none")
    trainer = RLDXTrainer(model=model, args=train_args)
    trainer.create_optimizer_and_scheduler(saved["target_updates"])
    if world > 1:
        import deepspeed
        from rldx.configs.base_config import Config
        ds = Config().get_deepspeed_config()
        ds.update(train_batch_size=saved["effective_batch"],
                  train_micro_batch_size_per_gpu=saved["effective_batch"] // (world * saved["accumulation"]),
                  gradient_accumulation_steps=saved["accumulation"], gradient_clipping=1.0)
        ds["fp16"]["enabled"], ds["bf16"]["enabled"] = False, True
        # HF owns this LambdaLR and saves scheduler.pt separately. The native
        # DeepSpeed checkpoint therefore intentionally contains lr_scheduler=None.
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=trainer.optimizer, config=ds)
        load_path, _ = engine.load_checkpoint(str(checkpoint), load_module_strict=True,
            load_optimizer_states=True, load_lr_scheduler_states=False)
        if load_path is None or engine.global_steps != state["global_step"]:
            raise RuntimeError("DeepSpeed optimizer checkpoint did not restore the expected step")
        optimizer = engine.optimizer.optimizer  # ZeRO-2's actual AdamW partition state.
    else:
        trainer.optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True))
        optimizer = trainer.optimizer
    # Metadata global_steps alone does not prove optimizer restoration: DeepSpeed
    # can fall back to model weights when its ZeRO optimizer checkpoint is absent.
    optimizer_steps, moment_numel = [], 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            moments = optimizer.state.get(parameter, {})
            step = float(moments.get("step", float("nan")))
            if not math.isfinite(step) or step != state["global_step"]:
                raise RuntimeError("AdamW moment state did not restore the expected step")
            for key in ("exp_avg", "exp_avg_sq"):
                value = moments.get(key)
                if not torch.is_tensor(value) or value.shape != parameter.shape or value.numel() == 0:
                    raise RuntimeError(f"AdamW {key} is missing or has the wrong partition shape")
                moment_numel += value.numel()
            optimizer_steps.append(int(step))
    if not optimizer_steps:
        raise RuntimeError("No restored AdamW moment states were found")
    scheduler_state = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
    trainer.lr_scheduler.load_state_dict(copy.deepcopy(scheduler_state))
    if (trainer.lr_scheduler.state_dict() != scheduler_state
            or trainer.lr_scheduler.last_epoch != state["global_step"]):
        raise RuntimeError("Saved HF scheduler fields did not restore exactly")
    group_lrs = [group["lr"] for group in optimizer.param_groups]
    if trainer.lr_scheduler.optimizer is not optimizer or group_lrs != trainer.lr_scheduler.get_last_lr():
        raise RuntimeError("Restored optimizer learning rates disagree with the HF scheduler")
    rank = int(os.environ.get("RANK", "0"))
    evidence = {"rank": rank, "checkpoint": str(checkpoint), "strict_model": True,
        "optimizer_updates": state["global_step"], "optimizer_steps": optimizer_steps,
        "optimizer_moment_states": len(optimizer_steps), "optimizer_moment_numel": moment_numel,
        "optimizer_group_lrs": group_lrs, "scheduler_source": "scheduler.pt",
        "scheduler_last_epoch": trainer.lr_scheduler.last_epoch, "scheduler_fields_equal": True,
        "additional_updates": 0, "rng_restored": False, "complete": True}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / f"reload-rank{rank}.json", evidence)
    all_ranks = [None] * world
    if world > 1:
        torch.distributed.all_gather_object(all_ranks, evidence)
    else:
        all_ranks[0] = evidence
    if rank == 0:
        write_json(args.output / "reload-evidence.json", dict(evidence, world_size=world, ranks=all_ranks))


def prepare_fresh_output(output):
    """Validate once after distributed initialization; release every rank together."""
    import torch
    import torch.distributed as dist
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not dist.is_initialized():
            # Native checkpoint export can take minutes on the shared filesystem.
            # The independently registered job still bounds total execution time.
            dist.init_process_group(backend="nccl", timeout=timedelta(seconds=600))
        if dist.get_world_size() != world:
            raise RuntimeError("Initialized process group differs from the declared training world")
    rank = dist.get_rank() if world > 1 else 0
    error = [None]
    if rank == 0:
        try:
            if output.exists() and any(output.iterdir()):
                raise FileExistsError("Fresh tune/formal output must be empty; resubmission never resumes silently")
            output.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if world > 1:
        dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise RuntimeError(error[0])
    if world > 1:
        dist.barrier()


def main():
    args = arguments()
    job_guard()
    kind = os.environ.get("MODEL_LAB_STAGE_KIND")
    limit = int(os.environ.get("MODEL_LAB_MAX_STEPS", "-1"))
    if args.command == "formal" and (kind != "formal" or limit != 500):
        raise RuntimeError("Formal acceptance requires the declared formal500 stage")
    if args.command == "tune" and (kind != "probe" or not 1 <= args.tune_steps <= min(12, limit)):
        raise RuntimeError("Tuning exceeds the declared probe optimizer-update limit")
    if args.command == "reload" and (kind != "none" or limit != 0):
        raise RuntimeError("Reload requires the declared zero-update stage")
    os.environ.setdefault("RLDX_ATTN_IMPL", "sdpa")
    if args.command == "reload":
        reload_checkpoint(args)
        return
    prepare_fresh_output(args.output)
    import torch
    from transformers import TrainerCallback
    initialization_started = time.monotonic()
    meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    baseline_mem_available_kib = int(meminfo["MemAvailable"].split()[0])
    trainer, processor, cfg, steps = build(args)
    initialization_wall_s = time.monotonic() - initialization_started
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rows = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()

    class Evidence(TrainerCallback):
        def on_log(self, arguments, state, control, logs=None, **kwargs):
            if "loss" not in (logs or {}):
                return
            if "grad_norm" not in logs or not all(math.isfinite(float(logs[key])) for key in ("loss", "grad_norm")):
                raise FloatingPointError(f"Missing/nonfinite optimizer evidence at update {state.global_step}: {logs}")
            row = dict(logs, optimizer_update=state.global_step, elapsed_s=time.monotonic() - started,
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            import psutil
            process = psutil.Process()
            row["process_and_workers_rss_bytes"] = sum(p.memory_info().rss for p in [process] + process.children(recursive=True) if p.is_running())
            rows.append(row)
            with (args.output / f"updates-rank{rank}.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")

    trainer.add_callback(Evidence())
    if rank == 0:
        write_json(args.output / "acceptance-args.json", {**{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "world_size": world, "target_updates": steps, "warmup_steps": 25 if steps == 500 else min(2, steps),
            "learning_rate": 1e-4, "attention": os.environ["RLDX_ATTN_IMPL"],
            "baseline_host_mem_available_kib": baseline_mem_available_kib,
            "model_data_initialization_wall_s": initialization_wall_s,
            "initialization_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)})
    trainer.train(resume_from_checkpoint=None)
    if trainer.state.global_step != steps or [row["optimizer_update"] for row in rows] != list(range(1, steps + 1)):
        raise RuntimeError("Requested consecutive finite optimizer updates were not proven")
    trainer.accelerator.wait_for_everyone()
    if rank == 0:
        checkpoint = args.output / f"checkpoint-{steps}"
        evidence = {"optimizer_updates": steps, "formal": args.command == "formal", "finite_loss_gradient_updates": len(rows),
                    "training_wall_s": time.monotonic() - started, "effective_batch": args.effective_batch,
                    "world_size": world, "slurm_job_id": os.environ["SLURM_JOB_ID"]}
        export_metadata(checkpoint, processor, args.backbone, args.base, evidence)
        write_json(args.output / "training-evidence.json", dict(evidence, checkpoint=str(checkpoint)))
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
