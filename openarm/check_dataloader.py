"""CPU acceptance of native Trainer construction and two spawned loader workers.

Two independent loaders represent the planned two GPU ranks, with one worker and
microbatch two each. This checks pickling, real decode, augmentation and collation;
it performs no model forward, backward or optimizer update.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time

from openarm.data import PillowDataset, job_guard, write_json


class WorkerAuditDataset(PillowDataset):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        if not getattr(self, "audit_written", False):
            import cv2
            import pyarrow
            import torch
            counts = {"arrow_cpu": pyarrow.cpu_count(), "arrow_io": pyarrow.io_thread_count(),
                      "opencv": cv2.getNumThreads(), "torch": torch.get_num_threads()}
            if any(count > 2 for count in counts.values()):
                raise RuntimeError(f"Spawn worker exceeded thread caps: {counts}")
            write_json(self.audit_dir / f"worker-{os.getpid()}.json", dict(pid=os.getpid(), threads=counts))
            self.audit_written = True
        return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "backbone", "overlay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    job_guard()
    if os.environ.get("MODEL_LAB_STAGE_KIND") != "none" or os.environ.get("MODEL_LAB_MAX_STEPS") != "0":
        raise RuntimeError("Loader check requires a zero-update model job")
    output_root = Path(os.environ["RLDX_OUTPUT"]).resolve()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(output_root) or args.output.exists():
        raise ValueError("Loader evidence must be fresh and private")
    args.output.mkdir(parents=True)
    import psutil
    import torch
    from transformers import TrainingArguments
    from rldx.experiment.trainer import RLDXTrainer
    from openarm.train import make_processor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    loaders, iterators, rows = [], [], []
    started = time.monotonic()
    try:
        for rank in range(2):
            dataset = WorkerAuditDataset(args.overlay / "train")
            dataset.audit_dir = args.output
            dataset.processor = make_processor(args.base, args.backbone, dataset)
            dataset.processor.train()
            trainer = RLDXTrainer(model=torch.nn.Linear(1, 1),
                args=TrainingArguments(output_dir=str(args.output / f"trainer-{rank}"), use_cpu=True,
                    per_device_train_batch_size=2, dataloader_num_workers=1, remove_unused_columns=False,
                    report_to="none", max_steps=1, seed=42 + rank, dataloader_pin_memory=False),
                train_dataset=dataset, data_collator=dataset.processor.collator,
                multiprocessing_context="spawn")
            loader = trainer.get_train_dataloader()
            if loader.num_workers != 1 or loader.batch_size != 2:
                raise AssertionError("Native trainer changed the requested worker/batch contract")
            loaders.append(loader)
            iterators.append(iter(loader))
        for index, iterator in enumerate(iterators):
            batch = next(iterator)["inputs"]
            if tuple(batch["action"].shape) != (2, 32, 64) or tuple(batch["state"].shape) != (2, 1, 64):
                raise AssertionError("Native collated state/action shapes differ from planned microbatch")
            if not batch["action_mask"][..., :16].all() or batch["action_mask"][..., 16:].any():
                raise AssertionError("Native action mask differs from sixteen real action channels")
            if not (batch["num_views"] == 4).all() or not (batch["num_frames"] == 1).all():
                raise AssertionError("Expected four current camera frames per example")
            if any(not torch.isfinite(value).all() for value in batch.values() if torch.is_tensor(value)):
                raise AssertionError("Native loader/collator produced a nonfinite tensor")
            process = psutil.Process()
            rows.append({"simulated_rank": index, "shapes": {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)},
                "finite": True, "wall_s": time.monotonic() - started,
                "process_and_workers_rss_bytes": sum(p.memory_info().rss for p in [process] + process.children(recursive=True) if p.is_running())})
    finally:
        # These iterators own these exact children; stop their bounded prefetch.
        for iterator in iterators:
            iterator._shutdown_workers()
    workers = [json.loads(path.read_text()) for path in sorted(args.output.glob("worker-*.json"))]
    if len(workers) != 2:
        raise AssertionError("Expected exactly two distinct spawned worker reports")
    write_json(args.output / "evidence.json", {"complete": True, "optimizer_updates": 0,
        "model": "tiny CPU constructor only; real checkpoint model not loaded",
        "workers_total": 2, "workers_per_rank": 1, "microbatch_per_rank": 2,
        "multiprocessing_context": "spawn", "checks": rows, "workers": workers,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "elapsed_s": time.monotonic() - started})


if __name__ == "__main__":
    main()
