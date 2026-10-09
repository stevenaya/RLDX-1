"""Read-only Pillow v2.1 data adapter; execute only in an isolated model job."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

CAMERAS = ["wrist_right", "wrist_left", "ceiling", "head_left"]
GROUPS = {"right_arm": (0, 7), "right_gripper": (7, 8), "left_arm": (8, 15), "left_gripper": (15, 16)}
LANGUAGE = "annotation.human.action.task_description"
HORIZON = 32
BASE_REVISION = "d67fc642a7e768049ba41ff8cf0dfe0ff17f556b"


def job_guard():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("OPENARM_MODEL_JOB") != "1":
        raise RuntimeError("Run in a registered isolated Slurm model job with OPENARM_MODEL_JOB=1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MAX_JOBS", "TORCHINDUCTOR_COMPILE_THREADS"):
        os.environ[name] = "2"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import pyarrow
    pyarrow.set_cpu_count(2)
    pyarrow.set_io_thread_count(2)
    import cv2
    cv2.setNumThreads(2)


def modality_config():
    from rldx.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=CAMERAS),
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(GROUPS)),
        "action": ModalityConfig(delta_indices=list(range(HORIZON)), modality_keys=list(GROUPS), action_configs=[
            ActionConfig(rep=ActionRepresentation.RELATIVE if "arm" in key else ActionRepresentation.ABSOLUTE,
                         type=ActionType.NON_EEF, format=ActionFormat.DEFAULT, state_key=key)
            for key in GROUPS]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[LANGUAGE]),
    }


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def prepare_overlay(source: Path, output: Path, modality: Path):
    """Create metadata and links; source contents are never modified."""
    job_guard()
    if output.exists():
        raise FileExistsError(f"Refuse to overwrite existing overlay: {output}")
    info = json.loads((source / "meta/info.json").read_text())
    external = json.loads(modality.read_text())
    if info["codebase_version"] != "v2.1" or info["fps"] != 30:
        raise ValueError("This adaptation requires verified Pillow v2.1 at 30 Hz")
    names = [f"{side}_joint{i}.pos" for side in ("right", "left") for i in range(1, 8)]
    expected = names[:7] + ["right_gripper.pos"] + names[7:] + ["left_gripper.pos"]
    for key in ("action", "observation.state"):
        if info["features"][key]["shape"] != [16] or info["features"][key]["names"] != expected:
            raise ValueError(f"Unverified order/shape in {key}")
    for mod in ("state", "action"):
        for key, (start, end) in GROUPS.items():
            if [external[mod][key]["start"], external[mod][key]["end"]] != [start, end]:
                raise ValueError(f"Unexpected {mod}/{key} layout")
    for key in CAMERAS:
        if external["video"][key]["original_key"] != f"observation.images.{key}":
            raise ValueError(f"Unexpected camera alias {key}")
    records = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines()]
    if len(records) != info["total_episodes"]:
        raise ValueError("Episode count differs from info.json")
    split_ids = {}
    for split in ("train", "val"):
        lo, hi = map(int, info["splits"][split].split(":"))
        split_ids[split] = set(range(lo, hi))
    if split_ids["train"] & split_ids["val"]:
        raise ValueError("Overlapping source splits")
    for split, ids in split_ids.items():
        root = output / split
        (root / "meta").mkdir(parents=True)
        selected = [row for row in records if row["episode_index"] in ids]
        if {row["episode_index"] for row in selected} != ids:
            raise ValueError(f"Missing episode ids in {split}")
        split_info = dict(info, total_episodes=len(selected), total_frames=sum(row["length"] for row in selected))
        split_info["splits"] = {split: info["splits"][split]}
        write_json(root / "meta/info.json", split_info)
        write_json(root / "meta/modality.json", external)
        (root / "meta/episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
        shutil.copyfile(source / "meta/tasks.jsonl", root / "meta/tasks.jsonl")
        (root / "videos").symlink_to(source / "videos", target_is_directory=True)
        for row in selected:
            eid = row["episode_index"]
            relative = Path(info["data_path"].format(episode_chunk=eid // info["chunks_size"], episode_index=eid))
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source / relative)
    write_json(output / "overlay.json", {
        "source": str(source), "source_info_sha256": hashlib.sha256((source / "meta/info.json").read_bytes()).hexdigest(),
        "splits": {name: sorted(ids) for name, ids in split_ids.items()},
        "camera_order": CAMERAS, "history": [0], "horizon": HORIZON,
        "source_action_semantics": "absolute targets; confirm sample units separately",
        "training_action_semantics": "current-state-relative arms, absolute grippers",
    })


def summarize(array):
    import numpy as np
    result = {"min": np.min(array, axis=0), "max": np.max(array, axis=0),
              "mean": np.mean(array, axis=0, dtype=np.float64), "std": np.std(array, axis=0, dtype=np.float64)}
    quantiles = np.quantile(array, [0.01, 0.99], axis=0, overwrite_input=True)
    result.update(q01=quantiles[0], q99=quantiles[1])
    return {key: value.tolist() for key, value in result.items()}


def prepare_stats(overlay: Path, scratch: Path):
    """Exact train-only statistics, including a separate statistic for each action offset."""
    job_guard()
    import numpy as np
    import pandas as pd
    root = overlay / "train"
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    count = sum(row["length"] for row in episodes)
    valid = sum(max(0, row["length"] - HORIZON + 1) for row in episodes)
    scratch.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stats_frames": count, "valid_starts": valid,
                      "temporary_disk_bytes": count * 32 * 4 + valid * HORIZON * 4}), flush=True)
    with tempfile.TemporaryDirectory(prefix="rldx-stats-", dir=scratch) as temporary:
        raw = np.memmap(Path(temporary) / "raw.bin", mode="w+", dtype="float32", shape=(count, 32))
        boundaries = []
        cursor = 0
        for index, row in enumerate(episodes):
            eid, length = row["episode_index"], row["length"]
            filename = info["data_path"].format(episode_chunk=eid // info["chunks_size"], episode_index=eid)
            frame = pd.read_parquet(root / filename, columns=["observation.state", "action"])
            values = np.concatenate([np.stack(frame[key].values) for key in ("observation.state", "action")], axis=1)
            if values.shape != (length, 32) or not np.isfinite(values).all():
                raise ValueError(f"Invalid numerical data in episode {eid}")
            raw[cursor:cursor + length] = values
            boundaries.append((cursor, length))
            cursor += length
            if index % 100 == 0:
                print(f"stats parquet {index}/{len(episodes)}", flush=True)
        absolute = {}
        for key, offset in (("observation.state", 0), ("action", 16)):
            # summarize sorts its input for exact quantiles; preserve raw source values.
            absolute[key] = summarize(np.array(raw[:, offset:offset + 16]))
        write_json(root / "meta/stats.json", absolute)
        relative = {}
        column = np.memmap(Path(temporary) / "relative.bin", mode="w+", dtype="float32", shape=(valid, HORIZON))
        for group in ("right_arm", "left_arm"):
            joint_stats = []
            for dim in range(*GROUPS[group]):
                cursor = 0
                for start, length in boundaries:
                    n = max(0, length - HORIZON + 1)
                    for h in range(HORIZON):
                        column[cursor:cursor + n, h] = raw[start + h:start + h + n, 16 + dim] - raw[start:start + n, dim]
                    cursor += n
                joint_stats.append(summarize(column))
                print(f"stats relative {group}/{dim} complete", flush=True)
            relative[group] = {key: np.stack([entry[key] for entry in joint_stats], axis=1).tolist() for key in joint_stats[0]}
        write_json(root / "meta/relative_stats.json", relative)
    # Validation uses training normalization; never calculate validation statistics.
    for filename in ("stats.json", "relative_stats.json"):
        shutil.copyfile(root / "meta" / filename, overlay / "val/meta" / filename)
    write_json(overlay / "statistics-evidence.json", {"source_split": "train", "frames": count,
        "valid_chunk_starts": valid, "horizon": HORIZON, "method": "exact per-offset relative min/max/mean/std/q01/q99",
        "validation_stats": "copied training statistics", "complete": True})


class PillowDataset:
    """Native low-dimensional loader and processor, bounded single-frame video decode."""
    def __init__(self, root, processor=None, decoder="pyav"):
        import numpy as np
        from rldx.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        self.root, self.processor, self.decoder = Path(root), processor, decoder
        self.mods = modality_config()
        self.loader = LeRobotEpisodeLoader(self.root, self.mods)
        self.lengths = [max(0, item["length"] - HORIZON + 1) for item in self.loader.episodes_metadata]
        self.cumulative = np.cumsum(self.lengths).tolist()
        self.seed = 42
        self.cached_id, self.cached_data = None, None
        self.runtime_pid = None

    def __len__(self):
        return self.cumulative[-1]

    def reset_seed(self, seed):
        self.seed = seed

    def get_dataset_statistics(self):
        return {"general_embodiment": self.loader.get_dataset_statistics()}

    def frame(self, episode_id, camera, index):
        import numpy as np
        path = self.root / self.loader.video_path_pattern.format(episode_chunk=episode_id // self.loader.chunk_size,
            video_key=f"observation.images.{camera}", episode_index=episode_id)
        if self.decoder == "torchcodec":
            from torchcodec.decoders import VideoDecoder
            decoder = VideoDecoder(str(path), device="cpu", dimension_order="NHWC", num_ffmpeg_threads=2)
            return decoder.get_frame_at(index).data.numpy()
        import av
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 2
            stream.codec_context.thread_type = "SLICE"
            target = index / self.loader.fps
            container.seek(int(target / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                position = round(float(frame.pts * stream.time_base) * self.loader.fps)
                if position == index:
                    return np.asarray(frame.to_ndarray(format="rgb24"))
                if position > index:
                    break
        raise ValueError(f"Unable to decode exact frame {index} in {path}")

    def raw(self, index):
        if self.runtime_pid != os.getpid():
            # Spawned workers do not inherit process-local Arrow/OpenCV limits.
            job_guard()
            self.runtime_pid = os.getpid()
        from rldx.data.dataset.sharded_single_step_dataset import extract_step_data
        from rldx.data.embodiment_tags import EmbodimentTag
        episode = bisect.bisect_right(self.cumulative, index)
        step = index - (self.cumulative[episode - 1] if episode else 0)
        eid = self.loader.episodes_metadata[episode]["episode_index"]
        if self.cached_id != eid:
            self.cached_data = self.loader._load_parquet_data(eid)
            self.cached_id = eid
        data = extract_step_data(self.cached_data, step, self.mods, EmbodimentTag.GENERAL_EMBODIMENT)
        data.images = {key: [self.frame(eid, key, step)] for key in CAMERAS}
        return data

    def __getitem__(self, index):
        from rldx.data.types import MessageType
        return self.processor([{"type": MessageType.EPISODE_STEP.value, "content": self.raw(index)}])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "stats", "warmup"])
    parser.add_argument("--source", type=Path)
    parser.add_argument("--modality", type=Path)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    job_guard()
    if os.environ.get("MODEL_LAB_STAGE_KIND") != "none" or os.environ.get("MODEL_LAB_MAX_STEPS") != "0":
        raise RuntimeError("Data preparation requires a declared zero-update stage")
    if args.command == "prepare":
        prepare_overlay(args.source.resolve(), args.overlay, args.modality)
    elif args.command == "stats":
        prepare_stats(args.overlay, args.scratch)
    else:
        import numpy as np
        data = PillowDataset(args.overlay / "val").raw(0)
        np.savez(args.output, qpos=np.concatenate([data.states[key][0] for key in GROUPS]), prompt=np.asarray(data.text),
                 **{f"camera_{key}": data.images[key][0] for key in CAMERAS})


if __name__ == "__main__":
    main()
