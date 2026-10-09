"""Synthetic split/normalization tests; run inside the private model job."""
import json
from pathlib import Path
import tempfile
import unittest

from openarm.data import CAMERAS, GROUPS, HORIZON, job_guard, modality_config, prepare_overlay, prepare_stats


class DataContractTest(unittest.TestCase):
    def test_disjoint_split_statistics_and_native_codec(self):
        job_guard()
        import numpy as np
        import pandas as pd
        from rldx.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
        from rldx.data.state_action.state_action_processor import StateActionProcessor

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "meta").mkdir(parents=True)
            (source / "data/chunk-000").mkdir(parents=True)
            (source / "videos").mkdir()
            names = [f"{side}_joint{i}.pos" for side in ("right", "left") for i in range(1, 8)]
            names = names[:7] + ["right_gripper.pos"] + names[7:] + ["left_gripper.pos"]
            info = {"codebase_version": "v2.1", "fps": 30, "chunks_size": 1000,
                    "total_episodes": 4, "total_frames": 144, "splits": {"train": "0:3", "val": "3:4"},
                    "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                    "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                    "features": {key: {"shape": [16], "names": names, "dtype": "float32"} for key in ("action", "observation.state")}}
            (source / "meta/info.json").write_text(json.dumps(info))
            (source / "meta/episodes.jsonl").write_text("".join(json.dumps({"episode_index": i, "length": 36}) + "\n" for i in range(4)))
            (source / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"synthetic"}\n')
            modality = {mod: {key: {"start": lo, "end": hi} for key, (lo, hi) in GROUPS.items()} for mod in ("state", "action")}
            modality["video"] = {key: {"original_key": "observation.images." + key} for key in CAMERAS}
            modality["annotation"] = {"human.action.task_description": {"original_key": "task_index"}}
            (root / "modality.json").write_text(json.dumps(modality))
            expected_chunks = []
            for episode in range(4):
                state = np.arange(36 * 16, dtype=np.float32).reshape(36, 16) / 100 + episode
                action = state + np.linspace(0, .1, 16, dtype=np.float32)
                if episode == 3:
                    state[:] = action[:] = 1e6  # Validation contamination is unmistakable.
                else:
                    expected_chunks.extend(action[start:start + HORIZON] - state[start] for start in range(5))
                pd.DataFrame({"observation.state": list(state), "action": list(action), "task_index": 0}).to_parquet(
                    source / f"data/chunk-000/episode_{episode:06d}.parquet")
            source_hashes = {path: path.read_bytes() for path in (source / "meta").iterdir()}
            overlay = root / "overlay"
            prepare_overlay(source, overlay, root / "modality.json")
            prepare_stats(overlay, root / "scratch")
            self.assertFalse((overlay / "train/data/chunk-000/episode_000003.parquet").exists())
            stats = json.loads((overlay / "train/meta/stats.json").read_text())
            self.assertLess(max(stats["action"]["max"]), 20)
            self.assertEqual((overlay / "train/meta/stats.json").read_bytes(), (overlay / "val/meta/stats.json").read_bytes())
            for path, before in source_hashes.items():
                self.assertEqual(path.read_bytes(), before)
            relative = json.loads((overlay / "train/meta/relative_stats.json").read_text())
            expected = np.stack(expected_chunks)
            for key in ("right_arm", "left_arm"):
                lo, hi = GROUPS[key]
                np.testing.assert_allclose(relative[key]["mean"], expected[:, :, lo:hi].mean(axis=0), atol=2e-6)
                np.testing.assert_allclose(relative[key]["q99"], np.quantile(expected[:, :, lo:hi], .99, axis=0), atol=1e-7)
            loader = LeRobotEpisodeLoader(overlay / "train", modality_config())
            sap = StateActionProcessor({"general_embodiment": modality_config()}, statistics={"general_embodiment": loader.get_dataset_statistics()},
                                       use_relative_action=True, clip_outliers=False)
            state = {key: np.ones((1, hi - lo), dtype=np.float32) for key, (lo, hi) in GROUPS.items()}
            actions = {key: np.full((HORIZON, hi - lo), 1.03, dtype=np.float32) for key, (lo, hi) in GROUPS.items()}
            normalized = sap.apply_action(actions, "general_embodiment", state=state)
            decoded = sap.unapply_action(normalized, "general_embodiment", state=state)
            # Constant synthetic relative ranges are degenerate; grippers still test native absolute ordering.
            for key in ("right_gripper", "left_gripper"):
                np.testing.assert_allclose(decoded[key], actions[key], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
