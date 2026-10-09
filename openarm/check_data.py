"""CPU check of real data through the exact native training processor/collator."""
import argparse
import json
import os
from pathlib import Path

from openarm.data import CAMERAS, GROUPS, HORIZON, PillowDataset, job_guard, write_json


def main():
    parser = argparse.ArgumentParser()
    for name in ("base", "backbone", "overlay", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    job_guard()
    if os.environ.get("MODEL_LAB_STAGE_KIND") != "none" or os.environ.get("MODEL_LAB_MAX_STEPS") != "0":
        raise RuntimeError("Processor check requires a zero-update model job")
    import numpy as np
    import torch
    from rldx.data.types import MessageType
    from openarm.train import make_processor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    train = PillowDataset(args.overlay / "train")
    processor = make_processor(args.base, args.backbone, train)
    processor.eval()
    stats = train.get_dataset_statistics()
    for modalities in stats.values():
        for groups in modalities.values():
            for values in groups.values():
                for numbers in values.values():
                    assert np.isfinite(np.asarray(numbers)).all()
    rows = []
    for split in ("train", "val"):
        dataset = PillowDataset(args.overlay / split)
        raw = dataset.raw(0)
        assert list(raw.images) == CAMERAS
        assert all(len(raw.images[key]) == 1 for key in CAMERAS)
        assert all(raw.actions[key].shape == (HORIZON, hi - lo) for key, (lo, hi) in GROUPS.items())
        sample = processor([{"type": MessageType.EPISODE_STEP.value, "content": raw}])
        batch = processor.collator([sample])["inputs"]
        assert tuple(batch["action"].shape) == (1, HORIZON, 64)
        assert tuple(batch["state"].shape) == (1, 1, 64)
        assert batch["action_mask"][..., :16].all() and not batch["action_mask"][..., 16:].any()
        assert batch["num_views"].item() == 4
        assert batch["num_frames"].item() == 1
        for value in batch.values():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all()
        sap = processor.state_action_processor
        encoded = sap.apply_action(raw.actions, "general_embodiment", state=raw.states)
        decoded = sap.unapply_action(encoded, "general_embodiment", state=raw.states)
        # Native absolute percentile clipping can intentionally change raw outliers.
        raw_error = max(float(np.max(np.abs(decoded[key] - raw.actions[key]))) for key in GROUPS)
        centers = {key: np.zeros_like(values) for key, values in encoded.items()}
        decoded_centers = sap.unapply_action(centers, "general_embodiment", state=raw.states)
        center_roundtrip = sap.apply_action(decoded_centers, "general_embodiment", state=raw.states)
        center_error = max(float(np.max(np.abs(values))) for values in center_roundtrip.values())
        assert center_error < 1e-4
        rows.append({"split": split, "episode_id": dataset.loader.episodes_metadata[0]["episode_index"],
            "raw_image_shapes": {key: list(raw.images[key][0].shape) for key in CAMERAS},
            "collated_shapes": {key: list(value.shape) for key, value in batch.items() if torch.is_tensor(value)},
            "all_finite": True, "valid_action_dimensions": 16, "history": 1,
            "raw_action_codec_max_error_with_native_clipping": raw_error,
            "interior_codec_normalized_max_error": center_error})
    write_json(args.output, {"complete": True, "checks": rows, "normalization":
        "native state/absolute grippers q01/q99; relative arms per-horizon min/max; clipping retained"})


if __name__ == "__main__":
    main()
