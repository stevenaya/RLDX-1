"""Native loader tests with mocked models; requires the private model environment."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch


class LocalAssetsTests(unittest.TestCase):
    def test_model_local_override_and_unchanged_default(self):
        from rldx.policy import policy_loader as loader

        config = NS(model_name="RLWRLD/RLDX-1-VLM", use_memory=False)
        model = MagicMock()
        with patch.object(loader.AutoConfig, "from_pretrained", return_value=config), \
                patch.object(loader.AutoModel, "from_pretrained", return_value=model) as load:
            loader._load_model(Path("/checkpoint"), "cpu", False)
            self.assertNotIn("config", load.call_args.kwargs)
            self.assertNotIn("transformers_loading_kwargs", load.call_args.kwargs)
            self.assertNotIn("output_loading_info", load.call_args.kwargs)
            info = {key: [] for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
            load.return_value = model, info
            loader._load_model(Path("/checkpoint"), "cpu", False, Path("/checkpoint/backbone"))
            self.assertEqual(load.call_args.kwargs["config"].model_name, "/checkpoint/backbone")
            self.assertTrue(load.call_args.kwargs["local_files_only"])
            self.assertEqual(load.call_args.kwargs["transformers_loading_kwargs"],
                             {"local_files_only": True})
            self.assertIs(load.call_args.kwargs["output_loading_info"], True)
            self.assertIs(model._openarm_loading_info, info)

    def test_local_checkpoint_discrepancies_are_rejected(self):
        from rldx.policy import policy_loader as loader

        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(field=field):
                model = MagicMock()
                config = NS(model_name="RLWRLD/RLDX-1-VLM", use_memory=False)
                with patch.object(loader.AutoConfig, "from_pretrained", return_value=config), \
                        patch.object(loader.AutoModel, "from_pretrained", return_value=(model, {field: ["test discrepancy"]})):
                    with self.assertRaisesRegex(RuntimeError, field):
                        loader._load_model(Path("/checkpoint"), "cpu", False, Path("/checkpoint/backbone"))
                model.eval.assert_not_called()

    def test_processor_local_override_and_unchanged_default(self):
        from rldx.policy import policy_loader as loader

        processor = MagicMock()
        processor.physics_keys = []
        model = NS(config=NS(physics_keys=[]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "processor").mkdir()
            with patch.object(loader.AutoProcessor, "from_pretrained", return_value=processor) as load:
                loader._load_processor(root, model)
                load.assert_called_with(root / "processor")
                loader._load_processor(root, model, root / "backbone")
                load.assert_called_with(
                    root / "processor", model_name=str(root / "backbone"),
                    transformers_loading_kwargs={"local_files_only": True},
                )

    def test_processor_whitelist_keeps_checkpoint_stats(self):
        from rldx.model.core.processing_rldx import RLDXProcessor

        class RecordingProcessor(RLDXProcessor):
            def __init__(self, **kwargs):
                self.recorded = kwargs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            saved = {"processor_kwargs": {"model_name": "/old/training/cache",
                                          "modality_configs": {}, "memory_length": 1}}
            stats = {"general_embodiment": {"source": "checkpoint-specific-test"}}
            (root / "processor_config.json").write_text(json.dumps(saved))
            (root / "statistics.json").write_text(json.dumps(stats))
            (root / "embodiment_id.json").write_text('{"general_embodiment": 0}')
            loaded = RecordingProcessor.from_pretrained(
                root, model_name="/deployment/backbone",
                transformers_loading_kwargs={"local_files_only": True},
            )
            self.assertEqual(loaded.recorded["model_name"], "/deployment/backbone")
            self.assertEqual(loaded.recorded["statistics"], stats)
            self.assertEqual(loaded.recorded["embodiment_id_mapping"], {"general_embodiment": 0})
            self.assertEqual(json.loads((root / "processor_config.json").read_text()), saved)


if __name__ == "__main__":
    unittest.main()
