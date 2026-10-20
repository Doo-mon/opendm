"""CPU tests for rejecting mixed or incorrectly identified FP8 artifacts."""

import copy
import errno
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opendm.infer.prefix_mlp_fp8_contract import (
    PREFIX_MLP_FORMAT_VERSION,
    SPEC,
    bundle_lock,
    cleanup_bundle_tree,
    prefix_mlp_model_contract,
    prefix_source_model_sha256,
    publish_bundle_generation,
    read_verified_file,
    validate_build_metadata,
    validate_bundle_manifest,
    validate_calibration,
    validate_model_contract,
    write_json_atomic,
)


class PrefixMLPFP8ContractTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"hidden_size": 8, "layer_count": 2, "sha256": "a" * 64}
        self.source_sha = "b" * 64
        self.artifact = {"file": "calibration.json", "bytes": 1, "sha256": "c" * 64}
        self.metadata = {
            "tensorrt": "10.12.0",
            "cuda": "12.8",
            "gpu_name": "test GPU",
            "compute_capability": [8, 9],
            "gpu_multiprocessors": 128,
        }
        self.manifest = {
            "format_version": PREFIX_MLP_FORMAT_VERSION,
            "spec": SPEC,
            "model_contract": self.identity,
            "prefix_source_sha256": self.source_sha,
            "contract": {
                "batch_size": 1,
                "hidden_size": 8,
                "min_seq_len": 4,
                "opt_seq_len": 8,
                "max_seq_len": 16,
                "layer_indices": [0, 1],
            },
            "build_metadata": self.metadata,
            "calibration": self.artifact,
            "engines": {
                str(i): {
                    "engine": f"layer_{i:02d}.engine",
                    "bytes": 4,
                    "sha256": "d" * 64,
                    "input_range": 1.5,
                    "down_range": 2.5,
                }
                for i in range(2)
            },
        }
        self.calibration = {
            "format_version": PREFIX_MLP_FORMAT_VERSION,
            "spec": SPEC,
            "model_contract": self.identity,
            "prefix_source_sha256": self.source_sha,
            "batch_size": 1,
            "sample_count": 2,
            "distinct_sample_count": 2,
            "vision_engine": {"file": "vision.engine", "bytes": 1, "sha256": "e" * 64},
            "layers": {
                str(i): {
                    key: {"visible_rows": {"rows": 8, "elements": 64, "max_abs": 2.0}}
                    for key in ("mlp_input", "down_input")
                }
                for i in range(2)
            },
        }

    def validate_manifest(self, manifest=None, buckets=(4, 8, 16)):
        validate_bundle_manifest(
            self.manifest if manifest is None else manifest,
            self.identity,
            buckets,
            prefix_source_sha256=self.source_sha,
        )

    def validate_calibration(self, calibration=None):
        validate_calibration(
            self.calibration if calibration is None else calibration,
            self.identity,
            prefix_source_sha256=self.source_sha,
        )

    def test_complete_bundle_and_calibration(self):
        self.validate_manifest()
        self.validate_calibration()
        validate_build_metadata(self.metadata, self.metadata)

    def test_changed_loaded_weights_rejected(self):
        changed = dict(self.identity, sha256="f" * 64)
        with self.assertRaisesRegex(ValueError, "model contract mismatch"):
            validate_model_contract(self.identity, changed)

    def test_upstream_non_mlp_identity_rejected(self):
        for payload, validate in (
            (self.manifest, self.validate_manifest),
            (self.calibration, self.validate_calibration),
        ):
            changed = copy.deepcopy(payload)
            changed["prefix_source_sha256"] = "f" * 64
            with (
                self.subTest(payload="engines" in payload),
                self.assertRaisesRegex(ValueError, "upstream VLM"),
            ):
                validate(changed)

    def test_unsupported_schema_recipe_and_shapes(self):
        mutations = [
            lambda p: p.update(format_version=1),
            lambda p: p.update(spec="fp8_unknown"),
            lambda p: p["contract"].update(batch_size=2),
            lambda p: p["contract"].update(hidden_size=16),
            lambda p: p["contract"].update(min_seq_len=32),
            lambda p: p["contract"].update(max_seq_len=0),
            lambda p: p["contract"].update(opt_seq_len=8.5),
            lambda p: p["contract"].update(layer_indices=[0]),
            lambda p: p["engines"].pop("1"),
            lambda p: p["engines"]["1"].update(engine="layer_00.engine"),
            lambda p: p["engines"]["0"].update(engine="../other.engine"),
            lambda p: p["engines"]["0"].update(sha256=""),
            lambda p: p["engines"]["0"].update(bytes=0),
            lambda p: p["engines"]["0"].update(input_range=float("nan")),
            lambda p: p["calibration"].update(file="/tmp/calibration.json"),
        ]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(self.manifest)
            mutate(changed)
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.validate_manifest(changed)
        for bucket in (3, 17):
            with (
                self.subTest(bucket=bucket),
                self.assertRaisesRegex(ValueError, "outside engine profile"),
            ):
                self.validate_manifest(buckets=(bucket,))

    def test_calibration_rejects_missing_or_nonfinite_visible_statistics(self):
        mutations = [
            lambda p: p.update(sample_count=1),
            lambda p: p.update(distinct_sample_count=1),
            lambda p: p["layers"].pop("1"),
            lambda p: p["layers"]["0"]["mlp_input"]["visible_rows"].update(
                max_abs=float("inf")
            ),
            lambda p: p["layers"]["0"]["down_input"]["visible_rows"].update(max_abs=0),
            lambda p: p["layers"]["0"]["down_input"]["visible_rows"].update(rows=0),
            lambda p: p["layers"]["0"]["down_input"]["visible_rows"].pop("elements"),
        ]
        for index, mutate in enumerate(mutations):
            changed = copy.deepcopy(self.calibration)
            mutate(changed)
            with self.subTest(case=index), self.assertRaises(ValueError):
                self.validate_calibration(changed)

    def test_trt_cuda_and_gpu_mismatch_rejected(self):
        for key in self.metadata:
            changed = dict(self.metadata, **{key: "different"})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "mismatch"):
                validate_build_metadata(self.metadata, changed)

    def test_same_size_corruption_and_symlinks_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layer.engine"
            path.write_bytes(b"good")
            record = {
                "engine": path.name,
                "bytes": 4,
                "sha256": hashlib.sha256(b"good").hexdigest(),
            }
            self.assertEqual(read_verified_file(path, record), b"good")
            path.write_bytes(b"evil")
            with self.assertRaisesRegex(ValueError, "integrity mismatch"):
                read_verified_file(path, record)
            path.unlink()
            target = Path(directory) / "other.engine"
            target.write_bytes(b"good")
            path.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symlink"):
                read_verified_file(path, record)

    def test_json_publication_preserves_previous_file_on_serialization_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            write_json_atomic(path, {"generation": 1})
            with self.assertRaises(ValueError):
                write_json_atomic(path, {"generation": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"generation": 1})
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_reader_lock_blocks_incomplete_generation(self):
        # Nonblocking flock in a second descriptor checks the actual kernel
        # lock, without a sleep-based race between threads/processes.
        import fcntl

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            lock_path = Path(directory) / ".bundle.lock"
            with bundle_lock(bundle, exclusive=True):
                with lock_path.open("rb") as reader:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(reader, fcntl.LOCK_SH | fcntl.LOCK_NB)
            with bundle_lock(bundle):
                with lock_path.open("rb") as other_reader:
                    fcntl.flock(other_reader, fcntl.LOCK_SH | fcntl.LOCK_NB)
                    fcntl.flock(other_reader, fcntl.LOCK_UN)

    def test_reader_lock_works_without_write_access(self):
        import fcntl

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            lock_path = Path(directory) / ".bundle.lock"
            lock_path.touch()
            original_open = Path.open

            def readonly_open(path, mode="r", *args, **kwargs):
                if path == lock_path and any(flag in mode for flag in "wax+"):
                    raise OSError(errno.EROFS, "Read-only file system", str(path))
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", readonly_open):
                with bundle_lock(bundle):
                    with lock_path.open("rb") as writer:
                        with self.assertRaises(BlockingIOError):
                            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(OSError):
                    with bundle_lock(bundle, exclusive=True):
                        self.fail("A publisher must require write access")

    def test_missing_readonly_lock_has_deployment_error(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "bundle"
            lock_path = Path(directory) / ".bundle.lock"
            original_open = Path.open

            def readonly_open(path, mode="r", *args, **kwargs):
                if path == lock_path and mode == "a+b":
                    raise OSError(errno.EROFS, "Read-only file system", str(path))
                return original_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", readonly_open):
                with self.assertRaisesRegex(
                    OSError, "keep the bundle and its sibling lock"
                ):
                    with bundle_lock(bundle):
                        self.fail("Missing lock must not be silently bypassed")


class BundlePublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.engine = self.root / "bundle"
        self.staging = self.root / ".bundle.building"
        self.backup = self.root / ".bundle.previous"
        self.engine.mkdir()
        (self.engine / "generation").write_text("old")
        self.staging.mkdir()
        (self.staging / "generation").write_text("new")
        self.original_rename = Path.rename

    def tearDown(self):
        self.temporary.cleanup()

    def assert_generation(self, path, expected):
        self.assertEqual((path / "generation").read_text(), expected)

    def test_replace_publishes_new_and_retires_old_generation(self):
        publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assert_generation(self.engine, "new")
        self.assertFalse(self.staging.exists())
        self.assertFalse(self.backup.exists())

    def test_failure_before_old_rename_preserves_both_inputs(self):
        primary = OSError("cannot create backup")

        def fail_old_rename(path, target):
            if path == self.engine:
                raise primary
            return self.original_rename(path, target)

        with patch.object(Path, "rename", new=fail_old_rename):
            with self.assertRaises(OSError) as caught:
                publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assertIs(caught.exception, primary)
        self.assert_generation(self.engine, "old")
        self.assert_generation(self.staging, "new")
        self.assertFalse(self.backup.exists())

    def test_interrupt_after_old_rename_rolls_back_and_propagates(self):
        primary = KeyboardInterrupt("after backup rename")

        def interrupt_after_old_rename(path, target):
            result = self.original_rename(path, target)
            if path == self.engine:
                raise primary
            return result

        with patch.object(Path, "rename", new=interrupt_after_old_rename):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assertIs(caught.exception, primary)
        self.assert_generation(self.engine, "old")
        self.assert_generation(self.staging, "new")
        self.assertFalse(self.backup.exists())

    def test_new_publish_failure_restores_old_and_preserves_primary(self):
        primary = OSError("new rename failed")

        def fail_new_rename(path, target):
            if path == self.staging:
                raise primary
            return self.original_rename(path, target)

        with patch.object(Path, "rename", new=fail_new_rename):
            with self.assertRaises(OSError) as caught:
                publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assertIs(caught.exception, primary)
        self.assert_generation(self.engine, "old")
        self.assert_generation(self.staging, "new")
        self.assertFalse(self.backup.exists())

    def test_interrupt_after_commit_keeps_new_and_backup_and_propagates(self):
        primary = KeyboardInterrupt("after commit")

        def interrupt_after_new_rename(path, target):
            result = self.original_rename(path, target)
            if path == self.staging:
                raise primary
            return result

        with patch.object(Path, "rename", new=interrupt_after_new_rename):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assertIs(caught.exception, primary)
        self.assert_generation(self.engine, "new")
        self.assertFalse(self.staging.exists())
        self.assert_generation(self.backup, "old")

    def test_rollback_failure_keeps_backup_and_original_exception(self):
        primary = OSError("publish failed")
        rollback = OSError("rollback failed")

        def fail_publish_and_rollback(path, target):
            if path == self.staging:
                raise primary
            if path == self.backup:
                raise rollback
            return self.original_rename(path, target)

        with patch.object(Path, "rename", new=fail_publish_and_rollback):
            with self.assertLogs("opendm.infer.prefix_mlp_fp8_contract", level="ERROR"):
                with self.assertRaises(OSError) as caught:
                    publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assertIs(caught.exception, primary)
        self.assertFalse(self.engine.exists())
        self.assert_generation(self.backup, "old")
        self.assert_generation(self.staging, "new")
        if hasattr(primary, "__notes__"):
            self.assertIn("rollback reported an error", " ".join(primary.__notes__))

    def test_interrupt_after_rollback_effect_keeps_primary_and_restored_bundle(self):
        primary = OSError("publish failed")
        rollback = KeyboardInterrupt("after rollback")

        def fail_publish_then_interrupt_after_rollback(path, target):
            if path == self.staging:
                raise primary
            result = self.original_rename(path, target)
            if path == self.backup:
                raise rollback
            return result

        with patch.object(
            Path, "rename", new=fail_publish_then_interrupt_after_rollback
        ):
            with self.assertLogs("opendm.infer.prefix_mlp_fp8_contract", level="ERROR"):
                with self.assertRaises(OSError) as caught:
                    publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assertIs(caught.exception, primary)
        self.assert_generation(self.engine, "old")
        self.assertFalse(self.backup.exists())
        self.assert_generation(self.staging, "new")

    def test_backup_cleanup_failure_does_not_fail_committed_publish(self):
        cleanup_error = OSError("backup is busy")

        def fail_backup_cleanup(path):
            if Path(path) == self.backup:
                raise cleanup_error
            raise AssertionError(f"Unexpected cleanup target: {path}")

        with patch(
            "opendm.infer.prefix_mlp_fp8_contract.shutil.rmtree",
            side_effect=fail_backup_cleanup,
        ):
            with self.assertLogs(
                "opendm.infer.prefix_mlp_fp8_contract", level="WARNING"
            ):
                publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assert_generation(self.engine, "new")
        self.assert_generation(self.backup, "old")

    def test_backup_cleanup_interrupt_propagates_after_commit(self):
        interrupt = KeyboardInterrupt("cleanup cancelled")

        with patch(
            "opendm.infer.prefix_mlp_fp8_contract.shutil.rmtree",
            side_effect=interrupt,
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assertIs(caught.exception, interrupt)
        self.assert_generation(self.engine, "new")
        self.assertFalse(self.staging.exists())
        self.assert_generation(self.backup, "old")

    def test_staging_cleanup_failure_never_masks_primary_failure(self):
        primary = RuntimeError("build failed")

        with patch(
            "opendm.infer.prefix_mlp_fp8_contract.shutil.rmtree",
            side_effect=OSError("staging is busy"),
        ):
            with self.assertLogs(
                "opendm.infer.prefix_mlp_fp8_contract", level="WARNING"
            ):
                with self.assertRaises(RuntimeError) as caught:
                    try:
                        raise primary
                    finally:
                        cleanup_bundle_tree(
                            self.staging,
                            purpose="incomplete bundle staging directory",
                        )
        self.assertIs(caught.exception, primary)
        self.assertTrue(self.staging.exists())

    def test_recovery_probe_failure_does_not_mask_publish_error(self):
        primary = OSError("publish failed")
        original_exists = Path.exists
        backup_probe_count = 0

        def fail_publish(path, target):
            if path == self.staging:
                raise primary
            return self.original_rename(path, target)

        def fail_recovery_probe(path):
            nonlocal backup_probe_count
            if path == self.backup:
                backup_probe_count += 1
                if backup_probe_count > 1:
                    raise PermissionError("cannot inspect backup")
            return original_exists(path)

        with (
            patch.object(Path, "rename", new=fail_publish),
            patch.object(Path, "exists", new=fail_recovery_probe),
            self.assertLogs("opendm.infer.prefix_mlp_fp8_contract", level="ERROR"),
            self.assertRaises(OSError) as caught,
        ):
            publish_bundle_generation(self.staging, self.engine, self.backup)

        self.assertIs(caught.exception, primary)
        self.assertFalse(self.engine.exists())
        self.assert_generation(self.backup, "old")
        self.assert_generation(self.staging, "new")

    def test_staging_exists_probe_failure_does_not_mask_primary(self):
        primary = RuntimeError("build failed")

        def fail_exists(_path):
            raise PermissionError("cannot inspect staging")

        with patch.object(Path, "exists", new=fail_exists):
            with self.assertLogs(
                "opendm.infer.prefix_mlp_fp8_contract", level="WARNING"
            ):
                with self.assertRaises(RuntimeError) as caught:
                    try:
                        raise primary
                    finally:
                        cleanup_bundle_tree(
                            self.staging,
                            purpose="incomplete bundle staging directory",
                        )
        self.assertIs(caught.exception, primary)


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "requires the runtime PyTorch/Transformers environment (CUDA is not required)",
)
class PrefixMLPModelIdentityTests(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
        from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel

        self.torch = torch
        config = Gemma3TextConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=64,
            sliding_window=32,
        )
        self.model = Gemma3TextModel(config).to(dtype=torch.bfloat16).eval()

    def test_real_gemma_mlp_contract_and_weight_changes(self):
        before = prefix_mlp_model_contract(self.model)
        self.assertEqual(before["layer_count"], 2)
        self.assertEqual(before, prefix_mlp_model_contract(self.model))
        with self.torch.no_grad():
            self.model.layers[0].mlp.gate_proj.weight[0, 0].add_(1)
        after = prefix_mlp_model_contract(self.model)
        with self.assertRaisesRegex(ValueError, "model contract mismatch"):
            validate_model_contract(before, after)

    def test_upstream_embedding_changes_source_identity(self):
        mlp_before = prefix_mlp_model_contract(self.model)
        source_before = prefix_source_model_sha256(self.model)
        with self.torch.no_grad():
            self.model.embed_tokens.weight[0, 0].add_(1)
        self.assertEqual(mlp_before, prefix_mlp_model_contract(self.model))
        self.assertNotEqual(source_before, prefix_source_model_sha256(self.model))

    def test_wrong_activation_and_bias_are_rejected(self):
        mlp = self.model.layers[0].mlp
        old_activation = mlp.act_fn
        mlp.act_fn = self.torch.nn.ReLU()
        with self.assertRaisesRegex(ValueError, "tanh-approximated GELU"):
            prefix_mlp_model_contract(self.model)
        mlp.act_fn = old_activation
        mlp.gate_proj = self.torch.nn.Linear(
            8, 16, bias=True, dtype=self.torch.bfloat16
        )
        with self.assertRaisesRegex(ValueError, "bias-free Linear"):
            prefix_mlp_model_contract(self.model)

    def test_source_identity_ignores_execution_backend_only(self):
        before = prefix_source_model_sha256(self.model)
        self.model.config._attn_implementation = "eager"
        self.assertEqual(before, prefix_source_model_sha256(self.model))
        self.model.config.rms_norm_eps *= 2
        self.assertNotEqual(before, prefix_source_model_sha256(self.model))


if __name__ == "__main__":
    unittest.main()
