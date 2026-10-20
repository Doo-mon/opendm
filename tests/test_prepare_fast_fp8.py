"""CPU tests for offline FP8 preparation and prepared artifact discovery."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from opendm.infer import prepare_fast as prepare
from opendm.infer import prepare_fast_fp8 as legacy_prepare
from opendm.infer.fast_fp8_prepared import (
    PREPARED_FORMAT_VERSION,
    PreparedArtifactError,
    artifact_record,
    load_prepared_fast_artifacts,
    write_json_atomic,
)


class SamplingTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_covers_tasks_then_episodes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for task, counts in (("task-a", (1, 30)), ("task-b", (1, 30))):
                for episode, count in enumerate(counts):
                    path = root / f"{task}-{episode}.jsonl"
                    path.write_text(
                        "".join(
                            json.dumps(
                                {
                                    "prompt": task,
                                    "episode_id": episode,
                                    "state": [index],
                                }
                            )
                            + "\n"
                            for index in range(count)
                        )
                    )
                    paths.append(path)
            first = prepare._sample_jsonl(paths, 8, 7)
            second = prepare._sample_jsonl(paths, 8, 7)

            def identity(values):
                return [
                    (item["prompt"], item["path"].name, item["line"]) for item in values
                ]

            self.assertEqual(identity(first), identity(second))
            self.assertEqual({item["prompt"] for item in first}, {"task-a", "task-b"})
            for task in ("task-a", "task-b"):
                task_paths = {
                    item["path"].name for item in first if item["prompt"] == task
                }
                self.assertEqual(len(task_paths), 2)

    def test_single_request_stream_honors_requested_sample_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"text": "one task", "states": [index]}) + "\n"
                    for index in range(300)
                )
            )
            chosen = prepare._sample_jsonl([path], 128, 11)
            self.assertEqual(len(chosen), 128)
            self.assertEqual(len({item["line"] for item in chosen}), 128)

    def test_buckets_respect_real_image_token_minimum(self):
        buckets, minimum = prepare._prepared_buckets(
            None,
            min_seq_len=576,
            prefix_capacity=1024,
            num_images=3,
            mm_tokens_per_image=256,
        )
        self.assertEqual(minimum, 771)
        self.assertEqual(buckets, [896, 1024])
        with self.assertRaisesRegex(ValueError, "image-token minimum"):
            prepare._prepared_buckets(
                [768, 896],
                min_seq_len=576,
                prefix_capacity=1024,
                num_images=3,
                mm_tokens_per_image=256,
            )

    def test_image_and_video_decode_are_rgb(self):
        import av

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "frame.png"
            Image.new("L", (7, 5), 123).save(image_path)
            jsonl = root / "episode.jsonl"
            jsonl.touch()
            args = SimpleNamespace(media_root=[], dataset_root=root)
            decoded = prepare._decode_media(
                {"type": "image", "url": "./frame.png"}, jsonl, args
            )
            self.assertEqual((decoded.mode, decoded.size), ("RGB", (7, 5)))

            video_path = root / "clip.mp4"
            with av.open(str(video_path), "w") as container:
                stream = container.add_stream("mpeg4", rate=1)
                stream.width = 16
                stream.height = 16
                stream.pix_fmt = "yuv420p"
                for value in (10, 200):
                    frame = av.VideoFrame.from_image(
                        Image.new("RGB", (16, 16), (value, value, value))
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            decoded = prepare._decode_media(
                {"type": "video", "url": "clip.mp4", "frame_idx": 1},
                jsonl,
                args,
            )
            self.assertEqual(decoded.getpixel((0, 0)), (200, 200, 200))
            for invalid in (True, 1.5, -1, "1"):
                with (
                    self.subTest(frame_idx=invalid),
                    self.assertRaisesRegex(ValueError, "nonnegative integer"),
                ):
                    prepare._decode_media(
                        {"type": "video", "url": "clip.mp4", "frame_idx": invalid},
                        jsonl,
                        args,
                    )

    def test_custom_python_module_supports_dataclass_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            module_path = Path(directory) / "custom.py"
            module_path.write_text(
                "from __future__ import annotations\n"
                "from dataclasses import dataclass\n"
                "@dataclass\n"
                "class Config:\n"
                "    value: str = 'ok'\n"
            )
            module = prepare._load_module(str(module_path))
            self.assertEqual(module.Config().value, "ok")

    def test_prompt_truncation_default_and_calibration_rejection(self):
        import pytest

        torch = pytest.importorskip("torch")
        pytest.importorskip("numpy")
        pytest.importorskip("megfile")
        pytest.importorskip("orjson")
        from opendm.data.transforms import ChatTokenization

        class Tokenizer:
            def convert_tokens_to_ids(self, _token):
                return 7

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                return list(range(len(text.split())))

            def decode(self, values, skip_special_tokens=False):
                del skip_special_tokens
                return " ".join(f"word{value}" for value in values)

        class Processor:
            tokenizer = Tokenizer()

            def __init__(self):
                self.calls = 0

            def apply_chat_template(self, messages, **_kwargs):
                self.calls += 1
                text = " ".join(item.get("text", "") for item in messages[0]["content"])
                length = len(text.split())
                return {
                    "input_ids": torch.arange(length).unsqueeze(0),
                    "attention_mask": torch.ones((1, length), dtype=torch.long),
                    "token_type_ids": torch.zeros((1, length), dtype=torch.long),
                    "pixel_values": torch.empty((0, 3, 1, 1)),
                }

        data = {
            "prompt": " ".join(f"task{i}" for i in range(30)),
            "images": [],
            "state": [],
            "meta_data": {"speed": "0.5"},
        }
        processor = Processor()
        normal = ChatTokenization(
            processor=processor,
            max_length=20,
            image_prompts=[],
            add_state=False,
        )
        self.assertLessEqual(normal(dict(data))["input_ids"].shape[1], 20)
        self.assertEqual(processor.calls, 2)

        processor = Processor()
        calibration = ChatTokenization(
            processor=processor,
            max_length=20,
            image_prompts=[],
            add_state=False,
            reject_prompt_truncation=True,
        )
        with self.assertRaisesRegex(ValueError, "shorten this prompt"):
            calibration(dict(data))
        self.assertEqual(processor.calls, 1)


class PreparedContractTests(unittest.TestCase):
    def _generation(self, root: Path):
        generation = root / "generations" / "one"
        bundle = generation / "prefix_mlp_fp8"
        bundle.mkdir(parents=True)
        (generation / "vision.engine").write_bytes(b"vision")
        calibration = b"calibration"
        engine = b"layer-engine"
        (bundle / "calibration.json").write_bytes(calibration)
        (bundle / "layer_00.engine").write_bytes(engine)
        manifest = {
            "calibration": {
                "file": "calibration.json",
                "bytes": len(calibration),
                "sha256": hashlib.sha256(calibration).hexdigest(),
            },
            "engines": {
                "0": {
                    "engine": "layer_00.engine",
                    "bytes": len(engine),
                    "sha256": hashlib.sha256(engine).hexdigest(),
                }
            },
        }
        (bundle / "prefix_mlp_quant_manifest.json").write_text(json.dumps(manifest))
        preprocessing = {"identity": "preprocessing"}
        deployment = {
            "num_images": 2,
            "chunk_size": 10,
            "diffusion_steps": 10,
            "prefix_capacity": 1024,
            "prefix_seq_len_buckets": [576, 704, 768, 896, 1024],
            "history": False,
        }
        ready = {
            "format_version": PREPARED_FORMAT_VERSION,
            "deployment": deployment,
            "preprocessing": preprocessing,
            "artifacts": {
                "vision_engine": artifact_record(root, generation / "vision.engine"),
                "prefix_manifest": artifact_record(
                    root, bundle / "prefix_mlp_quant_manifest.json"
                ),
            },
        }
        write_json_atomic(root / "prepared.json", ready)
        return deployment, preprocessing

    def test_valid_generation_and_relocated_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment, preprocessing = self._generation(root)
            prepared = load_prepared_fast_artifacts(
                checkpoint=Path(directory) / "relocated-model",
                prepared_dir=root,
                preprocessing=preprocessing,
                prefix_seq_len_buckets=None,
                is_history=False,
                **{
                    key: deployment[key]
                    for key in (
                        "num_images",
                        "chunk_size",
                        "diffusion_steps",
                        "prefix_capacity",
                    )
                },
            )
            self.assertEqual(prepared.vision_engine.read_bytes(), b"vision")

    def test_path_config_and_integrity_mismatches_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment, preprocessing = self._generation(root)
            kwargs = {
                "checkpoint": root / "model",
                "prepared_dir": root,
                "preprocessing": preprocessing,
                "prefix_seq_len_buckets": None,
                "is_history": False,
                **{
                    key: deployment[key]
                    for key in (
                        "num_images",
                        "chunk_size",
                        "diffusion_steps",
                        "prefix_capacity",
                    )
                },
            }
            with self.assertRaisesRegex(PreparedArtifactError, "preprocessing"):
                load_prepared_fast_artifacts(**dict(kwargs, preprocessing={}))
            with self.assertRaisesRegex(PreparedArtifactError, "deployment"):
                load_prepared_fast_artifacts(**dict(kwargs, chunk_size=50))
            prepared = load_prepared_fast_artifacts(**kwargs)
            prepared.vision_engine.write_bytes(b"damage")
            with self.assertRaisesRegex(PreparedArtifactError, "integrity"):
                load_prepared_fast_artifacts(**kwargs)

    def test_malformed_ready_and_bucket_schema_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready_path = root / "prepared.json"
            for contents in ("{", "[]"):
                ready_path.write_text(contents)
                self.assertIsNone(prepare._read_cached_ready(ready_path))

            deployment, preprocessing = self._generation(root)
            ready = json.loads(ready_path.read_text())
            ready["deployment"].pop("prefix_seq_len_buckets")
            ready_path.write_text(json.dumps(ready))
            with self.assertRaisesRegex(PreparedArtifactError, "bucket"):
                load_prepared_fast_artifacts(
                    checkpoint=root / "model",
                    prepared_dir=root,
                    preprocessing=preprocessing,
                    prefix_seq_len_buckets=None,
                    is_history=False,
                    **{
                        key: deployment[key]
                        for key in (
                            "num_images",
                            "chunk_size",
                            "diffusion_steps",
                            "prefix_capacity",
                        )
                    },
                )

    def test_missing_or_corrupt_layer_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deployment, preprocessing = self._generation(root)
            kwargs = {
                "checkpoint": root / "model",
                "prepared_dir": root,
                "preprocessing": preprocessing,
                "prefix_seq_len_buckets": None,
                "is_history": False,
                **{
                    key: deployment[key]
                    for key in (
                        "num_images",
                        "chunk_size",
                        "diffusion_steps",
                        "prefix_capacity",
                    )
                },
            }
            layer = root / "generations/one/prefix_mlp_fp8/layer_00.engine"
            layer.write_bytes(b"same-size---")
            with self.assertRaisesRegex(PreparedArtifactError, "incomplete"):
                load_prepared_fast_artifacts(**kwargs)


class CommandLineCompatibilityTests(unittest.TestCase):
    def _argv(self, root: Path) -> list[str]:
        return [
            "--checkpoint",
            str(root / "model"),
            "--inputs",
            str(root / "inputs.pt"),
        ]

    def test_canonical_requires_supported_precision_before_orchestration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "model/.opendm/fast_fp8"
            with patch.object(prepare, "_orchestrate") as orchestrate:
                with self.assertRaises(SystemExit):
                    prepare.main(self._argv(root))
                with self.assertRaises(SystemExit):
                    prepare.main(
                        [
                            *self._argv(root),
                            "--prefix-mlp-precision",
                            "bf16",
                        ]
                    )
                with self.assertRaises(SystemExit):
                    legacy_prepare.main(
                        [
                            *self._argv(root),
                            "--prefix-mlp-precision",
                            "bf16",
                        ]
                    )
            orchestrate.assert_not_called()
            self.assertFalse(output.exists())

    def test_legacy_alias_defaults_to_same_explicit_fp8_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def record(args, public_argv):
                calls.append((args, public_argv))

            with patch.object(prepare, "_orchestrate", side_effect=record):
                prepare.main(["--prefix-mlp-precision", "fp8", *self._argv(root)])
                legacy_prepare.main(self._argv(root))

            canonical_args, canonical_argv = calls[0]
            legacy_args, legacy_argv = calls[1]
            self.assertEqual(vars(canonical_args), vars(legacy_args))
            self.assertEqual(
                prepare._normalized_public_argv(canonical_args, canonical_argv),
                prepare._normalized_public_argv(legacy_args, legacy_argv),
            )
            self.assertNotIn("--prefix-mlp-precision", legacy_argv)
            normalized = prepare._normalized_public_argv(legacy_args, legacy_argv)
            stage = prepare._stage_command("preprocess", normalized, root / "staging")
            self.assertEqual(stage[stage.index("-m") + 1], "opendm.infer.prepare_fast")
            self.assertIn("--prefix-mlp-precision", stage)

    def test_precision_name_does_not_change_existing_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            canonical_args = prepare._parser().parse_args(
                [
                    *self._argv(Path(directory)),
                    "--prefix-mlp-precision",
                    "fp8",
                ]
            )
            legacy_args = prepare._parser(legacy_alias=True).parse_args(
                self._argv(Path(directory))
            )
            del legacy_args.prefix_mlp_precision
            kwargs = {
                "deployment": {"prefix_capacity": 1024},
                "preprocessing": {"identity": "same"},
                "checkpoint": [{"sha256": "a" * 64}],
                "sources": [{"sha256": "b" * 64}],
                "selected_media": None,
                "min_seq_len": 576,
                "opt_seq_len": 704,
            }
            current = prepare._invocation_identity(canonical_args, **kwargs)
            legacy = prepare._invocation_identity(legacy_args, **kwargs)

            self.assertEqual(current, legacy)
            self.assertNotIn("prefix_mlp_precision", current)
            self.assertEqual(
                set(current),
                {
                    "deployment",
                    "preprocessing",
                    "checkpoint",
                    "sources",
                    "selected_media",
                    "samples",
                    "seed",
                    "bucket",
                    "min_seq_len",
                    "opt_seq_len",
                    "vision_workspace_gb",
                    "prefix_workspace_gb",
                    "keep_onnx",
                    "config_overrides",
                    "playground",
                    "dataset_name",
                },
            )

    def test_help_names_both_engine_precisions_and_mlp_only_calibration(self):
        help_text = prepare._parser().format_help()
        self.assertIn("vision FP16 engine", help_text)
        self.assertIn("prefix MLP FP8 engines", help_text)
        self.assertIn("Calibration applies only to the prefix MLP", help_text)
        canonical_usage = help_text.partition("\n\n")[0]
        self.assertIn("--prefix-mlp-precision", canonical_usage)
        self.assertNotIn("[--prefix-mlp-precision", canonical_usage)

        legacy_help = prepare._parser(legacy_alias=True).format_help()
        self.assertIn("Legacy prepare_fast_fp8 alias", legacy_help)
        self.assertIn("defaults to fp8", legacy_help)
        self.assertIn("legacy alias default: fp8", legacy_help)
        self.assertIn("[--prefix-mlp-precision", legacy_help.partition("\n\n")[0])


class OrchestrationTests(unittest.TestCase):
    def _args(self, root: Path):
        checkpoint = root / "model"
        checkpoint.mkdir()
        inputs = root / "inputs.pt"
        inputs.write_bytes(b"inputs")
        return prepare._parser().parse_args(
            [
                "--checkpoint",
                str(checkpoint),
                "--prefix-mlp-precision",
                "fp8",
                "--inputs",
                str(inputs),
                "--samples",
                "2",
            ]
        )

    @staticmethod
    def _fake_stage(command):
        staging = (
            Path(command[command.index("--staging-dir") + 1])
            if "--staging-dir" in command
            else None
        )
        module = command[command.index("-m") + 1]
        if module == "opendm.infer.prepare_fast":
            (staging / "stage_status.json").write_text('{"reused": false}')
            metadata = {
                "checkpoint_provenance": "/model",
                "deployment": {
                    "num_images": 2,
                    "chunk_size": 10,
                    "diffusion_steps": 10,
                    "prefix_capacity": 1024,
                    "prefix_seq_len_buckets": [576, 704, 768, 896, 1024],
                    "history": False,
                },
                "preprocessing": {},
                "invocation_identity": {},
                "custom_output": False,
            }
            (staging / "prepare_metadata.json").write_text(json.dumps(metadata))
            (staging / "calibration_inputs.pt").write_bytes(b"inputs")
            (staging / "sampling_manifest.json").write_text("{}")
        elif module == "opendm.infer.build_vision_trt":
            Path(command[command.index("--engine-path") + 1]).write_bytes(b"vision")
        elif module == "opendm.infer.collect_prefix_mlp_calibration":
            Path(command[command.index("--output") + 1]).write_text("{}")
        elif module == "opendm.infer.build_prefix_mlp_fp8_trt":
            bundle = Path(command[command.index("--output-dir") + 1])
            bundle.mkdir()
            (bundle / "prefix_mlp_quant_manifest.json").write_text("{}")

    def test_stages_are_sequential_and_ready_publishes_last(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            calls = []

            def run(command):
                calls.append(command)
                self._fake_stage(command)

            with patch.object(prepare, "_run", side_effect=run):
                prepare._orchestrate(args, [])
            modules = [command[command.index("-m") + 1] for command in calls]
            self.assertEqual(
                modules,
                [
                    "opendm.infer.prepare_fast",
                    "opendm.infer.build_vision_trt",
                    "opendm.infer.collect_prefix_mlp_calibration",
                    "opendm.infer.build_prefix_mlp_fp8_trt",
                ],
            )
            self.assertEqual(
                calls[0].count("--prefix-mlp-precision"),
                1,
            )
            self.assertEqual(
                calls[0][calls[0].index("--prefix-mlp-precision") + 1], "fp8"
            )
            self.assertTrue(
                (args.checkpoint / ".opendm/fast_fp8/prepared.json").is_file()
            )

    def test_unsupported_capacity_is_rejected_before_output_or_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            args.prefix_capacity = 2048
            args.bucket = 2048
            output = args.checkpoint / ".opendm/fast_fp8"

            with (
                patch.object(prepare, "_run") as run,
                self.assertRaisesRegex(ValueError, "larger capacities such as 2048"),
            ):
                prepare._orchestrate(args, [])

            run.assert_not_called()
            self.assertFalse(output.exists())

    def test_complete_cache_skips_all_gpu_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._args(Path(directory))
            calls = []

            def run(command):
                calls.append(command)
                staging = Path(command[command.index("--staging-dir") + 1])
                (staging / "stage_status.json").write_text('{"reused": true}')

            with patch.object(prepare, "_run", side_effect=run):
                prepare._orchestrate(args, [])
            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0][calls[0].index("-m") + 1],
                "opendm.infer.prepare_fast",
            )
            self.assertIn("--prefix-mlp-precision", calls[0])

    def test_failed_build_does_not_publish_or_replace_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._args(root)
            ready = args.checkpoint / ".opendm/fast_fp8/prepared.json"
            ready.parent.mkdir(parents=True)
            ready.write_text('{"old": true}')
            calls = 0

            def run(command):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise RuntimeError("build failed")
                self._fake_stage(command)

            with (
                patch.object(prepare, "_run", side_effect=run),
                self.assertRaisesRegex(RuntimeError, "build failed"),
            ):
                prepare._orchestrate(args, [])
            self.assertEqual(json.loads(ready.read_text()), {"old": True})


if __name__ == "__main__":
    unittest.main()
