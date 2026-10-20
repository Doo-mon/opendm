"""Regression tests for the FP8 preparation merge with chat inference."""

from __future__ import annotations

import inspect
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from opendm.data.normalize import NormStats, NormStatsFile
from opendm.data.transforms import ChatTokenization, VLMChatTokenization
from opendm.exp.dm05_exp import DM05Exp, DM05InferenceConfig


class _Tokenizer:
    def convert_tokens_to_ids(self, _token):
        return 1


class _Processor:
    tokenizer = _Tokenizer()


def test_chat_tokenization_arguments_are_appended_after_dev_formatted_flag():
    parameters = list(inspect.signature(ChatTokenization).parameters)
    assert parameters[-2:] == ["prompt_is_formatted", "reject_prompt_truncation"]

    positional = ChatTokenization(
        _Processor(), 256, 1024, [], True, False, 32, False, True
    )
    assert positional.prompt_is_formatted is True
    assert positional.reject_prompt_truncation is False


def test_inference_fp8_fields_follow_all_existing_positional_fields():
    existing = [
        "port",
        "diffusion_steps",
        "max_history_images",
        "output_action_dim",
        "image_prompts",
        "backend",
        "vision_trt_engine_path",
        "force_rebuild",
        "prefix_seq_len_buckets",
        "chat_only",
        "enable_vlm_chat",
        "chat_max_new_tokens",
        "chat_max_input_tokens",
        "clip_to_bounds",
        "compose_eef_rot",
        "full_action_mask",
        "use_transformed_state",
        "state_layout",
        "denorm_state",
    ]
    names = [item.name for item in fields(DM05InferenceConfig)]
    assert names == [
        *existing,
        "prefix_mlp_fp8_engine_dir",
        "auto_prepared_fast_artifacts",
        "prepared_fast_artifacts_dir",
    ]

    config = DM05InferenceConfig(
        8000,
        11,
        6,
        14,
        ["Head"],
        "default",
        "vision.engine",
        True,
        [512],
        False,
        False,
        7,
        8,
        False,
        True,
        True,
        True,
        "layout",
        True,
    )
    assert config.denorm_state is True
    assert config.prefix_mlp_fp8_engine_dir is None
    assert config.auto_prepared_fast_artifacts is True


def test_chat_only_runtime_skips_dataset_norm_and_prepared_discovery():
    inference = SimpleNamespace(
        chat_only=True,
        backend="fast",
        prefix_seq_len_buckets=[1024],
        _initialize=Mock(),
    )
    model = object()
    model_config = SimpleNamespace(
        model_name_or_path="checkpoint",
        align_action_rope_to_language=False,
        llm_attn_implementation="auto",
        build_model=Mock(return_value=model),
    )
    data_config = SimpleNamespace(
        _dataset_info=Mock(side_effect=AssertionError("dataset must not be read")),
        norm_stats_path=Mock(side_effect=AssertionError("norm must not be read")),
    )
    exp = object.__new__(DM05Exp)
    exp.inference_config = inference
    exp.model_config = model_config
    exp.data_config = data_config
    exp.use_lora = False

    exp._initialize_inference_runtime()

    assert model_config.align_action_rope_to_language is True
    assert model_config.llm_attn_implementation == "eager"
    model_config.build_model.assert_called_once_with(use_lora=False)
    data_config._dataset_info.assert_not_called()
    data_config.norm_stats_path.assert_not_called()
    inference._initialize.assert_called_once_with(
        model=model,
        model_name_or_path="checkpoint",
        norm_stats_path="",
        n_bins=0,
        model_max_length=0,
        use_absolute_action=False,
    )


def test_chat_only_initializes_vlm_tokenization_without_norm_stats():
    config = DM05InferenceConfig(chat_only=True)
    model = SimpleNamespace(eval=Mock(), to=Mock())
    processor = _Processor()

    with (
        patch(
            "opendm.exp.dm05_exp.AutoProcessor.from_pretrained",
            return_value=processor,
        ),
        patch(
            "opendm.exp.dm05_exp.normalize.load_norm_stats_file",
            side_effect=AssertionError("chat-only must not load norm stats"),
        ),
    ):
        config._initialize(model, "checkpoint", "", 0, 0, False)

    assert isinstance(config.vlm_chat_tokenization, VLMChatTokenization)
    assert config.vlm_chat_tokenization.processor is processor


def test_default_action_runtime_delegates_to_shared_preprocessing():
    config = DM05InferenceConfig(backend="default")
    model = SimpleNamespace(eval=Mock(), to=Mock())
    config._initialize_preprocessing = Mock()

    config._initialize(
        model,
        "checkpoint",
        "norm.json",
        256,
        900,
        True,
        add_state=False,
        is_history=True,
    )

    config._initialize_preprocessing.assert_called_once_with(
        model_name_or_path="checkpoint",
        norm_stats_path="norm.json",
        n_bins=256,
        transform_max_length=900,
        use_absolute_action=True,
        add_state=False,
        is_history=True,
    )


@pytest.mark.parametrize("backend", ["default", "fast"])
def test_action_preprocessing_initializes_vlm_and_action_tokenizers(backend):
    config = DM05InferenceConfig(output_action_dim=1, backend=backend)
    processor = _Processor()
    stats = NormStats(
        mean=np.array([0.0]),
        std=np.array([1.0]),
        q01=np.array([-1.0]),
        q99=np.array([1.0]),
    )
    norm_file = NormStatsFile({"state": stats, "action": stats})

    with (
        patch(
            "opendm.exp.dm05_exp.AutoProcessor.from_pretrained",
            return_value=processor,
        ),
        patch(
            "opendm.exp.dm05_exp.normalize.load_norm_stats_file",
            return_value=norm_file,
        ),
    ):
        config._initialize_preprocessing(
            model_name_or_path="checkpoint",
            norm_stats_path="norm.json",
            n_bins=256,
            transform_max_length=1024,
            use_absolute_action=False,
            add_state=True,
            is_history=False,
            device="cpu",
        )

    assert isinstance(config.vlm_chat_tokenization, VLMChatTokenization)
    assert any(
        isinstance(step, ChatTokenization) for step in config.input_transform.transforms
    )
