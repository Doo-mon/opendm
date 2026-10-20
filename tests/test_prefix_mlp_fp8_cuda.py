"""Opt-in, real TensorRT Q/DQ and multi-bucket CUDA Graph integration test.

Run with OPENDM_TEST_FP8_CUDA=1 on a GPU supported by the installed TensorRT.
"""

import json
import os

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    os.environ.get("OPENDM_TEST_FP8_CUDA") != "1" or not torch.cuda.is_available(),
    reason="requires explicit OPENDM_TEST_FP8_CUDA=1 and CUDA/TensorRT",
)


def test_fp8_qdq_and_bucket_graph_replay(tmp_path):
    from transformers import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import Gemma3MLP

    from opendm.infer.build_prefix_mlp_fp8_trt import build_layer_engine
    from opendm.infer.dm05_prefix_mlp_trt import PrefixMLPFP8Backend
    from opendm.infer.prefix_mlp_fp8_contract import (
        PREFIX_MLP_FORMAT_VERSION,
        SPEC,
        current_build_metadata,
        prefix_mlp_model_contract,
        prefix_source_model_sha256,
        sha256_file,
    )

    config = Gemma3TextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    )
    language = torch.nn.Module()
    language.config = config
    layer = torch.nn.Module()
    torch.manual_seed(123)
    layer.mlp = Gemma3MLP(config).to(device="cuda", dtype=torch.bfloat16).eval()
    language.layers = torch.nn.ModuleList([layer])
    vlm = torch.nn.Module()
    vlm.config = config
    vlm.model = torch.nn.Module()
    vlm.model.language_model = language
    model_contract = prefix_mlp_model_contract(language)
    source_sha = prefix_source_model_sha256(vlm)

    def record(path, key):
        return {
            key: path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    vision = tmp_path / "fixture-vision.engine"
    vision.write_bytes(b"identity-only fixture; no vision execution in this MLP test")
    engine_path = tmp_path / "layer_00.engine"
    build_layer_engine(
        engine_path=engine_path,
        mlp=layer.mlp,
        hidden_size=256,
        input_amax=4.0,
        down_amax=2.0,
        min_seq_len=32,
        opt_seq_len=64,
        max_seq_len=96,
        workspace_gb=0.25,
    )
    calibration = {
        "format_version": PREFIX_MLP_FORMAT_VERSION,
        "spec": SPEC,
        "model_contract": model_contract,
        "prefix_source_sha256": source_sha,
        "batch_size": 1,
        "sample_count": 2,
        "distinct_sample_count": 2,
        "vision_engine": record(vision, "file"),
        "layers": {
            "0": {
                key: {"visible_rows": {"rows": 128, "elements": 32768, "max_abs": amax}}
                for key, amax in (("mlp_input", 4.0), ("down_input", 2.0))
            }
        },
    }
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    manifest = {
        "format_version": PREFIX_MLP_FORMAT_VERSION,
        "spec": SPEC,
        "model_contract": model_contract,
        "prefix_source_sha256": source_sha,
        "contract": {
            "batch_size": 1,
            "hidden_size": 256,
            "layer_indices": [0],
            "min_seq_len": 32,
            "opt_seq_len": 64,
            "max_seq_len": 96,
        },
        "calibration": record(calibration_path, "file"),
        "build_metadata": current_build_metadata(),
        "engines": {
            "0": {
                **record(engine_path, "engine"),
                "input_range": 4.0,
                "down_range": 2.0,
            }
        },
    }
    (tmp_path / "prefix_mlp_quant_manifest.json").write_text(json.dumps(manifest))
    backend = PrefixMLPFP8Backend(
        vlm,
        tmp_path,
        (32, 64, 96),
        device="cuda",
        vision_engine_identity=record(vision, "file"),
    )

    def qdq(value, amax):
        scale = max(float(amax) / 448, 1e-8)
        return (
            (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            * scale
        ).half()

    def reference(x):
        x = qdq(x.half(), 4.0)
        weights = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            weight = getattr(layer.mlp, name).weight.half()
            weights[name] = qdq(weight, weight.abs().max())
        gate = torch.nn.functional.linear(x, weights["gate_proj"])
        up = torch.nn.functional.linear(x, weights["up_proj"])
        product = torch.nn.functional.gelu(gate, approximate="tanh") * up
        return torch.nn.functional.linear(
            qdq(product, 2.0), weights["down_proj"]
        ).bfloat16()

    stream = torch.cuda.Stream()
    graphs, buffers = {}, {}
    with torch.inference_mode(), torch.cuda.stream(stream):
        for bucket in (32, 64, 96):
            x = torch.randn((1, bucket, 256), device="cuda", dtype=torch.bfloat16)
            y = torch.empty_like(x)
            for _ in range(2):
                backend.forward_into(0, x, y)
            stream.synchronize()
            expected = reference(x)
            torch.testing.assert_close(y, expected, atol=0.03, rtol=0.03)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                backend.forward_into(0, x, y)
            graphs[bucket], buffers[bucket] = graph, (x, y, y.clone())
        # Later contexts/captures must not overwrite earlier context metadata.
        for bucket in (32, 96, 64, 32):
            graphs[bucket].replay()
            _, actual, expected = buffers[bucket]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for bucket in (32, 96, 32):
            x, y, _ = buffers[bucket]
            x.zero_()
            graphs[bucket].replay()
            assert bool((y == 0).all())
            x.normal_()
            graphs[bucket].replay()
            torch.testing.assert_close(y, reference(x), atol=0.03, rtol=0.03)
    stream.synchronize()
