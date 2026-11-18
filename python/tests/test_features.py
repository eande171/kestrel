"""Checks the NumPy encoding against the PyTorch one and the ONNX evaluator against PyTorch."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kestrel.encoding import encode_planes, policy_index  # noqa: E402
from kestrel.features import encode_planes_np, policy_index_np, read_records  # noqa: E402
from kestrel.inference import OnnxEvaluator, TorchEvaluator  # noqa: E402
from kestrel.model import load_model  # noqa: E402


def main(shard: str, checkpoint: str, onnx_path: str, samples: int = 5000) -> None:
    records = read_records(Path(shard))[:samples]
    raw = torch.from_numpy(records)

    torch_planes = encode_planes(raw).numpy()
    numpy_planes = encode_planes_np(records)
    assert np.array_equal(torch_planes, numpy_planes), "plane encodings differ"

    black = records[:, 32] == 1
    torch_idx = policy_index(raw[:, 36], raw[:, 37], raw[:, 38], raw[:, 32].bool()).numpy()
    numpy_idx = policy_index_np(records[:, 36], records[:, 37], records[:, 38], black)
    assert np.array_equal(torch_idx, numpy_idx), "policy indices differ"

    model, _ = load_model(Path(checkpoint), "cpu")
    torch_policy, torch_value = TorchEvaluator(model, "cpu")(records)
    onnx_policy, onnx_value = OnnxEvaluator(Path(onnx_path), threads=4)(records)
    agree = float((torch_policy.argmax(1) == onnx_policy.argmax(1)).mean())
    value_diff = float(np.abs(torch_value - onnx_value).max())
    assert agree > 0.999, f"top move agreement only {agree:.4f}"
    assert value_diff < 1e-3, f"value difference {value_diff:.2e}"
    print(f"ok: {len(records)} positions, planes and indices identical, "
          f"ONNX top move agreement {agree:.2%}, max value difference {value_diff:.1e}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
