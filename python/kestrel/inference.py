"""Evaluators that turn position records into policy logits and a value, with ONNX Runtime or PyTorch."""

from pathlib import Path

import numpy as np

from .features import encode_planes_np, fen_to_record, legal_indices_np


def _expected_value(wdl: np.ndarray) -> np.ndarray:
    wdl = wdl - wdl.max(axis=1, keepdims=True)
    probs = np.exp(wdl)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs[:, 2] - probs[:, 0]


class OnnxEvaluator:
    """Runs an exported model on the CPU with ONNX Runtime, so PyTorch is not needed."""

    def __init__(self, path: Path, threads: int | None = None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.in_planes = int(self.session.get_inputs()[0].shape[1])
        self.description = f"ONNX Runtime CPU, {threads or 'default'} threads"

    def __call__(self, records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy, wdl = self.session.run(None, {"planes": encode_planes_np(records, self.in_planes)})
        return policy, _expected_value(wdl)


class TorchEvaluator:
    """Runs a PyTorch model, used for training time matches and the GPU web server."""

    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.in_planes = model.in_planes
        self.description = f"PyTorch {device}"

    def __call__(self, records: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        from .encoding import encode_planes

        with torch.no_grad():
            raw = torch.from_numpy(np.ascontiguousarray(records)).to(self.device)
            policy, wdl = self.model(encode_planes(raw, self.in_planes))
        return policy.float().cpu().numpy(), _expected_value(wdl.float().cpu().numpy())


def greedy_move(evaluator, fen: str, repetitions: int, legal: list[str]) -> str:
    """Returns the legal move the policy scores highest."""
    policy, _ = evaluator(fen_to_record(fen, repetitions)[None, :])
    scores = policy[0, legal_indices_np(legal, fen.split()[1] == "b")]
    return legal[int(np.argmax(scores))]
