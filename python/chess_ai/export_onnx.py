"""Exports a checkpoint to ONNX and checks it against PyTorch with onnxruntime."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .encoding import NUM_PLANES, POLICY_SIZE
from .model import ChessNet, count_parameters


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/main")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    run = Path(args.run)
    ckpt = torch.load(run / "checkpoints" / args.checkpoint, map_location="cpu", weights_only=False)
    model = ChessNet(**ckpt["model_config"])
    model.load_state_dict(ckpt["model"])
    model.eval()

    out = Path(args.out) if args.out else run / "model.onnx"
    dummy = torch.zeros(1, NUM_PLANES, 8, 8)
    torch.onnx.export(
        model,
        (dummy,),
        str(out),
        input_names=["planes"],
        output_names=["policy", "wdl"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "wdl": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    size = out.stat().st_size
    extra = out.with_name(out.name + ".data")
    if extra.exists():
        size += extra.stat().st_size

    import onnxruntime as ort

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = torch.randint(0, 2, (16, NUM_PLANES, 8, 8)).float()
    with torch.no_grad():
        ref_policy, ref_wdl = model(x)
    policy, wdl = session.run(None, {"planes": x.numpy()})
    assert policy.shape == (16, POLICY_SIZE)
    policy_diff = float(np.abs(policy - ref_policy.numpy()).max())
    wdl_diff = float(np.abs(wdl - ref_wdl.numpy()).max())
    print(f"exported {out}")
    print(f"parameters {count_parameters(model):,}, file size {size / 1e6:.2f} MB")
    print(f"max abs difference vs PyTorch: policy {policy_diff:.2e}, wdl {wdl_diff:.2e}")


if __name__ == "__main__":
    main()
