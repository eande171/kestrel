"""Exports a checkpoint as raw weights for the Rust network, with batch norm folded into the convolutions."""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

from .model import load_model

MAGIC = b"KSTN"
VERSION = 1


def fold(conv: torch.nn.Conv2d, bn: torch.nn.BatchNorm2d | None) -> tuple[np.ndarray, np.ndarray]:
    """Returns a convolution's weight and bias with the following batch norm applied."""
    weight = conv.weight.detach().double()
    bias = conv.bias.detach().double() if conv.bias is not None else torch.zeros(weight.shape[0], dtype=torch.float64)
    if bn is not None:
        scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
        weight = weight * scale[:, None, None, None]
        bias = (bias - bn.running_mean.double()) * scale + bn.bias.detach().double()
    return weight.numpy(), bias.numpy()


def conv3x3(conv, bn) -> list[np.ndarray]:
    weight, bias = fold(conv, bn)
    return [weight.transpose(2, 3, 1, 0), bias]


def conv1x1(conv, bn=None) -> list[np.ndarray]:
    weight, bias = fold(conv, bn)
    return [weight[:, :, 0, 0].T, bias]


def linear(layer: torch.nn.Linear) -> list[np.ndarray]:
    return [layer.weight.detach().double().numpy().T, layer.bias.detach().double().numpy()]


def tensors(model) -> list[np.ndarray]:
    """Returns every array in the order the Rust loader reads them."""
    out = conv3x3(model.stem[0], model.stem[1])
    for block in model.trunk:
        out += conv3x3(block.conv1, block.bn1) + conv3x3(block.conv2, block.bn2)
    out += conv3x3(model.policy_conv[0], model.policy_conv[1])
    out += conv1x1(model.query) + conv1x1(model.key)
    out.append(model.square_bias.detach().double().numpy())
    out += conv1x1(model.value[0], model.value[1])
    out += linear(model.value[4]) + linear(model.value[6])
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/evals6")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    run = Path(args.run)
    model, _ = load_model(run / "checkpoints" / args.checkpoint, "cpu")
    cfg = model.config
    out = Path(args.out) if args.out else run / "model.kstn"
    arrays = tensors(model)
    with open(out, "wb") as f:
        f.write(MAGIC + struct.pack("<5I", VERSION, cfg["blocks"], cfg["channels"], cfg["key_dim"], cfg["in_planes"]))
        for array in arrays:
            f.write(np.ascontiguousarray(array, dtype="<f4").tobytes())
    print(f"exported {out}, {sum(a.size for a in arrays):,} weights, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
