"""Exports a checkpoint to ONNX and checks it against PyTorch with onnxruntime, optionally with an int8 copy."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .features import POLICY_SIZE, encode_planes_np, read_records
from .model import count_parameters, load_model

CALIBRATION_POSITIONS = 2000
CHECK_POSITIONS = 10000


def quantize_trunk(source: Path, out: Path, calibration: np.ndarray) -> None:
    """Writes an int8 copy of the model with the stem and residual trunk quantized and both heads left in float."""
    import onnx
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                          quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.batches = iter([{"planes": encode_planes_np(calibration[i:i + 50])}
                                 for i in range(0, len(calibration), 50)])

        def get_next(self):
            return next(self.batches, None)

    with tempfile.TemporaryDirectory() as folder:
        prepared = Path(folder) / "prepared.onnx"
        quant_pre_process(str(source), str(prepared))
        trunk = [n.name for n in onnx.load(str(prepared)).graph.node
                 if n.op_type == "Conv" and n.name.startswith(("/stem", "/trunk"))]
        quantize_static(str(prepared), str(out), Reader(), quant_format=QuantFormat.QDQ, nodes_to_quantize=trunk,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                        calibrate_method=CalibrationMethod.Percentile)


def int8_agreement(full: Path, int8: Path, records: np.ndarray) -> tuple[float, float]:
    """Returns how often the int8 model's top move matches the float model's, and the mean value difference."""
    import onnxruntime as ort

    planes = encode_planes_np(records)
    results = [ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"planes": planes})
               for path in (full, int8)]
    values = []
    for _, wdl in results:
        probs = np.exp(wdl - wdl.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        values.append(probs[:, 2] - probs[:, 0])
    agree = float((results[0][0].argmax(1) == results[1][0].argmax(1)).mean())
    return agree, float(np.abs(values[0] - values[1]).mean())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="runs/main")
    p.add_argument("--checkpoint", default="best.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--int8", action="store_true", help="also write model_int8.onnx with the trunk quantized")
    p.add_argument("--calibration", type=Path, default=None,
                   help="shard of positions for int8 calibration, default the val shard of the run's last data folder")
    args = p.parse_args(argv)

    run = Path(args.run)
    model, _ = load_model(run / "checkpoints" / args.checkpoint, "cpu")

    out = Path(args.out) if args.out else run / "model.onnx"
    dummy = torch.zeros(1, model.in_planes, 8, 8)
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
    x = torch.randint(0, 2, (16, model.in_planes, 8, 8)).float()
    with torch.no_grad():
        ref_policy, ref_wdl = model(x)
    policy, wdl = session.run(None, {"planes": x.numpy()})
    assert policy.shape == (16, POLICY_SIZE)
    policy_diff = float(np.abs(policy - ref_policy.numpy()).max())
    wdl_diff = float(np.abs(wdl - ref_wdl.numpy()).max())
    print(f"exported {out}")
    print(f"parameters {count_parameters(model):,}, file size {size / 1e6:.2f} MB")
    print(f"max abs difference vs PyTorch: policy {policy_diff:.2e}, wdl {wdl_diff:.2e}")

    if args.int8:
        shard = args.calibration or Path(json.loads((run / "config.json").read_text())["data"][-1]) / "val_000.bin"
        rows = read_records(shard)
        int8_out = out.with_name(out.stem + "_int8.onnx")
        quantize_trunk(out, int8_out, rows[:CALIBRATION_POSITIONS])
        agree, value_diff = int8_agreement(out, int8_out, rows[-CHECK_POSITIONS:])
        print(f"exported {int8_out}, file size {int8_out.stat().st_size / 1e6:.2f} MB, calibrated on {shard}")
        print(f"int8 against float on {CHECK_POSITIONS} other positions: same top move {agree:.1%}, "
              f"mean value difference {value_diff:.4f}")


if __name__ == "__main__":
    main()
