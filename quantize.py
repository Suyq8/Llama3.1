import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from module import Transformer


def dynamically_quantize_per_channel(x, quant_min, quant_max, target_dtype):
    # assumes symmetric quantization
    # assumes axis == 0
    # assumes dense memory format
    # TODO(future): relax ^ as needed

    device = 'cpu'
    x = x.to(device=device)

    # default setup for affine quantization of activations
    eps = torch.finfo(torch.float32).eps

    # get min and max
    min_val, max_val = torch.aminmax(x, dim=1)

    # calculate scales and zero_points based on min and max
    # reference: https://fburl.com/code/srbiybme
    min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
    max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

    # reference: https://fburl.com/code/4wll53rk
    max_val_pos = torch.max(-min_val_neg, max_val_pos)
    scales = max_val_pos / (float(quant_max - quant_min) / 2)
    # ensure scales is the same dtype as the original tensor
    scales = torch.clamp(scales, min=eps).to(dtype=x.dtype)
    zero_points = torch.zeros(min_val_neg.size(), dtype=torch.int64, device=device)

    # quantize based on qmin/qmax/scales/zp
    # reference: https://www.internalfb.com/code/fbsource/[8edc275012b1]/fbcode/caffe2/torch/ao/quantization/fx/_decomposed.py?lines=63
    x_div = x / scales.unsqueeze(-1)
    x_round = torch.round(x_div)
    x_zp = x_round + zero_points.unsqueeze(-1)
    quant = torch.clamp(x_zp, quant_min, quant_max).to(dtype=target_dtype)

    return quant, scales, zero_points


def replace_linear_weight_only_int8_per_channel(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                WeightOnlyInt8Linear(child.in_features, child.out_features),
            )
        else:
            replace_linear_weight_only_int8_per_channel(child)


class WeightOnlyInt8QuantHandler:
    def __init__(self, mod):
        self.mod = mod

    @torch.no_grad()
    def create_quantized_state_dict(self):
        cur_state_dict = self.mod.state_dict()
        for fqn, mod in self.mod.named_modules():
            if isinstance(mod, torch.nn.Linear):
                int8_weight, scales, _ = dynamically_quantize_per_channel(
                    mod.weight.float(), -128, 127, torch.int8
                )
                cur_state_dict[f"{fqn}.weight"] = int8_weight
                cur_state_dict[f"{fqn}.scales"] = scales.to(mod.weight.dtype)

        return cur_state_dict

    def convert_for_runtime(self):
        replace_linear_weight_only_int8_per_channel(self.mod)
        return self.mod


class WeightOnlyInt8Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.empty((out_features, in_features), dtype=torch.int8)
        )
        self.register_buffer("scales", torch.ones(out_features, dtype=torch.bfloat16))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(dtype=input.dtype)) * self.scales


def quantize(quantize_args: argparse.Namespace):
    assert quantize_args.checkpoint_path.is_file(), quantize_args.checkpoint_path

    #device = "cpu"
    precision = torch.bfloat16

    print("Loading model ...")
    t0 = time.time()

    with torch.device("meta"):
        model = Transformer(quantize_args)

    checkpoint = torch.load(
        str(quantize_args.checkpoint_path), map_location="cpu", weights_only=True
    )
    model.load_state_dict(checkpoint, strict=False, assign=True)
    model = model.to(dtype=precision, device=quantize_args.device)
    del checkpoint

    print(
        "Quantizing model weights for int8 weight-only symmetric per-channel quantization"
    )
    quant_handler = WeightOnlyInt8QuantHandler(model)
    quantized_state_dict = quant_handler.create_quantized_state_dict()

    dir_name = quantize_args.save_path
    base_name = quantize_args.checkpoint_path.name
    new_base_name = base_name.replace(".pth", f"{quantize_args.label}int8.pth")

    quantize_path = dir_name / new_base_name
    print(f"Writing quantized weights to {quantize_path}")
    quantize_path.unlink(missing_ok=True)  # remove existing file if one already there
    torch.save(quantized_state_dict, quantize_path)
    print(f"Quantization complete took {time.time() - t0:.02f} seconds")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize a model.")
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        required=True,
        help="Path to the model checkpoint to be quantized.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Path to save quantized model.",
    )
    parser.add_argument(
        "--label", type=str, default="_", help="label to add to output filename"
    )
    parser.add_argument("--vocab_size", type=int, default=128256)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--n_layer", type=int, default=32)
    parser.add_argument("--n_head", type=int, default=32)
    parser.add_argument("--n_kv_head", type=int, default=8)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--ffn_dim_multiplier", type=float, default=1.3)
    parser.add_argument("--norm_eps", type=float, default=1e-5)
    parser.add_argument("--rope_theta", type=float, default=500000)
    parser.add_argument("--use_scaled_rope", type=bool, default=True)
    parser.add_argument("--multiple_of", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use_flash", type=bool, default=False)

    args = parser.parse_args()
    quantize(args)
