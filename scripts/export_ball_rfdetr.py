"""RF-DETR 球偵測權重 → Core ML。

複製自教授 PingPongTracker/Tools/export_coreml.py (coremltools 修補原封不動),
只改三處 (不修改教授 repo):
  1. --resolution: 原版寫死 512 (= RFDETRSmall)。預設改為模型類別本身的解析度。
  2. 權重嚴格比對: 原版 --model-class 選錯時 rfdetr 會悄悄丟掉對不上的參數
     (2026-09-30 的權重是 RFDETRLarge, 用預設 Small 轉會少掉一層 decoder 且不報錯)。
  3. 預設輸出改到本專案 Models/, 不覆蓋教授 app 內建的 BallDetector.mlpackage。

用法 (需要 ~/Developer/tt-export-venv311 環境):
  python scripts/export_ball_rfdetr.py checkpoints/ball/pingpong20260930_rfdetr.pt \
      --model-class RFDETRLarge --out Models/BallDetector_rfdetr_20260930.mlpackage
"""
import math

import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
import rfdetr
from coremltools.converters.mil.frontend.torch import ops as _torch_ops
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op

# coremltools' `_cast` (used by the traced `aten::Int`/`aten::Bool` ops) calls
# `int(x.val)`/`bool(x.val)` on a folded constant that is a length-1 numpy array
# rather than a true 0-d scalar. NumPy 2.x removed implicit scalar conversion for
# any ndim > 0 array (even size-1), so this raises
# "only 0-dimensional arrays can be converted to Python scalars". Patch to use
# `.item()`, which numpy still supports for any size-1 array regardless of ndim.
_orig_cast = _torch_ops._cast


def _patched_cast(context, node, dtype, dtype_name):
    inputs = _torch_ops._get_inputs(context, node, expected=1)
    x = inputs[0]
    if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
        raise ValueError("input to cast must be either a scalar or a length 1 tensor")
    if x.can_be_folded_to_const():
        val = x.val
        scalar = val.item() if hasattr(val, "item") else val
        res = _torch_ops.mb.const(val=dtype(scalar), name=node.name)
    elif len(x.shape) > 0:
        squeezed = _torch_ops.mb.squeeze(x=x, name=node.name + "_item")
        res = _torch_ops.mb.cast(x=squeezed, dtype=dtype_name, name=node.name)
    else:
        res = _torch_ops.mb.cast(x=x, dtype=dtype_name, name=node.name)
    context.add(res, node.name)


_torch_ops._cast = _patched_cast

# `torch._shape_as_tensor` is used in transformer.py's spatial_shapes construction
# specifically to keep shape values as *dynamic* tensors (their comment: "keep symbolic
# ... in ONNX", i.e. so a re-exported ONNX graph supports variable input resolutions).
# We only ever export one fixed 512x512 shape for this mobile app, and coremltools'
# shape-inference doesn't constant-fold this private op the way it folds plain
# `.shape` access, which was producing a downstream `torch.linspace(steps=<dynamic>)` ->
# `meshgrid` that coremltools' MIL frontend can't convert ("non-1d tensor"). Since our
# input shape is static, replacing it with a plain constant tensor of the same values is
# numerically identical and lets everything downstream constant-fold as intended.
torch._shape_as_tensor = lambda x: torch.tensor(list(x.shape), dtype=torch.int64)

# The traced per-level `torch.linspace(0, height-1, height)` calls in
# gen_encoder_output_proposals fold to constants of shape (level_size, 1) instead of
# (level_size,) — an extra trailing singleton dim picked up somewhere in tracing this
# private-op-derived shape math. coremltools' meshgrid only accepts rank-1 inputs, and
# both observed inputs are foldable constants, so squeezing away the trailing size-1
# dim is value-preserving. Reimplements the rest of coremltools' own meshgrid body
# (view -> tile -> optional transpose) verbatim, just starting from the fixed inputs.
@register_torch_op(torch_alias=["meshgrid.indexing"], override=True)
def meshgrid(context, node):
    inputs = _torch_ops._get_inputs(context, node, expected=[1, 2])
    tensor_inputs = list(inputs[0])
    indexing = inputs[1].val if len(inputs) > 1 else "ij"
    indexing = _torch_ops._get_kwinputs(context, node, "indexing", default=[indexing])[0]

    fixed_inputs = []
    for i, t in enumerate(tensor_inputs):
        if t.rank > 1 and math.prod(t.shape[1:]) == 1:
            t = _torch_ops.mb.reshape(x=t, shape=(t.shape[0],), name=f"{node.name}_flatten_{i}")
        fixed_inputs.append(t)
    tensor_inputs = fixed_inputs

    if any(t.rank > 1 for t in tensor_inputs):
        raise ValueError("meshgrid received non-1d tensor.")
    if indexing not in ("ij", "xy"):
        raise ValueError(f"indexing mode {indexing} not supported")

    result_shape = [t.shape[0] for t in tensor_inputs]
    size = len(tensor_inputs)
    grids = []
    for i in range(size):
        view_shape = [1] * size
        view_shape[i] = -1
        view = _torch_ops.mb.reshape(x=tensor_inputs[i], shape=tuple(view_shape), name=f"{node.name}_view_{i}")
        reps = result_shape.copy()
        reps[i] = 1
        res = _torch_ops.mb.tile(x=view, reps=reps, name=f"{node.name}_expand_{i}")
        if indexing == "xy":
            perm = [1, 0] + list(range(2, size))
            res = _torch_ops.mb.transpose(x=res, perm=perm, name=f"{node.name}_transpose_{i}")
        grids.append(res)
    context.add(tuple(grids), node.name)


# MSDeformAttn.forward builds a literal (batch, len_query, n_heads, n_levels, n_points, 2)
# sampling_offsets/sampling_locations tensor — rank 6. Core ML caps tensor rank at 5.
# We only ever export batch_size=1, so folding the batch axis away throughout this one
# module (restoring it afterwards for the rest of the transformer, which expects
# (batch, seq, C) tensors) is value-preserving and gets every intermediate to rank <= 5.
# ms_deform_attn_core_pytorch already collapses batch*n_heads together internally for
# grid_sample, so only the *pre*-core-function tensors needed rewriting.
import rfdetr.models.ops.modules.ms_deform_attn as _msda_mod
from rfdetr.utilities.tensors import _bilinear_grid_sample


def _patched_core_pytorch_batchless(value, value_spatial_shapes, sampling_locations, attention_weights, value_spatial_shapes_hw=None):
    # value: (n_heads, head_dim, spatial_size)
    # sampling_locations: (len_query, n_heads, num_levels, num_points, 2)
    # attention_weights: (len_query, n_heads, num_levels*num_points)
    n_heads, head_dim, _ = value.shape
    len_query, _, num_levels, num_points, _ = sampling_locations.shape
    shapes = value_spatial_shapes_hw if value_spatial_shapes_hw is not None else value_spatial_shapes
    value_list = value.split([height * width for height, width in shapes], dim=2)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level_index, (height, width) in enumerate(shapes):
        value_l_ = value_list[level_index].view(n_heads, head_dim, height, width)
        sampling_grid_l_ = sampling_grids[:, :, level_index].transpose(0, 1)
        sampling_value_l_ = _bilinear_grid_sample(value_l_, sampling_grid_l_, padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    attention_weights = attention_weights.transpose(0, 1).reshape(n_heads, 1, len_query, num_levels * num_points)
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(n_heads * head_dim, len_query)
    return output.transpose(0, 1).contiguous()


def _patched_msda_forward(
    self,
    query,
    reference_points,
    input_flatten,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask=None,
    input_spatial_shapes_hw=None,
):
    batch_size, len_query, _ = query.shape
    assert batch_size == 1, "export-time MSDeformAttn patch assumes batch_size=1"
    _, len_input, _ = input_flatten.shape

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))

    q = query[0]
    sampling_offsets = self.sampling_offsets(q).view(len_query, self.n_heads, self.n_levels, self.n_points, 2)
    attention_weights = self.attention_weights(q).view(len_query, self.n_heads, self.n_levels * self.n_points)
    rp = reference_points[0]

    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
        sampling_locations = rp[:, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, :, None, :]
    elif reference_points.shape[-1] == 4:
        sampling_locations = (
            rp[:, None, :, None, :2] + sampling_offsets / self.n_points * rp[:, None, :, None, 2:] * 0.5
        )
    else:
        raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]} instead.")

    attention_weights = torch.nn.functional.softmax(attention_weights, -1)
    value = value.transpose(1, 2).contiguous().view(batch_size, self.n_heads, self.d_model // self.n_heads, len_input)[0]

    output = _patched_core_pytorch_batchless(
        value,
        input_spatial_shapes,
        sampling_locations,
        attention_weights,
        value_spatial_shapes_hw=input_spatial_shapes_hw,
    )
    output = self.output_proj(output.unsqueeze(0))
    return output


_msda_mod.MSDeformAttn.forward = _patched_msda_forward

import argparse
import os

_parser = argparse.ArgumentParser(description="Export an RF-DETR checkpoint to Core ML for PingPongTracker.")
_parser.add_argument("checkpoint", help="Path to the trained .pt checkpoint (e.g. RFDETRSmall weights).")
_parser.add_argument(
    "--out",
    default=os.path.join(os.path.dirname(__file__), "..", "Models", "BallDetector_rfdetr.mlpackage"),
    help="Output .mlpackage path (預設存到本專案 Models/, 不覆蓋教授 app 的模型)",
)
_parser.add_argument("--model-class", default="RFDETRSmall", help="rfdetr class name, e.g. RFDETRSmall/RFDETRMedium/RFDETRLarge.")
_parser.add_argument("--resolution", type=int, default=None,
                     help="輸入邊長 (預設用模型類別本身的解析度, 例如 Small=512, Large=704)")
_args = _parser.parse_args()

CKPT = _args.checkpoint
OUT = os.path.abspath(_args.out)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ExportWrapper(nn.Module):
    """Bakes ImageNet normalization into the graph so the Core ML input node can
    just take a raw 0-255 RGB image (scale=1/255, bias=0) via ct.ImageType."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        coord, cls = self.base(x)
        return coord, cls


print("Loading checkpoint...")
model_class = getattr(rfdetr, _args.model_class)
wrapper_model = model_class(pretrain_weights=CKPT, num_classes=1)
torch_model = wrapper_model.model.model
torch_model.eval()

# 嚴格比對: checkpoint 的每個參數都要在模型裡、且形狀相同, 否則代表 --model-class 選錯
_ckpt_sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
_model_sd = torch_model.state_dict()
_extra = [k for k in _ckpt_sd if k not in _model_sd]
_bad = [k for k in _ckpt_sd if k in _model_sd and _ckpt_sd[k].shape != _model_sd[k].shape]
if _extra or _bad:
    raise SystemExit(f"checkpoint 與 {_args.model_class} 不符: 多出 {len(_extra)} 個參數, "
                     f"{len(_bad)} 個形狀不合 (例: {(_extra + _bad)[:3]})。請改用正確的 --model-class")
print(f"Weights match {_args.model_class}: {len(_ckpt_sd)} tensors")

RESOLUTION = _args.resolution or wrapper_model.model_config.resolution
print("Resolution:", RESOLUTION)
torch_model.export()  # rebinds forward -> forward_export across submodules

export_model = ExportWrapper(torch_model).eval()

example = torch.rand(1, 3, RESOLUTION, RESOLUTION)
print("Tracing...")
with torch.no_grad():
    traced = torch.jit.trace(export_model, example, strict=False)
    coord, cls = traced(example)
    print("pred_boxes shape:", coord.shape, "pred_logits shape:", cls.shape)

print("Converting to Core ML...")
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.ImageType(name="image", shape=(1, 3, RESOLUTION, RESOLUTION), scale=1.0 / 255.0, bias=[0, 0, 0], color_layout=ct.colorlayout.RGB)
    ],
    outputs=[
        ct.TensorType(name="pred_boxes"),
        ct.TensorType(name="pred_logits"),
    ],
    minimum_deployment_target=ct.target.iOS17,
    compute_units=ct.ComputeUnit.ALL,
    convert_to="mlprogram",
)
mlmodel.short_description = f"Ping pong ball detector ({_args.model_class} @ {RESOLUTION}, single class)."
# 供下游 (landing_points.py 等) 自動判斷輸入尺寸與輸出格式
mlmodel.user_defined_metadata["arch"] = "rfdetr"
mlmodel.user_defined_metadata["model_class"] = _args.model_class
mlmodel.user_defined_metadata["imgsz"] = str(RESOLUTION)
mlmodel.save(OUT)
print("Saved to", OUT)
