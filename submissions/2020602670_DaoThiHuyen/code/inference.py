"""inference.py - các phương pháp suy luận (TTA, ensemble, temperature scaling, fusion BN)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.to(device)
    model.eval()
    filenames, y_true, logits = [], [], []
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 3:
                x, y, name = batch
            else:
                x, y = batch
                name = [str(i) for i in range(len(y))]
            x = x.to(device)
            if view is not None:
                view_x = view(x)
                if isinstance(view_x, (list, tuple)):
                    outs = []
                    for xx in view_x:
                        outs.append(model(xx.to(device)))
                    logits_batch = torch.stack(outs, dim=0).mean(0)
                else:
                    logits_batch = model(view_x.to(device))
            else:
                logits_batch = model(x)
            filenames.extend(list(name))
            y_true.append(y.numpy())
            logits.append(logits_batch.cpu().numpy())
    return filenames, np.concatenate(y_true, axis=0), np.concatenate(logits, axis=0)


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """Trả về list các crop từ 4 góc + giữa."""
    n, c, h, w = x.shape
    base = []
    for top, left in [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), ((h - crop) // 2, (w - crop) // 2)]:
        if top + crop > h or left + crop > w:
            continue
        base.append(x[:, :, top:top + crop, left:left + crop])
    return base


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`."""
    outputs = []
    for size in sizes:
        outputs.append(F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False))
    return outputs


def aggregate_views(logits_per_view, space: str = "prob"):
    """Trung bình softmax hoặc logit của các view."""
    logits = [np.asarray(v, dtype=np.float64) for v in logits_per_view]
    if space == "prob":
        arr = np.stack([_softmax(v) for v in logits], axis=0)
        return arr.mean(axis=0)
    if space == "logit":
        arr = np.stack(logits, axis=0)
        return _softmax(arr.mean(axis=0))
    raise ValueError(f"space phải là 'prob' hoặc 'logit', nhận {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình."""
    arr = np.stack([np.asarray(p, dtype=np.float64) for p in list_of_probs], axis=0)
    return arr.mean(axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm T bằng tìm kiếm lưới (nhiều giá trị) trên log-scale."""
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)
    best_T = 1.0
    best_loss = float("inf")
    for T in np.geomspace(0.05, 10.0, 250):
        probs = _softmax(logits / T)
        loss = -(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None))).mean()
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    return best_T


def apply_temperature(logits, T: float):
    """Trả về softmax(logits/T)."""
    logits = np.asarray(logits, dtype=np.float64)
    return _softmax(logits / T)


def fuse_conv_bn(model):
    """Gộp BatchNorm vào Conv nếu có."""
    model.eval()

    def fuse_seq(seq):
        for i in range(len(seq) - 1):
            conv = seq[i]
            bn = seq[i + 1]
            if not isinstance(conv, nn.Conv2d) or not isinstance(bn, nn.BatchNorm2d):
                continue
            fused = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                conv.kernel_size,
                conv.stride,
                conv.padding,
                conv.dilation,
                conv.groups,
                bias=True,
                padding_mode=conv.padding_mode,
            )
            if conv.bias is not None:
                fused.bias.data = conv.bias.data.clone()
            running_var = bn.running_var + bn.eps
            scale = bn.weight / torch.sqrt(running_var)
            fused.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)
            if conv.bias is not None:
                fused.bias.data = (
                    conv.bias.data + bn.bias + (bn.weight / torch.sqrt(running_var)) * (conv.bias.data - bn.running_mean)
                )
            else:
                fused.bias.data = bn.bias + (bn.weight / torch.sqrt(running_var)) * (-bn.running_mean)
            seq[i] = fused
            seq[i + 1] = nn.Identity()

    for module in model.modules():
        if isinstance(module, nn.Sequential):
            fuse_seq(module)
    return model
