"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC."""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp và (tùy chọn) đóng băng backbone."""
    model_name = SUGGESTED_BACKBONES.get(name, name)
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_param_ids(model):
    head_ids = set()
    for module_name, module in model.named_modules():
        if module_name == "":
            continue
        if any(token in module_name.lower() for token in ("head", "classifier", "fc", "pre_logits")):
            for param in module.parameters(recurse=False):
                head_ids.add(id(param))
    if hasattr(model, "get_classifier"):
        for param in model.get_classifier().parameters():
            head_ids.add(id(param))
    return head_ids


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head."""
    head_ids = _head_param_ids(model)
    for name, param in model.named_parameters():
        if id(param) in head_ids:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành các nhóm backbone/head cho AdamW."""
    backbone_w, backbone_b = [], []
    head = []
    head_ids = _head_param_ids(model)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in head_ids or any(token in name.lower() for token in ("head", "classifier", "fc", "pre_logits")):
            head.append(param)
        elif param.ndim <= 1:
            backbone_b.append(param)
        else:
            backbone_w.append(param)

    groups = []
    if backbone_w:
        groups.append({"params": backbone_w, "lr": lr_backbone, "weight_decay": weight_decay})
    if backbone_b:
        groups.append({"params": backbone_b, "lr": lr_backbone, "weight_decay": 0.0})
    if head:
        groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
    return groups


def count_params(model) -> float:
    """Số tham số (triệu) trên toàn bộ model."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3ximg_sizeximg_size, dùng hook tự tính."""
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(1, 3, img_size, img_size, device=device)
    total = 0.0

    def hook(module, inputs, outputs):
        nonlocal total
        if isinstance(module, nn.Conv2d):
            out_hw = outputs.shape[-1] * outputs.shape[-2]
            kernel = module.kernel_size[0] * module.kernel_size[1]
            in_ch = module.in_channels
            out_ch = module.out_channels
            total += 2.0 * out_hw * kernel * in_ch * out_ch / module.groups
        elif isinstance(module, nn.Linear):
            total += 2.0 * module.in_features * module.out_features

    handles = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for handle in handles:
            handle.remove()
    return total / 1e9
