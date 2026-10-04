"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về loss cho cross-entropy, label smoothing, focal, weighted CE."""
    kind = kind.lower()
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing=float(kw.get("smoothing", 0.1)))
    if kind == "focal":
        return FocalLoss(gamma=float(kw.get("gamma", 2.0)), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight)
    raise ValueError(f"Unsupported loss kind: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits, target):
        if self.smoothing <= 1e-12:
            return F.cross_entropy(logits, target)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.scatter_(1, target.unsqueeze(1), 1.0)
            smooth = torch.full_like(log_probs, self.smoothing / logits.size(1))
            smooth = smooth * (1.0 - true_dist)
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth * log_probs).sum(dim=-1).mean()


class FocalLoss(nn.Module):
    """Focal loss đa lớp."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=1)
        pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = torch.exp(pt)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device).gather(0, target)
        else:
            alpha_t = torch.ones_like(pt)
        loss = -alpha_t * (1.0 - p_t) ** self.gamma * pt
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp, chuẩn hoá về trung bình 1 hoặc tổng số lớp."""
    arr = np.asarray(counts, dtype=np.float64)
    arr = arr.clip(min=1e-12)
    if beta <= 0:
        w = 1.0 / arr
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - beta ** arr)
        w = w / w.sum() * len(w)
    return torch.as_tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn theo mixup hoặc cutmix."""
    if alpha <= 0:
        raise ValueError("alpha phải > 0")
    if x.dim() != 4:
        raise ValueError("x phải có dạng (N, C, H, W)")
    y = y.to(x.device)
    perm = torch.randperm(x.size(0), device=x.device)
    lam = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
    y_a = y
    y_b = y[perm]

    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
        return x_mix, (y_a, y_b, lam)

    if mode == "cutmix":
        h, w = x.size(-2), x.size(-1)
        cut_ratio = torch.sqrt(1.0 - lam)
        cut_h = int(round(h * cut_ratio.item()))
        cut_w = int(round(w * cut_ratio.item()))
        cut_h = max(1, min(h, cut_h))
        cut_w = max(1, min(w, cut_w))
        cy = torch.randint(0, h - cut_h + 1, (1,), device=x.device).item()
        cx = torch.randint(0, w - cut_w + 1, (1,), device=x.device).item()
        x_mix = x.clone()
        x_patch = x[perm].clone()
        x_mix[:, :, cy:cy + cut_h, cx:cx + cut_w] = x_patch[:, :, cy:cy + cut_h, cx:cx + cut_w]
        lam = 1.0 - (cut_h * cut_w) / (h * w)
        return x_mix, (y_a, y_b, torch.tensor(lam, device=x.device))

    raise ValueError(f"mode không hỗ trợ: {mode}")


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam*CE(y_a)+(1-lam)*CE(y_b)."""
    y_a, y_b, lam = targets
    lam = lam.to(logits.device)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
