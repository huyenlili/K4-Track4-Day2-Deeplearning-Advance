"""benchmark.py - đo độ trễ suy luận đúng cách."""
from __future__ import annotations

import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` theo ms."""
    if sync is None:
        sync = lambda: None
    for _ in range(warmup):
        fn();
        sync()
    samples = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(arr.mean()),
        "n": int(iters),
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của model với batch 1 hoặc batch lớn."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    if dtype == "fp16":
        model.half()
        x = x.half()
    elif dtype == "amp":
        model.float()
    else:
        model.float()

    sync = torch.cuda.synchronize if device.type == "cuda" else None

    def fn():
        with torch.inference_mode():
            if dtype == "amp" and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(x)
            else:
                model(x)

    res = bench(fn, warmup=warmup, iters=iters, sync=sync)
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": res["p50"],
        "p95": res["p95"],
        "p99": res["p99"],
        "mean": res["mean"],
        "images_per_s": batch_size / (res["p50"] / 1000.0),
        "torch": torch.__version__,
    }


def tta_latency(model, k_views: int, **kw) -> dict:
    """Xấp xỉ độ trễ TTA = K x latency của một view."""
    base = latency_report(model, **kw)
    return {
        "k_views": int(k_views),
        "p50": base["p50"] * k_views,
        "p95": base["p95"] * k_views,
        "p99": base["p99"] * k_views,
        "mean": base["mean"] * k_views,
        "n": base["n"] if "n" in base else 1,
    }
