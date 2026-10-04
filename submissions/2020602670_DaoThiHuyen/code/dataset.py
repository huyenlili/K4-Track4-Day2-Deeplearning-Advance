"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Giao diện giữ nguyên cho notebook/train/eval.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv."""
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    for df in (train_df, val_df, test_df):
        if "Filename" not in df.columns or "Label" not in df.columns:
            raise ValueError(f"CSV không đúng schema: {df.columns.tolist()}")
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train."""
    images_dir = Path(images_dir)
    train_names = set(train_df["Filename"].astype(str).tolist())
    val_names = set(val_df["Filename"].astype(str).tolist())
    test_names = set(test_df["Filename"].astype(str).tolist())

    overlap = {
        "train_val": sorted(train_names & val_names)[:10],
        "train_test": sorted(train_names & test_names)[:10],
        "val_test": sorted(val_names & test_names)[:10],
    }
    if train_names & val_names or train_names & test_names or val_names & test_names:
        raise ValueError(f"Tập train/val/test có giao không rỗng: {overlap}")

    n_total = len(train_df) + len(val_df) + len(test_df)
    if n_total != 17509:
        raise ValueError(f"Tổng số ảnh không đúng 17,509: {n_total}")

    missing = []
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        for filename in split_df["Filename"].astype(str):
            if not (images_dir / filename).exists():
                missing.append((split_name, filename))
    if missing:
        raise FileNotFoundError(f"Một số file ảnh không tồn tại trong {images_dir}: {missing[:5]}")

    n = {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))}
    per_class = {}
    for split_name, split_df in (("train", train_df), ("val", val_df), ("test", test_df)):
        counts = split_df["Label"].value_counts().sort_index().to_dict()
        per_class[split_name] = {int(k): int(v) for k, v in counts.items()}

    all_counts = list(per_class["train"].values()) + list(per_class["val"].values()) + list(per_class["test"].values())
    if any(count <= 0 for count in all_counts):
        raise ValueError(f"Có lớp bị thiếu trong một tập: {per_class}")

    report = {"n": n, "per_class": per_class, "overlap": overlap}
    print("Split stats:")
    print(report)
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. Chọn "basic" như training cơ bản; còn lại ổn định."""
    base = [transforms.Resize((img_size, img_size))]
    if train:
        if aug == "basic":
            ops = [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
            ]
        elif aug == "color":
            ops = [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            ]
        elif aug == "trivial":
            ops = [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
            ]
        elif aug == "randaug":
            ops = [
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandAugment(),
            ]
        else:
            ops = [transforms.RandomResizedCrop(img_size), transforms.RandomHorizontalFlip()]
        return transforms.Compose(ops + [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.samples = []
        for _, row in self.df.iterrows():
            filename = str(row["Filename"])
            label = int(row["Label"])
            self.samples.append((filename, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        filename, label = self.samples[i]
        image_path = self.images_dir / filename
        with Image.open(image_path).convert("RGB") as image:
            if self.transform is not None:
                image = self.transform(image)
        return image, int(label), filename


def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df, images_dir, transform=transform)
    if sampler == "balanced":
        counts = np.bincount(df["Label"].astype(int).to_numpy(), minlength=NUM_CLASSES)
        weights = 1.0 / np.clip(counts, 1, None)
        sample_weights = weights[df["Label"].astype(int).to_numpy()]
        sampler_obj = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
        shuffle = False
    else:
        sampler_obj = None
        shuffle = train

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler_obj,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_worker_init_fn,
    )
    return loader
