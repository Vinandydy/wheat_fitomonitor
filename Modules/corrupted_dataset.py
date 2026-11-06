"""Dataset с естественными искажениями (corruptions) для ПР2.

Читает манифест, возвращает искажённое изображение после стандартного transform.
Seed каждого объекта зависит от базового seed и индекса, что гарантирует
воспроизводимость независимо от порядка загрузки батчей.
"""
import csv
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from Modules.corruptions import corrupt_image


class CorruptedWheatLeafDataset(Dataset):
    """Возвращает (image_tensor, label) с применённым corruption."""

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        split: str,
        corruption_type: str,
        severity: float,
        seed: int,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.corruption_type = corruption_type
        self.severity = severity
        self.seed = seed
        self.samples: List[Tuple[str, int]] = []

        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append((row["path"], int(row["class_idx"])))

        if not self.samples:
            raise ValueError(
                f"Манифест {manifest_path} не содержит записей split='{split}'"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        image = Image.open(self.data_root / rel_path).convert("RGB")
        image = corrupt_image(
            image,
            corruption_type=self.corruption_type,
            severity=self.severity,
            seed=self.seed + idx,
        )
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def labels(self):
        return [label for _, label in self.samples]

    @property
    def paths(self):
        return [path for path, _ in self.samples]
