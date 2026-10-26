"""Dataset болезней пшеницы по CSV-манифесту.

Манифест собирается скриптом prepare_dataset.py в корне проекта.
Пути в манифесте относительные (POSIX); data_root — базовая папка датасета.
"""
import csv
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset


class WheatLeafDataset(Dataset):
    """Читает манифест и отдаёт пары (изображение, метка) для одного сплита."""

    def __init__(
        self,
        manifest_path: Path,
        data_root: Path,
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        with open(manifest_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.samples.append(
                        (row["path"], int(row["class_idx"]))
                    )

        if not self.samples:
            raise ValueError(
                f"Манифест {manifest_path} не содержит записей split='{split}'"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        image = Image.open(self.data_root / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @property
    def labels(self) -> List[int]:
        return [label for _, label in self.samples]

    @property
    def paths(self) -> List[str]:
        return [path for path, _ in self.samples]
