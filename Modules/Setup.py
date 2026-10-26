"""Создание DataLoader'ов для датасета болезней пшеницы.

Датасет разбит на train/valid/test папками с разными именами классов,
поэтому вместо ImageFolder используется манифест (см. prepare_dataset.py).
"""
import csv
import os
from pathlib import Path
from typing import Callable, Dict, List

import torch
from torch.utils.data import DataLoader

from Modules.wheat_dataset import WheatLeafDataset

NUM_WORKERS = os.cpu_count()


# Чтение списка классов из манифеста (индекс -> имя)
def read_class_names(manifest_path: Path) -> List[str]:
    class_to_idx: dict = {}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            class_to_idx[row["class_name"]] = int(row["class_idx"])
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]


# Метод для создания даталоадеров по манифесту
def create_dataloaders(
        manifest_path: Path,
        data_root: Path,
        transform: Callable,
        batch_size: int,
        num_workers: int = NUM_WORKERS,
) -> Dict[str, DataLoader]:
    class_names = read_class_names(manifest_path)

    # Собираем даталоадеры; shuffle только для train
    loaders: Dict[str, DataLoader] = {}
    for split in ("train", "valid", "test"):
        dataset = WheatLeafDataset(
            manifest_path=manifest_path,
            data_root=data_root,
            split=split,
            transform=transform,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return loaders, class_names
