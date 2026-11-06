"""Сбор предсказаний модели по DataLoader.

Вынесено в отдельный модуль, чтобы использовать и при обучении baseline,
и при оценке robustness / adversarial / privacy.
"""
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Возвращает logits, probs, labels, losses, paths.

    Предполагается, что у loader.dataset есть атрибут paths.
    """
    model.eval()
    use_amp = amp and device.type == "cuda"
    all_logits, all_labels, all_paths = [], [], []
    offset = 0
    with torch.inference_mode():
        for X, y in loader:
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X.to(device, non_blocking=True)).float()
            all_logits.append(logits.cpu())
            all_labels.append(y)
            bs = len(y)
            all_paths.extend(loader.dataset.paths[offset:offset + bs])
            offset += bs
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    ce = nn.CrossEntropyLoss(reduction="none")
    losses = ce(torch.from_numpy(logits), torch.from_numpy(labels)).numpy()
    return logits, probs, labels, losses, all_paths
