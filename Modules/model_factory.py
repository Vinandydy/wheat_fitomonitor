"""Фабрика baseline-модели EfficientNet-B5.

Создание архитектуры и transform'ов вынесено в отдельный модуль,
чтобы использовать одну и ту же модель в обучении и во всех ПР.
"""
from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import Compose


def create_effnetb5_model(
    num_classes: int, seed: int = 42
) -> Tuple[nn.Module, Compose]:
    """Создать EfficientNet-B5 с замороженным backbone и обучаемым head.

    Возвращает (model, transforms).
    """
    weights = models.EfficientNet_B5_Weights.DEFAULT
    transforms = weights.transforms()
    model = models.efficientnet_b5(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=2048, out_features=num_classes),
    )
    return model, transforms
