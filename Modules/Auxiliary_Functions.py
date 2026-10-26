"""Вспомогательные функции: seed'ы, графики обучения, инференс."""
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


# Метод для ручного ввода seed'ов (воспроизводимость экспериментов)
def set_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    # Сид для всех операций pytorch
    torch.manual_seed(seed)
    # Сид для CUDA ядер
    torch.cuda.manual_seed_all(seed)


# Метод для создания графика потерь и точности
def Graph_loss(results: dict):
    train_loss = results["train_loss"]
    valid_loss = results["valid_loss"]
    train_acc = results["train_acc"]
    valid_acc = results["valid_acc"]
    valid_f1 = results.get("valid_macro_f1")

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(18, 5))

    # Размещаем потери
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, valid_loss, label="valid_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Размещаем точность
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_acc, label="train_accuracy")
    plt.plot(epochs, valid_acc, label="valid_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    # Размещаем Macro-F1
    plt.subplot(1, 3, 3)
    if valid_f1:
        plt.plot(epochs, valid_f1, label="valid_macro_f1", color="tab:green")
    plt.title("Macro-F1")
    plt.xlabel("Epochs")
    plt.legend()

    plt.tight_layout()


# Расчёт времени, которое ушло на тренировку
def print_train_time(start, end, device=None):
    total_time = end - start
    print(f"\nТренировка была проведена на {device} за: {total_time:.1f} секунд")
    return total_time


# Метод для инференса одного изображения (вероятности по классам)
@torch.inference_mode()
def predict_image(model: torch.nn.Module,
                  image: torch.Tensor,
                  device: torch.device) -> torch.Tensor:
    """image — тензор после transform, без batch-измерения.
    Возвращает вектор softmax-вероятностей."""
    model.to(device)
    model.eval()
    probs = torch.softmax(model(image.unsqueeze(0).to(device)), dim=1)
    return probs.squeeze(0).cpu()
