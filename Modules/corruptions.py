"""Естественные искажения (corruptions) для RGB-изображений.

Все функции принимают PIL.Image в режиме RGB, уровень искажения и seed,
возвращают искажённое PIL.Image RGB. Искажения применяются в пространстве
[0, 255] uint8, что соответствует реальным входным данным модели.
"""
from __future__ import annotations

from io import BytesIO
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


# ---------------------------------------------------------------------------
# Низкоуровневые функции искажений
# ---------------------------------------------------------------------------
def apply_gaussian_noise(img: Image.Image, severity: float, seed: int) -> Image.Image:
    """Аддитивный гауссов белый шум, severity = σ в диапазоне [0, 1]."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    noise = rng.normal(loc=0.0, scale=severity, size=arr.shape)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    noisy_uint8 = (noisy * 255.0).astype(np.uint8)
    return Image.fromarray(noisy_uint8, mode="RGB")


def apply_gaussian_blur(img: Image.Image, severity: float, seed: int) -> Image.Image:
    """Размытие по Гауссу, severity = радиус (σ) в пикселях."""
    _ = seed  # детерминированная операция, seed не требуется
    radius = float(severity)
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def apply_jpeg_compression(img: Image.Image, severity: int, seed: int) -> Image.Image:
    """JPEG-сжатие с заданным качеством; severity = Q (100 - почти без потерь)."""
    _ = seed  # кодирование JPEG детерминировано при фиксированном Q
    quality = int(severity)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def apply_brightness(img: Image.Image, severity: float, seed: int) -> Image.Image:
    """Изменение яркости, severity = factor (<1 - темнее, 1 - оригинал)."""
    _ = seed
    factor = float(severity)
    return ImageEnhance.Brightness(img).enhance(factor)


def apply_contrast(img: Image.Image, severity: float, seed: int) -> Image.Image:
    """Изменение контраста, severity = factor (<1 - меньше контраста, 1 - оригинал)."""
    _ = seed
    factor = float(severity)
    return ImageEnhance.Contrast(img).enhance(factor)


def apply_salt_and_pepper(img: Image.Image, severity: float, seed: int) -> Image.Image:
    """Шум "соль и перец", severity = доля зашумлённых пикселей [0, 1]."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.uint8).copy()
    h, w, _ = arr.shape
    n_pixels = h * w
    n_noisy = int(n_pixels * severity)

    coords = rng.choice(n_pixels, size=n_noisy, replace=False)
    vals = rng.integers(0, 2, size=n_noisy) * 255  # 0 (перец) или 255 (соль)

    rows = coords // w
    cols = coords % w
    arr[rows, cols, :] = vals[:, None]
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Реестр искажений
# ---------------------------------------------------------------------------
CORRUPTIONS: Dict[str, Callable[[Image.Image, float, int], Image.Image]] = {
    "gaussian_noise": apply_gaussian_noise,
    "gaussian_blur": apply_gaussian_blur,
    "jpeg_compression": apply_jpeg_compression,
    "brightness": apply_brightness,
    "contrast": apply_contrast,
    "salt_and_pepper": apply_salt_and_pepper,
}


# Конфигурация уровней искажения.
# levels — параметры, отображаемые в отчёте/на графиках.
# intensity — монотонно возрастающая физическая интенсивность искажения,
# используемая для корректного расчёта R-AUC методом трапеций.
CORRUPTION_CONFIG: Dict[str, Dict[str, object]] = {
    "gaussian_noise": {
        "levels": [0.01, 0.03, 0.05, 0.08],
        "intensity": [0.01, 0.03, 0.05, 0.08],
        "label": r"Gaussian noise $\sigma$",
        "unit": "",
    },
    "gaussian_blur": {
        "levels": [1.0, 2.0, 3.0, 5.0],
        "intensity": [1.0, 2.0, 3.0, 5.0],
        "label": "Gaussian blur radius",
        "unit": "px",
    },
    "jpeg_compression": {
        "levels": [85, 65, 45, 25],
        "intensity": [15, 35, 55, 75],  # 100 - quality
        "label": "JPEG quality",
        "unit": "",
    },
    "brightness": {
        "levels": [0.85, 0.70, 0.55, 0.40],
        "intensity": [0.15, 0.30, 0.45, 0.60],  # 1 - factor
        "label": "Brightness factor",
        "unit": "",
    },
    "contrast": {
        "levels": [0.85, 0.70, 0.55, 0.40],
        "intensity": [0.15, 0.30, 0.45, 0.60],  # 1 - factor
        "label": "Contrast factor",
        "unit": "",
    },
    "salt_and_pepper": {
        "levels": [0.01, 0.03, 0.05, 0.10],
        "intensity": [0.01, 0.03, 0.05, 0.10],
        "label": "Salt & pepper probability",
        "unit": "",
    },
}


def corrupt_image(img: Image.Image, corruption_type: str, severity: float, seed: int) -> Image.Image:
    """Применить искажение по имени."""
    if corruption_type not in CORRUPTIONS:
        raise ValueError(f"Неизвестное искажение: {corruption_type}. Доступны: {list(CORRUPTIONS)}")
    return CORRUPTIONS[corruption_type](img, severity, seed)


def corruption_level_name(corruption_type: str, severity: float) -> str:
    """Строковое представление уровня для именования файлов."""
    cfg = CORRUPTION_CONFIG[corruption_type]
    unit = cfg.get("unit", "")
    suffix = f"{unit}" if unit else ""
    if isinstance(severity, float):
        return f"{corruption_type}_s{severity:.4f}{suffix}"
    return f"{corruption_type}_s{severity}{suffix}"


def all_corruption_types() -> List[str]:
    return list(CORRUPTION_CONFIG.keys())
