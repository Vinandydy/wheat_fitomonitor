"""Статистика для доверительных интервалов (методичка курса).

B = 2000 — минимально допустимое учебное значение bootstrap-перевыборок.
"""
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

DEFAULT_B = 2000
Z_95 = 1.959963984540054  # z-критический для 95% двустороннего интервала


def wilson_ci(k: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """95%-й интервал Уилсона для доли k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def wilson_ci_bool(correct: Sequence[bool], z: float = Z_95) -> Tuple[float, float]:
    """Интервал Уилсона по вектору успехов True/False."""
    arr = np.asarray(correct, dtype=bool)
    return wilson_ci(int(arr.sum()), len(arr), z)


def bootstrap_ci(
    stat_fn: Callable[..., float],
    data: np.ndarray,
    n_boot: int = DEFAULT_B,
    alpha: float = 0.05,
    seed: int = 42,
    paired_with: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Percentile bootstrap CI.

    stat_fn(idx) считает статистику по индексам перевыборки.
    Если paired_with задан — парный bootstrap: stat_fn(idx, paired_with[idx]).
    Возвращает (точечная оценка, lo, hi).
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    point = stat_fn(np.arange(n))
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if paired_with is None:
            stats[b] = stat_fn(idx)
        else:
            stats[b] = stat_fn(idx, paired_with[idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def paired_bootstrap_mean_ci(
    clean_values: Sequence[float],
    other_values: Sequence[float],
    n_boot: int = DEFAULT_B,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Парный bootstrap для разности средних (clean - other).

    Возвращает (дельта, lo, hi). Если CI не содержит 0 — изменение стат. значимо.
    """
    a = np.asarray(clean_values, dtype=np.float64)
    b = np.asarray(other_values, dtype=np.float64)
    assert len(a) == len(b), "Парный bootstrap требует одинаковой длины выборок"
    rng = np.random.default_rng(seed)
    n = len(a)
    point = float((a - b).mean())
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = (a[idx] - b[idx]).mean()
    lo, hi = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)


def trapz_auc(x: Sequence[float], y: Sequence[float]) -> float:
    """Площадь под кривой методом трапеций (для R-AUC)."""
    return float(np.trapezoid(np.asarray(y, dtype=np.float64),
                              np.asarray(x, dtype=np.float64)))
