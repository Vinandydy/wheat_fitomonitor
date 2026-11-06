"""Визуализация результатов ПР2: Robustness Curves и примеры искажений."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Визуализация ПР2")
    parser.add_argument("--metrics", type=Path, default=BASE / "results" / "pr2_metrics.json")
    parser.add_argument("--example-dir", type=Path, default=BASE / "results" / "example_images")
    parser.add_argument("--output-dir", type=Path, default=BASE / "results" / "plots")
    return parser.parse_args()


def plot_robustness_curves(metrics: dict, out_path: Path):
    """Robustness Curves для всех искажений на одном графике."""
    baseline_acc = metrics["baseline"]["accuracy"]
    corruption_results = metrics["corruption_results"]
    n_types = len(corruption_results)

    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(14, 16), sharey=True
    )
    axes = axes.flatten()

    for ax, (ctype, res) in zip(axes, corruption_results.items()):
        levels = res["levels"]
        accs = res["mean_accuracy"]
        ci_lo = [c[0] for c in res["accuracy_ci95_bootstrap"]]
        ci_hi = [c[1] for c in res["accuracy_ci95_bootstrap"]]

        ax.plot(levels, accs, marker="o", label=res["label"])
        ax.fill_between(levels, ci_lo, ci_hi, alpha=0.25)
        ax.axhline(baseline_acc, color="gray", linestyle="--", linewidth=1, label="Baseline")
        ax.set_xlabel(res["label"])
        ax.set_ylabel("Accuracy")
        ax.set_title(ctype.replace("_", " ").title())
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Robustness Curves (среднее по 7 реализациям шума, 95% CI)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Инфо] Robustness curves сохранены: {out_path}")


def plot_relative_retention(metrics: dict, out_path: Path):
    """График относительного сохранения качества."""
    corruption_results = metrics["corruption_results"]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 16), sharey=True)
    axes = axes.flatten()

    for ax, (ctype, res) in zip(axes, corruption_results.items()):
        levels = res["levels"]
        rel = res["relative_retention"]
        ax.plot(levels, rel, marker="s", color="tab:orange")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(res["label"])
        ax.set_ylabel("Relative retention M(s) / M(0)")
        ax.set_title(ctype.replace("_", " ").title())
        ax.set_ylim(0.0, 1.1)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Относительное сохранение качества", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Инфо] Relative retention curves сохранены: {out_path}")


def plot_example_grids(example_dir: Path, out_path: Path, n_samples: int = 4):
    """Сетка примеров искажённых изображений по всем типам."""
    if not example_dir.exists():
        print(f"[Предупреждение] Папка с примерами не найдена: {example_dir}")
        return

    corruption_types = sorted([d.name for d in example_dir.iterdir() if d.is_dir()])
    n_types = len(corruption_types)
    # 1 original + 4 levels
    n_cols = 5

    fig, axes = plt.subplots(nrows=n_types, ncols=n_cols, figsize=(16, 3.2 * n_types))
    if n_types == 1:
        axes = np.array([axes])

    for row_idx, ctype in enumerate(corruption_types):
        ctype_dir = example_dir / ctype
        # Используем sample_0
        sample_idx = 0
        originals = list(ctype_dir.glob(f"sample_{sample_idx}_original.png"))
        if not originals:
            continue
        orig_img = Image.open(originals[0])
        axes[row_idx, 0].imshow(orig_img)
        axes[row_idx, 0].set_title("Original")
        axes[row_idx, 0].axis("off")

        level_files = sorted(
            ctype_dir.glob(f"sample_{sample_idx}_level*.png"),
            key=lambda p: int(p.stem.split("level")[1].split("_")[0]),
        )
        for col_idx, lvl_file in enumerate(level_files[: n_cols - 1], start=1):
            img = Image.open(lvl_file)
            axes[row_idx, col_idx].imshow(img)
            # Имя уровня из имени файла
            parts = lvl_file.stem.split("_")
            lvl_value = parts[-1]
            axes[row_idx, col_idx].set_title(f"Level {col_idx}: {lvl_value}")
            axes[row_idx, col_idx].axis("off")

        # Убираем пустые оси, если уровней меньше
        for col_idx in range(len(level_files) + 1, n_cols):
            axes[row_idx, col_idx].axis("off")

    fig.suptitle("Примеры естественных искажений (sample 0)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Инфо] Grid примеров сохранён: {out_path}")


def plot_auc_bar(metrics: dict, out_path: Path):
    """Столбчатая диаграмма R-AUC с доверительными интервалами."""
    corruption_results = metrics["corruption_results"]
    names = [k.replace("_", " ").title() for k in corruption_results.keys()]
    aucs = [res["r_auc"] for res in corruption_results.values()]
    ci_lo = [res["r_auc_ci95_bootstrap"][0] for res in corruption_results.values()]
    ci_hi = [res["r_auc_ci95_bootstrap"][1] for res in corruption_results.values()]
    errs = [[a - l for a, l in zip(aucs, ci_lo)], [h - a for a, h in zip(aucs, ci_hi)]]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(names))
    ax.bar(x, aucs, yerr=errs, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("R-AUC")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("R-AUC по типам искажений (с 95% bootstrap CI)")
    ax.axhline(metrics["baseline"]["accuracy"], color="gray", linestyle="--", label="Baseline accuracy")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Инфо] R-AUC bar chart сохранён: {out_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metrics, encoding="utf-8") as f:
        metrics = json.load(f)

    plot_robustness_curves(metrics, args.output_dir / "robustness_curves.png")
    plot_relative_retention(metrics, args.output_dir / "relative_retention.png")
    plot_auc_bar(metrics, args.output_dir / "r_auc_bar.png")
    plot_example_grids(args.example_dir, args.output_dir / "example_grid.png")


if __name__ == "__main__":
    main()
