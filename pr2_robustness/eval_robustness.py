"""Оценка corruption robustness baseline-модели (ПР2).

Загружает обученную EfficientNet-B5, применяет 6 типов естественных искажений
к тестовой выборке, усредняет результаты по 7 независимым реализациям шума,
строит доверительные интервалы и считает R-AUC.

Запуск:
    uv run pr2_robustness/eval_robustness.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from Modules.Auxiliary_Functions import set_seeds
from Modules.corruptions import CORRUPTION_CONFIG, all_corruption_types, corrupt_image
from Modules.corrupted_dataset import CorruptedWheatLeafDataset
from Modules.model_factory import create_effnetb5_model
from Modules.predictions import collect_predictions
from Modules.stats import bootstrap_ci, paired_bootstrap_mean_ci, trapz_auc, wilson_ci
from Modules.wheat_dataset import WheatLeafDataset


def accuracy_from_preds(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def macro_f1_from_preds(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(f1_score(labels, preds, average="macro"))


def save_example_images(
    corruption_type: str,
    levels: List[float],
    base_seed: int,
    data_root: Path,
    manifest_path: Path,
    example_dir: Path,
    n_samples: int = 4,
):
    """Сохраняет примеры искажённых изображений для отчёта."""
    # Читаем первые n_samples из тестового сплита
    samples: List[tuple] = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                samples.append((row["path"], int(row["class_idx"])))
                if len(samples) >= n_samples:
                    break

    out = example_dir / corruption_type
    out.mkdir(parents=True, exist_ok=True)

    for idx, (rel_path, label) in enumerate(samples):
        img = Image.open(data_root / rel_path).convert("RGB")
        # Оригинал
        img.save(out / f"sample_{idx}_original.png")
        for lvl_idx, lvl in enumerate(levels):
            corrupted = corrupt_image(img, corruption_type, lvl, base_seed + idx)
            corrupted.save(out / f"sample_{idx}_level{lvl_idx}_{lvl}.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Оценка corruption robustness (ПР2)")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\andy7\stud\archive\data"))
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "artifacts" / "dataset_manifest.csv")
    parser.add_argument("--model-weights", type=Path, default=PROJECT_ROOT / "artifacts" / "models" / "baseline_effnetb5.pth")
    parser.add_argument("--train-config", type=Path, default=PROJECT_ROOT / "artifacts" / "train_config.json")
    parser.add_argument("--output-dir", type=Path, default=BASE / "results")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--n-realizations", type=int, default=7)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--n-boot", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seeds(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Инфо] Устройство: {device}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    example_dir = out_dir / "example_images"
    example_dir.mkdir(parents=True, exist_ok=True)

    # Загружаем конфиг и модель
    with open(args.train_config, encoding="utf-8") as f:
        train_config = json.load(f)
    class_names = train_config["class_names"]
    num_classes = len(class_names)

    model, transform = create_effnetb5_model(num_classes=num_classes, seed=42)
    model.load_state_dict(
        torch.load(args.model_weights, map_location=device, weights_only=True)
    )
    model = model.to(device)

    # ---------- Baseline на чистой тестовой выборке ----------
    baseline_dataset = WheatLeafDataset(
        manifest_path=args.manifest,
        data_root=args.data_root,
        split="test",
        transform=transform,
    )
    baseline_loader = DataLoader(
        baseline_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    base_logits, _, base_labels, _, _ = collect_predictions(
        model, baseline_loader, device, amp=True
    )
    base_preds = base_logits.argmax(axis=1)
    n_test = len(base_labels)

    baseline_acc = accuracy_from_preds(base_preds, base_labels)
    n_correct_base = int((base_preds == base_labels).sum())
    base_acc_lo, base_acc_hi = wilson_ci(n_correct_base, n_test)
    baseline_macro_f1 = macro_f1_from_preds(base_preds, base_labels)
    _, f1_lo, f1_hi = bootstrap_ci(
        lambda idx: macro_f1_from_preds(base_preds[idx], base_labels[idx]),
        np.arange(n_test),
        n_boot=args.n_boot,
        seed=42,
    )

    baseline_result = {
        "accuracy": baseline_acc,
        "accuracy_ci95_wilson": [base_acc_lo, base_acc_hi],
        "macro_f1": baseline_macro_f1,
        "macro_f1_ci95_bootstrap": [f1_lo, f1_hi],
        "n_test": n_test,
    }
    base_correct = (base_preds == base_labels).astype(np.float64)
    print(f"[Baseline] Accuracy: {baseline_acc:.4f} [{base_acc_lo:.4f}, {base_acc_hi:.4f}]")
    print(f"[Baseline] Macro-F1: {baseline_macro_f1:.4f} [{f1_lo:.4f}, {f1_hi:.4f}]")

    # ---------- Оценка по искажениям ----------
    seeds = [args.base_seed + i for i in range(args.n_realizations)]
    corruption_results: Dict[str, dict] = {}

    for ctype in all_corruption_types():
        cfg = CORRUPTION_CONFIG[ctype]
        levels = cfg["levels"]
        intensity = cfg["intensity"]
        print(f"\n[Инфо] Искажение: {ctype}, уровни: {levels}, интенсивность: {intensity}, реализаций: {args.n_realizations}")

        # Сохраняем примеры изображений (только для первого seed)
        save_example_images(
            ctype, levels, seeds[0], args.data_root, args.manifest, example_dir
        )

        per_level_corrects: List[np.ndarray] = []  # (n_seeds, n_test)
        per_level_preds: List[np.ndarray] = []     # (n_seeds, n_test)
        per_level_accs: List[float] = []
        per_level_f1s: List[float] = []

        t0 = time.time()
        for lvl_idx, lvl in enumerate(levels):
            level_corrects = []
            level_preds = []
            for seed in seeds:
                ds = CorruptedWheatLeafDataset(
                    manifest_path=args.manifest,
                    data_root=args.data_root,
                    split="test",
                    corruption_type=ctype,
                    severity=lvl,
                    seed=seed,
                    transform=transform,
                )
                loader = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                logits, _, labels, _, _ = collect_predictions(
                    model, loader, device, amp=True
                )
                preds = logits.argmax(axis=1)
                level_corrects.append((preds == labels).astype(np.float64))
                level_preds.append(preds)

            stacked_corrects = np.stack(level_corrects, axis=0)
            stacked_preds = np.stack(level_preds, axis=0)
            per_level_corrects.append(stacked_corrects)
            per_level_preds.append(stacked_preds)
            per_level_accs.append(float(stacked_corrects.mean()))
            # Macro-F1 усредняем по реализациям
            per_level_f1s.append(
                float(np.mean([
                    macro_f1_from_preds(p, base_labels) for p in level_preds
                ]))
            )
            print(f"  level {lvl}: accuracy={per_level_accs[-1]:.4f}, macro_f1={per_level_f1s[-1]:.4f}")

        # Усреднённый correct vector по реализациям для каждого уровня
        mean_corrects = [c.mean(axis=0) for c in per_level_corrects]
        mean_accs = per_level_accs
        mean_f1s = per_level_f1s

        # Bootstrap CI для accuracy (по объектам, усреднение по seed внутри)
        acc_cis = []
        rng = np.random.default_rng(42)
        for cvec in mean_corrects:
            boots = []
            for _ in range(args.n_boot):
                idx = rng.integers(0, n_test, size=n_test)
                boots.append(float(cvec[idx].mean()))
            boots = np.asarray(boots)
            lo, hi = np.quantile(boots, [0.025, 0.975])
            acc_cis.append([float(lo), float(hi)])

        # Парный bootstrap для ΔAccuracy
        deltas = []
        delta_cis = []
        for cvec in mean_corrects:
            delta, lo, hi = paired_bootstrap_mean_ci(
                base_correct, cvec, n_boot=args.n_boot, seed=42
            )
            deltas.append(float(delta))
            delta_cis.append([float(lo), float(hi)])

        # Относительное сохранение качества
        rel_retention = [float(m / baseline_acc) for m in mean_accs]

        # R-AUC (метод трапеций, нормированный)
        # Интенсивность монотонно возрастает с ростом искажения
        x = np.asarray([0.0] + [float(v) for v in intensity], dtype=np.float64)
        y = np.asarray([baseline_acc] + mean_accs, dtype=np.float64)
        r_auc_point = trapz_auc(x, y) / (x.max() - x.min())

        # Bootstrap R-AUC по объектам
        rng = np.random.default_rng(42)
        r_auc_boots = []
        for _ in range(args.n_boot):
            idx = rng.integers(0, n_test, size=n_test)
            y_boot = [baseline_acc]
            for cmat in per_level_corrects:
                y_boot.append(float(cmat[:, idx].mean()))
            y_boot = np.asarray(y_boot, dtype=np.float64)
            r_auc_boots.append(trapz_auc(x, y_boot) / (x.max() - x.min()))
        r_auc_boots = np.asarray(r_auc_boots)
        r_auc_lo, r_auc_hi = np.quantile(r_auc_boots, [0.025, 0.975])

        corruption_results[ctype] = {
            "label": cfg["label"],
            "levels": levels,
            "intensity": intensity,
            "mean_accuracy": mean_accs,
            "accuracy_ci95_bootstrap": acc_cis,
            "mean_macro_f1": mean_f1s,
            "delta_accuracy": deltas,
            "delta_accuracy_ci95_paired_bootstrap": delta_cis,
            "relative_retention": rel_retention,
            "r_auc": float(r_auc_point),
            "r_auc_ci95_bootstrap": [float(r_auc_lo), float(r_auc_hi)],
            "seeds": seeds,
        }
        print(
            f"  R-AUC={r_auc_point:.4f} [{r_auc_lo:.4f}, {r_auc_hi:.4f}]  "
            f"({time.time() - t0:.1f}s)"
        )

    # ---------- Сохранение артефактов ----------
    final = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": baseline_result,
        "corruption_results": corruption_results,
        "config": {
            "n_realizations": args.n_realizations,
            "base_seed": args.base_seed,
            "seeds": seeds,
            "n_boot": args.n_boot,
            "corruption_types": all_corruption_types(),
            "batch_size": args.batch_size,
            "device": str(device),
        },
    }
    metrics_path = out_dir / "pr2_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

    # CSV для кривых
    rows = []
    for ctype, res in corruption_results.items():
        for i, lvl in enumerate(res["levels"]):
            rows.append({
                "corruption_type": ctype,
                "level": lvl,
                "intensity": res["intensity"][i],
                "level_index": i,
                "mean_accuracy": res["mean_accuracy"][i],
                "accuracy_ci95_lo": res["accuracy_ci95_bootstrap"][i][0],
                "accuracy_ci95_hi": res["accuracy_ci95_bootstrap"][i][1],
                "mean_macro_f1": res["mean_macro_f1"][i],
                "delta_accuracy": res["delta_accuracy"][i],
                "delta_ci95_lo": res["delta_accuracy_ci95_paired_bootstrap"][i][0],
                "delta_ci95_hi": res["delta_accuracy_ci95_paired_bootstrap"][i][1],
                "relative_retention": res["relative_retention"][i],
                "r_auc": res["r_auc"],
                "r_auc_ci95_lo": res["r_auc_ci95_bootstrap"][0],
                "r_auc_ci95_hi": res["r_auc_ci95_bootstrap"][1],
            })
    csv_path = out_dir / "pr2_curve_data.csv"
    fieldnames = [
        "corruption_type", "level", "intensity", "level_index", "mean_accuracy",
        "accuracy_ci95_lo", "accuracy_ci95_hi", "mean_macro_f1",
        "delta_accuracy", "delta_ci95_lo", "delta_ci95_hi",
        "relative_retention", "r_auc", "r_auc_ci95_lo", "r_auc_ci95_hi",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Инфо] Метрики сохранены: {metrics_path}")
    print(f"[Инфо] CSV сохранён:       {csv_path}")
    print(f"[Инфо] Примеры изображений: {example_dir}")


if __name__ == "__main__":
    main()
