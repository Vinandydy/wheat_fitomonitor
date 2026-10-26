"""ПР1: Исследование качества набора данных.

Запуск по частям (рекомендуется):
    uv run python pr1_data_quality/pr1_data_quality.py --stage profile
    uv run python pr1_data_quality/pr1_data_quality.py --stage duplicates
    uv run python pr1_data_quality/pr1_data_quality.py --stage leakage
    uv run python pr1_data_quality/pr1_data_quality.py --stage labels      # долго (~30-50 мин)
    uv run python pr1_data_quality/pr1_data_quality.py --stage shift
    uv run python pr1_data_quality/pr1_data_quality.py --stage anomalies
    uv run python pr1_data_quality/pr1_data_quality.py --stage report

Запуск всего сразу:
    uv run python pr1_data_quality/pr1_data_quality.py --stage all
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from cleanlab.filter import find_label_issues

from Modules.Setup import read_class_names
from Modules.stats import wilson_ci
from pr1_data_quality.pr1_utils import (
    B5FeatureExtractor,
    anomaly_summary,
    collect_anomaly_features,
    compute_oof_probabilities,
    c2st_score,
    extract_embeddings,
    find_exact_duplicates,
    find_near_duplicates,
    find_train_test_leakage,
    get_env_info,
    load_json,
    load_manifest,
    permutation_test_mmd,
    save_json,
)

BASE = Path(__file__).resolve().parent
PROJECT_ROOT = BASE.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "artifacts" / "dataset_manifest.csv"
DEFAULT_DATA_ROOT = Path(r"C:\Users\andy7\stud\archive\data")
DEFAULT_OUTPUT = BASE / "results"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "models" / "baseline_effnetb5.pth"


def parse_args():
    parser = argparse.ArgumentParser(description="ПР1: качество данных")
    parser.add_argument("--stage", type=str, default="profile",
                        choices=["profile", "duplicates", "leakage", "labels",
                                 "shift", "anomalies", "report", "all"],
                        help="Какую часть запустить")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--phash-threshold", type=int, default=10)
    parser.add_argument("--cleanlab-folds", type=int, default=5)
    parser.add_argument("--cleanlab-epochs", type=int, default=10)
    parser.add_argument("--cleanlab-patience", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-brisque", action="store_true",
                        help="Не считать BRISQUE (ускоряет anomaly detection в ~10-20 раз)")
    return parser.parse_args()


def make_config(args) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "phash_threshold": args.phash_threshold,
        "cleanlab_folds": args.cleanlab_folds,
        "cleanlab_epochs": args.cleanlab_epochs,
        "cleanlab_patience": args.cleanlab_patience,
        "amp": not args.no_amp,
        "use_brisque": not args.no_brisque,
        "environment": get_env_info(),
    }


def run_profile(args):
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = PROJECT_ROOT / "artifacts" / "dataset_profile.json"
    if not profile_path.exists():
        raise FileNotFoundError("Сначала запустите prepare_dataset.py")
    profile = load_json(profile_path)
    profile["pr1_config"] = make_config(args)
    save_json(out_dir / "pr1_profile.json", profile)
    print(f"[ПР1] Профиль сохранён: {out_dir / 'pr1_profile.json'}")


def run_duplicates(args):
    rows = load_manifest(args.manifest)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exact = find_exact_duplicates(rows, args.data_root)
    near = find_near_duplicates(rows, args.data_root, threshold=args.phash_threshold)

    result = {
        "config": make_config(args),
        "exact_duplicates": exact,
        "near_duplicates": near,
    }
    save_json(out_dir / "pr1_duplicates.json", result)
    print(f"[ПР1] Exact duplicates: {exact['duplicate_images']} / {exact['total_images']} "
          f"({exact['duplicate_rate']:.4f})")
    print(f"[ПР1] Near duplicates:  {near['near_duplicate_images']} / {near['total_images']} "
          f"({near['near_duplicate_rate']:.4f})")
    print(f"[ПР1] Результат: {out_dir / 'pr1_duplicates.json'}")


def run_leakage(args):
    rows = load_manifest(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    leakage = find_train_test_leakage(train_rows, test_rows, args.data_root,
                                      threshold=args.phash_threshold)
    result = {"config": make_config(args), "train_test_leakage": leakage}
    save_json(out_dir / "pr1_leakage.json", result)
    print(f"[ПР1] Train-test leakage: {leakage['leaked_test_images']} / {leakage['test_images']} "
          f"({leakage['leakage_rate']:.4f})")
    print(f"[ПР1] Результат: {out_dir / 'pr1_leakage.json'}")


def run_labels(args):
    rows = load_manifest(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    num_classes = len(read_class_names(args.manifest))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ПР1] Запуск 5-fold CV для cleanlab (классов: {num_classes})...")
    t0 = time.time()
    oof = compute_oof_probabilities(
        manifest_path=args.manifest,
        data_root=args.data_root,
        num_classes=num_classes,
        device=device,
        n_splits=args.cleanlab_folds,
        epochs=args.cleanlab_epochs,
        batch_size=args.batch_size,
        patience=args.cleanlab_patience,
        amp=not args.no_amp,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"[ПР1] 5-fold CV завершён за {time.time() - t0:.1f}s")

    issues = find_label_issues(oof["labels"], oof["oof_probs"], return_indices_ranked_by="self_confidence")
    n_issues = len(issues)
    n_total = len(train_rows)
    error_rate = n_issues / n_total

    suspicious = [{
        "path": train_rows[i]["path"],
        "class_name": train_rows[i]["class_name"],
        "class_idx": int(oof["labels"][i]),
        "prob_true_class": round(float(oof["oof_probs"][i, oof["labels"][i]]), 4),
    } for i in issues[:100]]

    np.savez_compressed(out_dir / "pr1_train_oof_probs.npz",
                        probs=oof["oof_probs"], labels=oof["labels"])

    result = {
        "config": make_config(args),
        "method": "cleanlab Confident Learning (5-fold CV out-of-sample probs)",
        "n_train": n_total,
        "n_issues": n_issues,
        "label_error_rate": error_rate,
        "label_error_rate_ci95": wilson_ci(n_issues, n_total),
        "suspicious_top100": suspicious,
    }
    save_json(out_dir / "pr1_labels.json", result)
    pd.DataFrame(suspicious).to_csv(out_dir / "pr1_suspicious_labels.csv", index=False)
    print(f"[ПР1] Подозрительных меток: {n_issues} / {n_total} ({error_rate:.4f})")
    print(f"[ПР1] Результат: {out_dir / 'pr1_labels.json'}")


def run_shift(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = args.manifest
    data_root = args.data_root
    num_workers = args.num_workers
    batch_size = args.batch_size
    amp = not args.no_amp
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = torchvision.models.EfficientNet_B5_Weights.DEFAULT.transforms()

    extractor = B5FeatureExtractor().to(device)
    extractor.eval()

    from Modules.wheat_dataset import WheatLeafDataset
    train_loader = torch.utils.data.DataLoader(
        WheatLeafDataset(manifest_path, data_root, "train", transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda")
    test_loader = torch.utils.data.DataLoader(
        WheatLeafDataset(manifest_path, data_root, "test", transform),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda")

    print("[ПР1] Извлекаю embeddings train/test...")
    x_train = extract_embeddings(extractor, train_loader, device, amp=amp)
    x_test = extract_embeddings(extractor, test_loader, device, amp=amp)
    print(f"[ПР1] Train embeddings: {x_train.shape}, Test embeddings: {x_test.shape}")

    print("[ПР1] Считаю MMD + permutation test...")
    mmd_value, p_value = permutation_test_mmd(x_train, x_test, n_permutations=100,
                                              max_n=400, seed=args.seed)

    print("[ПР1] Считаю C2ST...")
    c2st_acc, c2st_lo, c2st_hi = c2st_score(x_train, x_test, max_n=400,
                                            n_bootstraps=200, seed=args.seed)

    result = {
        "config": make_config(args),
        "feature_extractor": "EfficientNet-B5 (features before classifier, 2048-dim)",
        "mmd": {"value": float(mmd_value), "p_value": float(p_value)},
        "c2st": {"accuracy": float(c2st_acc), "ci95": [float(c2st_lo), float(c2st_hi)]},
    }
    save_json(out_dir / "pr1_shift.json", result)
    print(f"[ПР1] MMD = {mmd_value:.6f}, p-value = {p_value:.4f}")
    print(f"[ПР1] C2ST accuracy = {c2st_acc:.4f} [{c2st_lo:.4f}, {c2st_hi:.4f}]")
    print(f"[ПР1] Результат: {out_dir / 'pr1_shift.json'}")


def run_anomalies(args):
    rows = load_manifest(args.manifest)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    use_brisque = not args.no_brisque
    if not use_brisque:
        print("[ПР1] BRISQUE отключён (--no-brisque); считаются только быстрые признаки.")

    records = collect_anomaly_features(rows, args.data_root, use_brisque=use_brisque)
    summary = anomaly_summary(records)

    # Топ аномальных по blur (низкий = размыто)
    df = pd.DataFrame([r for r in records if "error" not in r])
    df["blur_zscore"] = (df["laplacian_blur"] - df["laplacian_blur"].mean()) / df["laplacian_blur"].std()
    cols = ["path", "split", "class_name", "width", "height", "aspect_ratio",
            "laplacian_blur", "brightness", "contrast"]
    if use_brisque:
        cols.append("brisque")
    top_blur = df.nsmallest(50, "laplacian_blur")[cols]

    result = {
        "config": make_config(args),
        "summary": summary,
    }
    save_json(out_dir / "pr1_anomalies.json", result)
    top_blur.to_csv(out_dir / "pr1_top_blur.csv", index=False)
    print(f"[ПР1] Аномалий: broken={summary['broken']}, extreme_aspect={summary['extreme_aspect_count']}, "
          f"low_blur={summary['low_blur_count']}")
    print(f"[ПР1] Результат: {out_dir / 'pr1_anomalies.json'}")


def run_report(args):
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = BASE / "report_pr1.md"

    def load(stage_file):
        p = out_dir / stage_file
        return load_json(p) if p.exists() else None

    profile = load("pr1_profile.json")
    duplicates = load("pr1_duplicates.json")
    leakage = load("pr1_leakage.json")
    labels = load("pr1_labels.json")
    shift = load("pr1_shift.json")
    anomalies = load("pr1_anomalies.json")

    lines = ["# ПР1. Исследование качества набора данных\n"]
    lines.append(f"**Датасет:** {profile['n_classes']} классов, "
                 f"всего {sum(profile['totals'].values())} изображений\n")
    lines.append(f"**Split:** train={profile['totals']['train']}, "
                 f"valid={profile['totals']['valid']}, test={profile['totals']['test']}\n\n")

    lines.append("## Сводная таблица качества данных\n")
    lines.append("| Свойство | Метрика | Значение | 95% CI | Интерпретация |\n")
    lines.append("|---|---|---|---|---|\n")

    if duplicates:
        d = duplicates["exact_duplicates"]
        lines.append(f"| Уникальность | exact duplicate rate | {d['duplicate_rate']:.4f} | "
                     f"[{d['duplicate_rate_ci95'][0]:.4f}, {d['duplicate_rate_ci95'][1]:.4f}] | "
                     f"{d['duplicate_images']} дубликатов |\n")
        n = duplicates["near_duplicates"]
        lines.append(f"| Уникальность | near duplicate rate | {n['near_duplicate_rate']:.4f} | "
                     f"[{n['near_duplicate_rate_ci95'][0]:.4f}, {n['near_duplicate_rate_ci95'][1]:.4f}] | "
                     f"{n['near_duplicate_images']} вовлечено |\n")

    if leakage:
        l = leakage["train_test_leakage"]
        lines.append(f"| Train-test leakage | leakage rate | {l['leakage_rate']:.4f} | "
                     f"[{l['leakage_rate_ci95'][0]:.4f}, {l['leakage_rate_ci95'][1]:.4f}] | "
                     f"{l['leaked_test_images']} тестовых образцов |\n")

    if labels:
        lines.append(f"| Качество меток | label error rate | {labels['label_error_rate']:.4f} | "
                     f"[{labels['label_error_rate_ci95'][0]:.4f}, {labels['label_error_rate_ci95'][1]:.4f}] | "
                     f"cleanlab, {labels['n_issues']} подозрительных |\n")

    if profile:
        lines.append(f"| Баланс | imbalance ratio | {profile['train_imbalance_ratio']:.2f} | — | "
                     f"{profile['train_class_entropy_norm']:.3f} норм. энтропии |\n")

    if shift:
        mmd = shift["mmd"]
        c2st = shift["c2st"]
        lines.append(f"| Репрезентативность | MMD | {mmd['value']:.6f} | p={mmd['p_value']:.4f} | "
                     f"{'дрейф обнаружен' if mmd['p_value'] < 0.05 else 'дрейф не значим'} |\n")
        lines.append(f"| Репрезентативность | C2ST accuracy | {c2st['accuracy']:.4f} | "
                     f"[{c2st['ci95'][0]:.4f}, {c2st['ci95'][1]:.4f}] | "
                     f"{'дрейф' if c2st['accuracy'] > 0.55 else 'нет дрейфа'} |\n")

    if anomalies:
        s = anomalies["summary"]
        lines.append(f"| Аномалии | broken rate | {s['broken_rate']:.4f} | "
                     f"[{s['broken_rate_ci95'][0]:.4f}, {s['broken_rate_ci95'][1]:.4f}] | "
                     f"{s['broken']} битых файлов |\n")

    lines.append("\n## Ограничения\n")
    lines.append("- Результаты относятся к текущему фиксированному сплиту и seed=42.\n")
    lines.append("- BRISQUE/NIQE: используется piq.brisque; NIQE не доступен в piq 0.8.0.\n")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ПР1] Отчёт сохранён: {report_path}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    stage_map = {
        "profile": run_profile,
        "duplicates": run_duplicates,
        "leakage": run_leakage,
        "labels": run_labels,
        "shift": run_shift,
        "anomalies": run_anomalies,
        "report": run_report,
    }

    if args.stage == "all":
        for name in ("profile", "duplicates", "leakage", "labels", "shift", "anomalies", "report"):
            print(f"\n========== {name.upper()} ==========")
            stage_map[name](args)
    else:
        stage_map[args.stage](args)


if __name__ == "__main__":
    main()
