"""Сборка манифеста датасета болезней пшеницы.

Сканирует <data-root>/train|valid|test, нормализует имена классов
("Black Rust" / "black_rust_test" -> "black_rust") и сохраняет:
  - artifacts/dataset_manifest.csv  (path, split, class_name, class_idx)
  - artifacts/dataset_profile.json  (счётчики, баланс, энтропия)
  - artifacts/class_names.txt

Запуск (из корня проекта):
    uv run prepare_dataset.py
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


# Приводим имя папки класса к единому виду: lower + underscore, без суффикса сплита
def normalize_class_name(folder: str) -> str:
    name = folder.strip().lower().replace(" ", "_")
    for suffix in ("_valid", "_test", "_train"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


# Сканируем все сплиты и собираем строки манифеста
def scan(data_root: Path) -> list[dict]:
    rows = []
    for split in SPLITS:
        split_dir = data_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Не найдена папка сплита: {split_dir}")
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            class_name = normalize_class_name(class_dir.name)
            files = sorted(
                f for f in class_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in IMG_EXTS
            )
            for f in files:
                rows.append({
                    "path": f.relative_to(data_root).as_posix(),
                    "split": split,
                    "class_name": class_name,
                })
    return rows


# Считаем профиль качества: баланс классов, энтропию, счётчики
def build_profile(rows: list[dict]) -> dict:
    class_names = sorted({r["class_name"] for r in rows})
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    for r in rows:
        r["class_idx"] = class_to_idx[r["class_name"]]

    per_split = defaultdict(Counter)
    for r in rows:
        per_split[r["split"]][r["class_name"]] += 1

    train_counts = per_split["train"]
    counts = list(train_counts.values())
    total = sum(counts)
    probs = [c / total for c in counts]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(counts))

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_classes": len(class_names),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "counts": {split: dict(cnt) for split, cnt in per_split.items()},
        "totals": {split: sum(cnt.values()) for split, cnt in per_split.items()},
        "train_imbalance_ratio": max(counts) / min(counts),
        "train_class_entropy": entropy,
        "train_class_entropy_max": max_entropy,
        "train_class_entropy_norm": entropy / max_entropy,
    }


def main():
    parser = argparse.ArgumentParser(description="Сборка манифеста датасета")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"C:\Users\andy7\stud\archive\data"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=BASE / "artifacts"
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Инфо] Сканирую датасет: {data_root}")
    rows = scan(data_root)
    profile = build_profile(rows)

    manifest_path = out_dir / "dataset_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "split", "class_name", "class_idx"]
        )
        writer.writeheader()
        writer.writerows(rows)

    profile_path = out_dir / "dataset_profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

    classes_path = out_dir / "class_names.txt"
    classes_path.write_text("\n".join(profile["class_names"]), encoding="utf-8")

    print(f"[Инфо] Классов: {profile['n_classes']}")
    print(f"[Инфо] Всего изображений: {sum(profile['totals'].values())}")
    for split in SPLITS:
        print(f"  {split}: {profile['totals'].get(split, 0)}")
    print(f"[Инфо] Disbalance ratio (train): "
          f"{profile['train_imbalance_ratio']:.2f}")
    print(f"[Инфо] Манифест: {manifest_path}")
    print(f"[Инфо] Профиль:  {profile_path}")


if __name__ == "__main__":
    main()
