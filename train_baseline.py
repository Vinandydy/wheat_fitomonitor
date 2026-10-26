"""Обучение baseline-модели EfficientNet-B5 (классификация болезней пшеницы).

Этап 0 курса «Безопасность и доверенность ИИ»: одна базовая модель
для всех 4 практических работ (ПР1-ПР4).

Запуск (из корня проекта):
    uv run train_baseline.py

Сохраняет в artifacts/:
    models/baseline_effnetb5.pth   лучшие веса (по val Macro-F1)
    train_config.json              конфиг эксперимента (seed, split, версии)
    train_history.json             метрики по эпохам
    test_metrics.json              итоговые метрики на test с 95% CI
    predictions/*.npz              logits/probs/labels/losses для ПР1-ПР4
    plots/training_curves.png      кривые обучения
"""
import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torchvision import models

from Modules import Setup, Savekt, TrainModule
from Modules.Auxiliary_Functions import set_seeds, Graph_loss, print_train_time
from Modules.stats import bootstrap_ci, wilson_ci

BASE = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение baseline EfficientNet-B5")
    parser.add_argument("--data-root", type=Path, default=Path(r"C:\Users\andy7\stud\archive\data"))
    parser.add_argument("--manifest", type=Path, default=BASE / "artifacts" / "dataset_manifest.csv")
    parser.add_argument("--output-dir", type=Path, default=BASE / "artifacts")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=6, help="Early stopping по val Macro-F1")
    parser.add_argument("--no-amp", action="store_true", help="Отключить AMP (смешанную точность)")
    return parser.parse_args()


# Метод создания архитектуры baseline-модели: замороженный B5 + свой head
def create_effnetb5_model(num_classes: int, seed: int = 42):
    # Инициализация базовых весов модели EfficientNet-B5 (ImageNet)
    weights = models.EfficientNet_B5_Weights.DEFAULT
    # Трансформер изображений по структуре весов (resize 456 + нормализация ImageNet)
    transforms = weights.transforms()
    # Стартовая модель
    model = models.efficientnet_b5(weights=weights)

    # Замораживаем backbone: обучается только head
    for param in model.parameters():
        param.requires_grad = False

    # Сид весов head для воспроизводимости
    torch.manual_seed(seed)

    # Преобразуем кол-во классов под нашу задачу (число классов из манифеста)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=2048, out_features=num_classes),
    )
    return model, transforms


# Сбор предсказаний (logits/probs/labels/losses) по всему даталоадеру
@torch.inference_mode()
def collect_predictions(model, loader, device, amp=False):
    model.eval()
    use_amp = amp and device.type == "cuda"
    all_logits, all_labels, all_paths = [], [], []
    offset = 0
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


# Обучение с early stopping по val Macro-F1; лучшие веса сохраняются на диск
def train_with_early_stopping(model, loaders, optimizer, loss_fn, device,
                              epochs, patience, amp, best_path):
    results = {"train_loss": [], "train_acc": [],
               "valid_loss": [], "valid_acc": [], "valid_macro_f1": []}
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = TrainModule.train_step(
            model, loaders["train"], loss_fn, optimizer, device, amp)
        valid_loss, valid_acc = TrainModule.test_step(
            model, loaders["valid"], loss_fn, device, amp)
        valid_f1 = TrainModule.eval_macro_f1(
            model, loaders["valid"], device, amp)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)
        results["valid_macro_f1"].append(valid_f1)

        mark = ""
        if valid_f1 > best_f1:
            best_f1, best_epoch, bad_epochs = valid_f1, epoch, 0
            torch.save(model.state_dict(), best_path)
            mark = "  <- сохранено"
        else:
            bad_epochs += 1
        print(
            f"Epoch: {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f} Acc: {valid_acc:.4f} "
            f"F1: {valid_f1:.4f} | {time.time() - t0:.1f}s{mark}"
        )
        if bad_epochs >= patience:
            print(f"[Инфо] Early stopping на эпохе {epoch} (best epoch {best_epoch})")
            break

    return results, best_epoch


# Сохранение итоговых метрик на test с 95% CI (Уилсон + bootstrap)
def evaluate_test_with_ci(model, loader, device, class_names, seed, out_path):
    logits, probs, labels, losses, paths = collect_predictions(model, loader, device, amp=True)
    y_pred = logits.argmax(1)
    n_correct = int((labels == y_pred).sum())
    n_total = len(labels)

    acc = n_correct / n_total
    acc_lo, acc_hi = wilson_ci(n_correct, n_total)
    macro_f1 = float(f1_score(labels, y_pred, average="macro"))
    _, f1_lo, f1_hi = bootstrap_ci(
        lambda idx: f1_score(labels[idx], y_pred[idx], average="macro"),
        np.arange(n_total), n_boot=2000, seed=seed,
    )

    report = classification_report(
        labels, y_pred, target_names=class_names, digits=4, output_dict=True)
    cm = confusion_matrix(labels, y_pred).tolist()

    test_metrics = {
        "baseline_metric": "Accuracy (основная), Macro-F1 (дополнительная)",
        "n_test": n_total,
        "accuracy": float(acc),
        "accuracy_ci95_wilson": [acc_lo, acc_hi],
        "macro_f1": macro_f1,
        "macro_f1_ci95_bootstrap": [f1_lo, f1_hi],
        "bootstrap_n": 2000,
        "per_class": {name: report[name] for name in class_names},
        "confusion_matrix": cm,
        "class_names": class_names,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    # Отдельно сохраняем предсказания для ПР1-ПР4 (cleanlab, MIA и т.д.)
    return {
        "accuracy": float(acc),
        "accuracy_ci95": [acc_lo, acc_hi],
        "macro_f1": macro_f1,
        "macro_f1_ci95": [f1_lo, f1_hi],
    }, (logits, probs, labels, losses, paths)


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Инфо] Устройство: {device}")
    if device.type == "cuda":
        print(f"[Инфо] GPU: {torch.cuda.get_device_name(0)}")

    # Создаём папки для результатов
    out_dir = args.output_dir.resolve()
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Число классов определяется из манифеста датасета (не хардкод)
    class_names = Setup.read_class_names(args.manifest)
    num_classes = len(class_names)

    # Модель и трансформеры
    model, transform = create_effnetb5_model(num_classes=num_classes, seed=args.seed)

    # Даталоадеры по манифесту
    loaders, class_names_from_data = Setup.create_dataloaders(
        manifest_path=args.manifest,
        data_root=args.data_root,
        transform=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    assert class_names_from_data == class_names, "Классы манифеста не совпадают"
    n_train = len(loaders["train"].dataset)
    n_valid = len(loaders["valid"].dataset)
    n_test = len(loaders["test"].dataset)
    print(f"[Инфо] train/valid/test: {n_train}/{n_valid}/{n_test}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[Инфо] Обучаемых параметров: {sum(p.numel() for p in trainable):,}")

    # Фиксируем конфиг эксперимента (п.1.1 методички — воспроизводимость)
    weights_name = models.EfficientNet_B5_Weights.DEFAULT
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model": "efficientnet_b5",
        "weights": str(weights_name),
        "num_classes": len(class_names),
        "class_names": class_names,
        "head": f"Dropout(0.2) + Linear(2048 -> {num_classes})",
        "backbone_frozen": True,
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": "Adam",
        "loss": "CrossEntropyLoss",
        "amp": not args.no_amp,
        "early_stopping": {"monitor": "valid_macro_f1", "patience": args.patience},
        "data_root": str(args.data_root),
        "manifest": str(args.manifest),
        "split_sizes": {"train": n_train, "valid": n_valid, "test": n_test},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        },
    }
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Обучение
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    amp = not args.no_amp
    best_path = out_dir / "models" / "baseline_effnetb5.pth"

    print(f"[Инфо] Начинаю обучение (AMP={'on' if amp else 'off'})...")
    t_start = time.time()
    results, best_epoch = train_with_early_stopping(
        model, loaders, optimizer, loss_fn, device,
        epochs=args.epochs, patience=args.patience, amp=amp, best_path=best_path,
    )
    total_time = time.time() - t_start
    print_train_time(t_start, time.time(), device)

    # Сохраняем историю и график
    with open(out_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump({"history": results, "best_epoch": best_epoch,
                   "total_time_sec": round(total_time, 1)}, f, ensure_ascii=False, indent=2)
    Graph_loss(results)
    plt.savefig(out_dir / "plots" / "training_curves.png", dpi=150)
    plt.close()

    # Загружаем лучшие веса и оцениваем на test
    print(f"[Инфо] Загружаю лучшие веса (epoch {best_epoch})...")
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))

    # Предсказания по всем сплитам для будущих ПР
    for split in ("train", "valid", "test"):
        logits, probs, labels, losses, paths = collect_predictions(
            model, loaders[split], device, amp=amp)
        np.savez_compressed(
            out_dir / "predictions" / f"{split}_predictions.npz",
            logits=logits, probs=probs, labels=labels, losses=losses,
            paths=np.array(paths), class_names=np.array(class_names),
        )

    # Итоговые метрики на test с CI
    summary, _ = evaluate_test_with_ci(
        model, loaders["test"], device, class_names, args.seed,
        out_dir / "test_metrics.json")
    with open(out_dir / "test_metrics.json", "r", encoding="utf-8") as f:
        test_metrics = json.load(f)
    test_metrics["best_epoch"] = best_epoch
    test_metrics["total_train_time_sec"] = round(total_time, 1)
    test_metrics["split_results"] = {
        split: {"n": len(loaders[split].dataset)}
        for split in ("train", "valid", "test")
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, ensure_ascii=False, indent=2)

    print("\n================ BASELINE (test) ================")
    print(f"Accuracy:  {summary['accuracy']:.4f}  "
          f"95% CI [{summary['accuracy_ci95'][0]:.4f}, {summary['accuracy_ci95'][1]:.4f}] (Wilson)")
    print(f"Macro-F1:  {summary['macro_f1']:.4f}  "
          f"95% CI [{summary['macro_f1_ci95'][0]:.4f}, {summary['macro_f1_ci95'][1]:.4f}] (bootstrap)")
    print(f"Веса:      {best_path}")
    print(f"Метрики:   {out_dir / 'test_metrics.json'}")
    print(f"Предсказания: {out_dir / 'predictions'}")
    print("=================================================")


if __name__ == "__main__":
    main()
