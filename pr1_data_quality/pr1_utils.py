"""Вспомогательные функции для ПР1: качество данных."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import csv
import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
import piq
import torch
import torchvision
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from torchvision import models
from tqdm import tqdm

from Modules.stats import bootstrap_ci, wilson_ci
from Modules.wheat_dataset import WheatLeafDataset


def load_manifest(manifest_path: Path) -> List[dict]:
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["class_idx"] = int(row["class_idx"])
            rows.append(row)
    return rows


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_env_info() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "cpu",
    }


def read_image_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# --- Дубликаты ---

def compute_hashes(rows: List[dict], data_root: Path) -> Dict[str, str]:
    hashes = {}
    for row in tqdm(rows, desc="Хеширование"):
        path = data_root / row["path"]
        hashes[row["path"]] = hashlib.md5(path.read_bytes()).hexdigest()
    return hashes


def find_exact_duplicates(rows: List[dict], data_root: Path) -> dict:
    hashes = compute_hashes(rows, data_root)
    hash_to_paths: Dict[str, List[str]] = defaultdict(list)
    for rel, h in hashes.items():
        hash_to_paths[h].append(rel)

    duplicate_groups = [g for g in hash_to_paths.values() if len(g) > 1]
    duplicate_paths = set()
    for g in duplicate_groups:
        duplicate_paths.update(g)

    return {
        "method": "MD5",
        "total_images": len(rows),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": len(duplicate_paths),
        "duplicate_rate": len(duplicate_paths) / len(rows),
        "duplicate_rate_ci95": wilson_ci(len(duplicate_paths), len(rows)),
        "groups": duplicate_groups,
    }


def compute_phashes(rows: List[dict], data_root: Path) -> Dict[str, imagehash.ImageHash]:
    phashes = {}
    for row in tqdm(rows, desc="Perceptual hash"):
        path = data_root / row["path"]
        with Image.open(path) as img:
            phashes[row["path"]] = imagehash.phash(img.convert("RGB"))
    return phashes


def _find_near_pairs(paths: List[str], phashes: Dict[str, imagehash.ImageHash],
                     threshold: int) -> List[Tuple[str, str, int]]:
    pairs = []
    n = len(paths)
    for i in range(n):
        for j in range(i + 1, n):
            diff = phashes[paths[i]] - phashes[paths[j]]
            if diff <= threshold:
                pairs.append((paths[i], paths[j], int(diff)))
    return pairs


def find_near_duplicates(rows: List[dict], data_root: Path,
                         threshold: int = 10) -> dict:
    phashes = compute_phashes(rows, data_root)
    paths = [r["path"] for r in rows]
    pairs = _find_near_pairs(paths, phashes, threshold)
    involved = set()
    for a, b, _ in pairs:
        involved.add(a)
        involved.add(b)

    return {
        "method": f"pHash (threshold={threshold})",
        "total_images": len(rows),
        "near_duplicate_pairs": len(pairs),
        "near_duplicate_images": len(involved),
        "near_duplicate_rate": len(involved) / len(rows),
        "near_duplicate_rate_ci95": wilson_ci(len(involved), len(rows)),
        "pairs": pairs,
    }


def find_train_test_leakage(train_rows: List[dict], test_rows: List[dict],
                            data_root: Path, threshold: int = 10) -> dict:
    train_phashes = compute_phashes(train_rows, data_root)
    test_phashes = compute_phashes(test_rows, data_root)
    pairs = []
    for t_path, t_hash in tqdm(test_phashes.items(), desc="Leakage search"):
        for tr_path, tr_hash in train_phashes.items():
            diff = t_hash - tr_hash
            if diff <= threshold:
                pairs.append((t_path, tr_path, int(diff)))

    involved_test = set(p[0] for p in pairs)
    return {
        "method": f"pHash train-test (threshold={threshold})",
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "leakage_pairs": len(pairs),
        "leaked_test_images": len(involved_test),
        "leakage_rate": len(involved_test) / len(test_rows),
        "leakage_rate_ci95": wilson_ci(len(involved_test), len(test_rows)),
        "pairs": pairs,
    }


# --- Репрезентативность / дрейф (MMD + C2ST) ---

class B5FeatureExtractor(torch.nn.Module):
    """Извлекает 2048-мерные embeddings из EfficientNet-B5 (до classifier)."""

    def __init__(self, weights=None):
        super().__init__()
        if weights is None:
            weights = models.EfficientNet_B5_Weights.DEFAULT
        model = models.efficientnet_b5(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)


@torch.inference_mode()
def extract_embeddings(extractor: B5FeatureExtractor, loader: DataLoader,
                       device: torch.device, amp: bool = False) -> np.ndarray:
    extractor.to(device)
    use_amp = amp and device.type == "cuda"
    feats = []
    for X, _ in tqdm(loader, desc="Embeddings"):
        with torch.amp.autocast("cuda", enabled=use_amp):
            f = extractor(X.to(device, non_blocking=True)).float()
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def rbf_mmd(x: np.ndarray, y: np.ndarray, gamma: Optional[float] = None) -> float:
    n, m = len(x), len(y)
    if gamma is None:
        xy = np.vstack([x, y])
        dists = cdist(xy, xy, metric="sqeuclidean")
        gamma = 1.0 / float(np.median(dists[dists > 0]))

    kxx = np.exp(-gamma * cdist(x, x, metric="sqeuclidean"))
    kyy = np.exp(-gamma * cdist(y, y, metric="sqeuclidean"))
    kxy = np.exp(-gamma * cdist(x, y, metric="sqeuclidean"))

    return float(kxx.sum() / (n * n) + kyy.sum() / (m * m) - 2 * kxy.sum() / (n * m))


def permutation_test_mmd(x: np.ndarray, y: np.ndarray,
                         n_permutations: int = 1000,
                         max_n: int = 2000,
                         seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    # Для permutation test ограничиваем размер выборки, иначе O(n^2) на 1000 перестановок неподъёмно
    if len(x) > max_n:
        x = x[rng.choice(len(x), max_n, replace=False)]
    if len(y) > max_n:
        y = y[rng.choice(len(y), max_n, replace=False)]

    # Вычисляем gamma один раз на полных данных для скорости
    xy = np.vstack([x, y])
    dists = cdist(xy, xy, metric="sqeuclidean")
    gamma = 1.0 / float(np.median(dists[dists > 0]))

    observed = rbf_mmd(x, y, gamma=gamma)
    pooled = np.vstack([x, y])
    n = len(x)
    count = 0
    for _ in range(n_permutations):
        rng.shuffle(pooled)
        perm_x, perm_y = pooled[:n], pooled[n:]
        if rbf_mmd(perm_x, perm_y, gamma=gamma) >= observed:
            count += 1
    return observed, count / n_permutations


def c2st_score(x_train: np.ndarray, x_test: np.ndarray,
               max_n: int = 2000,
               n_bootstraps: int = 1000,
               seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    if len(x_train) > max_n:
        x_train = x_train[rng.choice(len(x_train), max_n, replace=False)]
    if len(x_test) > max_n:
        x_test = x_test[rng.choice(len(x_test), max_n, replace=False)]

    X = np.vstack([x_train, x_test])
    y = np.concatenate([np.zeros(len(x_train)), np.ones(len(x_test))])

    def stat_fn(idx):
        # LogisticRegression заметно быстрее RandomForest на 2048-мерных embeddings
        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(X[idx], y[idx])
        return accuracy_score(y[idx], clf.predict(X[idx]))

    point = stat_fn(np.arange(len(X)))
    n_train, n_test = len(x_train), len(x_test)
    stats = []
    for _ in range(n_bootstraps):
        tr_idx = rng.integers(0, n_train, size=n_train)
        te_idx = rng.integers(0, n_test, size=n_test)
        idx = np.concatenate([tr_idx, n_train + te_idx])
        stats.append(stat_fn(idx))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return float(point), float(lo), float(hi)


# --- Аномалии ---

def compute_image_features(path: Path, use_brisque: bool = True,
                          brisque_max_size: int = 512) -> dict:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        return {"error": "cannot_read"}

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    contrast = float(gray.std())

    brisque = None
    if use_brisque:
        try:
            # BRISQUE очень медленный на полном разрешении; уменьшаем изображение
            scale = min(1.0, brisque_max_size / max(h, w))
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                rgb_small = cv2.cvtColor(
                    cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB,
                )
            else:
                rgb_small = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb_small).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            with torch.inference_mode():
                brisque = float(piq.brisque(tensor))
        except Exception:
            brisque = None

    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / h, 4),
        "laplacian_blur": round(blur, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "brisque": round(brisque, 4) if brisque is not None else None,
    }


def collect_anomaly_features(rows: List[dict], data_root: Path,
                             use_brisque: bool = True) -> List[dict]:
    records = []
    for row in tqdm(rows, desc="Аномалии"):
        feats = compute_image_features(data_root / row["path"], use_brisque=use_brisque)
        feats["path"] = row["path"]
        feats["split"] = row["split"]
        feats["class_name"] = row["class_name"]
        records.append(feats)
    return records


def anomaly_summary(records: List[dict]) -> dict:
    numeric = ["aspect_ratio", "laplacian_blur", "brightness", "contrast", "brisque"]
    summary = {}
    for key in numeric:
        values = [r[key] for r in records if r.get(key) is not None]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "median": float(np.median(arr)),
        }

    broken = [r for r in records if r.get("error")]
    extreme_aspect = [r for r in records if r.get("aspect_ratio", 1) < 0.5
                      or r.get("aspect_ratio", 1) > 2.0]
    low_blur = [r for r in records if r.get("laplacian_blur", 1e9) < 50]

    return {
        "total": len(records),
        "broken": len(broken),
        "broken_rate": len(broken) / len(records),
        "broken_rate_ci95": wilson_ci(len(broken), len(records)),
        "extreme_aspect_count": len(extreme_aspect),
        "low_blur_count": len(low_blur),
        "feature_stats": summary,
    }


# --- Cleanlab: 5-fold CV для out-of-sample вероятностей ---

def create_b5_head_model(num_classes: int, seed: int = 42):
    weights = models.EfficientNet_B5_Weights.DEFAULT
    model = models.efficientnet_b5(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    torch.manual_seed(seed)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=2048, out_features=num_classes),
    )
    return model, weights.transforms()


def train_head_model(model: torch.nn.Module, train_loader: DataLoader,
                     valid_loader: DataLoader, device: torch.device,
                     epochs: int, lr: float, patience: int,
                     amp: bool = False, seed: int = 42) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_loss, bad_epochs = float("inf"), 0
    best_state = None
    for _ in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(X).float()
                val_loss += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
        val_loss /= len(valid_loader.dataset)

        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def compute_oof_probabilities(manifest_path: Path, data_root: Path,
                              num_classes: int, device: torch.device,
                              n_splits: int = 5, epochs: int = 10,
                              batch_size: int = 16, lr: float = 1e-3,
                              patience: int = 3, amp: bool = False,
                              num_workers: int = 4, seed: int = 42) -> dict:
    # Получаем transforms из той же функции, что и для обучения baseline
    _, transform = create_b5_head_model(num_classes, seed=seed)
    full_dataset = WheatLeafDataset(manifest_path, data_root, split="train", transform=transform)
    labels = np.asarray(full_dataset.labels)
    oof_probs = np.zeros((len(labels), num_classes), dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"[Cleanlab] Fold {fold + 1}/{n_splits}")
        train_subset = Subset(full_dataset, tr_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=device.type == "cuda")
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=device.type == "cuda")

        model, _ = create_b5_head_model(num_classes, seed=seed + fold)
        model = train_head_model(model, train_loader, val_loader, device,
                                 epochs=epochs, lr=lr, patience=patience,
                                 amp=amp, seed=seed + fold)

        model.eval()
        with torch.inference_mode():
            offset = 0
            for X, _ in val_loader:
                use_amp = amp and device.type == "cuda"
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(X.to(device)).float()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                oof_probs[val_idx[offset:offset + len(probs)]] = probs
                offset += len(probs)

    return {
        "oof_probs": oof_probs,
        "labels": labels,
        "n_splits": n_splits,
        "epochs": epochs,
    }
