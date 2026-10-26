"""Модуль обучения: train/test шаги и полный цикл train().

Сохранена структура легаси-проекта; добавлены AMP (смешанная точность)
и расчёт Macro-F1 на валидации.
"""
import torch

from sklearn.metrics import f1_score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


# Метод для проведения одного шага тренировки
def train_step(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         optimizer: torch.optim.Optimizer,
         device: torch.device,
         amp: bool = False) -> Tuple[float, float]:

    model.train()
    # Смешанная точность ускоряет свёртки на GPU (Tensor Cores)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Переменные, которые отслеживают результаты тренеровки
    train_loss, train_acc = 0, 0

    # Засунем в пакеты данные из даталоадера и отправим на девайс
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        # Обратная пропагация ошибки (через scaler для AMP)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        # Расчет вероятностей (кол-во верных обнаружений)
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # Полученные метрики
    return train_loss, train_acc


# Метод для проведения одного шага теста (оценки)
@torch.inference_mode()
def test_step(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         device: torch.device,
         amp: bool = False) -> Tuple[float, float]:

    model.eval()
    use_amp = amp and device.type == "cuda"

    # Переменные, отслеживающие результат тренеровки
    test_loss, test_acc = 0, 0

    # Режим инференса модели
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)

        test_loss += loss.item()

        # Расчет точности модели (тестовый)
        test_pred_labels = test_pred_logits.argmax(dim=1)
        test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Получаем среднее значение
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc


# Оценка Macro-F1 на валидации (для мультикласса с дисбалансом)
@torch.inference_mode()
def eval_macro_f1(model: torch.nn.Module,
           dataloader: torch.utils.data.DataLoader,
           device: torch.device,
           amp: bool = False) -> float:

    model.eval()
    use_amp = amp and device.type == "cuda"
    all_preds, all_labels = [], []

    for X, y in dataloader:
        X = X.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            preds = model(X).argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y)

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    return float(f1_score(y_true, y_pred, average="macro"))


# Полная тренировка модели
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epoch: int,
          device: torch.device,
          amp: bool = False) -> Dict[str, List]:

    # Словарь, в котором будут вписаны результаты
    results = {"train_loss": [], "train_acc": [],
               "valid_loss": [], "valid_acc": [], "valid_macro_f1": []}

    # Отправка модели на целевой девайс
    model.to(device)

    # Цикл для прохода тренировочного и тестировочного шага
    for epoch in tqdm(range(epoch)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            amp=amp,
        )
        valid_loss, valid_acc = test_step(
            model=model,
            dataloader=valid_dataloader,
            loss_fn=loss_fn,
            device=device,
            amp=amp,
        )
        valid_f1 = eval_macro_f1(
            model=model,
            dataloader=valid_dataloader,
            device=device,
            amp=amp,
        )

        # Результаты очередной эпохи
        print(
            f"Epoch: {epoch + 1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid Acc: {valid_acc:.4f} | "
            f"Valid Macro-F1: {valid_f1:.4f}"
        )

        # Обновляем словарь со значениями (для графика)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)
        results["valid_macro_f1"].append(valid_f1)

    return results
