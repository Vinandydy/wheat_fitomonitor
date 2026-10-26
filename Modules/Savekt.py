import torch
from pathlib import Path


# Метод для сохранения модели
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    # Создаём директорию для сохранения модели (если её нет)
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Проверяем, что имя модели заканчивается на .pth или .pt
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Сохраняем state_dict() нашей модели (лучшая практика:)
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)
