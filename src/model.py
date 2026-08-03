"""
Модуль модели ResNet50 для классификации клеток крови.
"""
import torch
import torch.nn as nn
from torchvision import models

# Список классов
CLASSES = [
    "Базофилы", "Бласты", "Лимфоциты", "Метамиелоциты", "Миелоциты",
    "Моноциты", "Нормобласты", "Палочкоядерные нейтрофилы",
    "Сегментоядерные нейтрофилы", "Тромбоциты гигантские", "Эозинофилы"
]

# Классы, требующие особого внимания (опухолевые/незрелые формы)
PATHOLOGICAL = ["Бласты", "Миелоциты", "Метамиелоциты"]

CLASS_NAMES = CLASSES

def get_model(num_classes=11, device="cpu"):
    """Создаёт и настраивает модель ResNet50"""
    # Берём архитектуру без предобученных весов, после будут загружены свои веса
    model = models.resnet50(weights=None)
    # Меняем последний слой под наше количество классов
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def load_model(path, device="cpu"):
    """Загружает веса модели с диска"""
    model = get_model(num_classes=11, device=device)
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:  # новый формат чекпоинта
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.eval()
    return model


def predict(model, tensor, classes=CLASSES):
    """Функция делает предсказание для входного тензора"""
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]

    result = {}
    for cls, p in zip(classes, probs):
        result[cls] = {
            "prob": float(p),
            "pct": float(p * 100),
            "path": cls in PATHOLOGICAL
        }
    # Сортируем по убыванию вероятности
    return dict(sorted(result.items(), key=lambda x: x[1]["prob"], reverse=True))