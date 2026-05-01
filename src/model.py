"""
Архитектура модели ResNet50 для классификации клеток крови.
"""
import torch
import torch.nn as nn
from torchvision import models

# Названия классов (порядок важен!)
CLASS_NAMES = [
    "Базофилы",
    "Бласты",
    "Лимфоциты",
    "Метамиелоциты",
    "Миелоциты",
    "Моноциты",
    "Нормобласты",
    "Палочкоядерные нейтрофилы",
    "Сегментоядерные нейтрофилы",
    "Тромбоциты гигантские",
    "Эозинофилы"
]

# Классы, требующие особого внимания (опухолевые/патологические)
PATHOLOGICAL_CLASSES = ["Бласты", "Миелоциты", "Метамиелоциты"]


def get_model(num_classes: int = 11, device: str = "cpu") -> nn.Module:
    """
    Создаёт и загружает модель ResNet50.

    Args:
        num_classes: Количество классов для классификации
        device: Устройство для вычислений ('cuda' или 'cpu')

    Returns:
        Готовая к использованию модель
    """
    # Создаём архитектуру без предобученных весов
    model = models.resnet50(weights=None)

    # Заменяем последний слой под наше количество классов
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Переносим на нужное устройство
    model = model.to(device)
    model.eval()  # Режим инференса

    return model


def load_model(model_path: str, device: str = "cpu") -> nn.Module:
    """
    Загружает веса модели с диска.

    Args:
        model_path: Путь к файлу с весами
        device: Устройство для вычислений

    Returns:
        Модель с загруженными весами
    """
    model = get_model(num_classes=11, device=device)

    # Загружаем веса
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    return model


def predict(model: nn.Module, image_tensor: torch.Tensor,
            class_names: list = CLASS_NAMES) -> dict:
    """
    Делает предсказание для изображения.

    Args:
        model: Обученная модель
        image_tensor: Тензор изображения [1, 3, 224, 224]
        class_names: Список названий классов

    Returns:
        Словарь с результатами: {class: probability, ...}
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    results = {}
    for i, (cls, prob) in enumerate(zip(class_names, probabilities)):
        results[cls] = {
            'probability': float(prob),
            'percentage': float(prob * 100),
            'is_pathological': cls in PATHOLOGICAL_CLASSES
        }

    # Сортируем по убыванию вероятности
    results = dict(sorted(results.items(),
                          key=lambda x: x[1]['probability'],
                          reverse=True))

    return results