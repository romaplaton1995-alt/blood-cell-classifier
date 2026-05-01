#!/usr/bin/env python3
"""
Скрипт для проверки работоспособности модели.
Запускается при старте контейнера для тестирования.
"""
import torch
from pathlib import Path
from src.model import load_model, CLASS_NAMES


def check_model():
    """Проверяет наличие и работоспособность модели"""
    print("\n" + "=" * 60)
    print("🩸 BLOOD CELL CLASSIFIER - Model Check")
    print("=" * 60)

    model_path = Path("/app/models/model.pth")

    # Проверяем наличие файла
    if not model_path.exists():
        print("❌ ОШИБКА: Файл модели не найден!")
        print(f"   Путь: {model_path}")
        print("\n💡 Запустите сначала: python scripts/download_weight.py")
        return False

    print(f"✅ Файл модели найден: {model_path}")
    print(f"   Размер: {model_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Определяем устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Устройство: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Загружаем модель
    try:
        print("\n📦 Загрузка модели...")
        model = load_model(str(model_path), device)
        print("✅ Модель успешно загружена")

        # Показываем информацию
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Параметров: {total_params:,}")
        print(f"   Классов: {len(CLASS_NAMES)}")

        print("\n📋 Классы для классификации:")
        for i, cls in enumerate(CLASS_NAMES, 1):
            marker = "⚠️" if cls in ["Бласты", "Миелоциты", "Метамиелоциты"] else "✓"
            print(f"   {i}. {marker} {cls}")

        print("\n" + "=" * 60)
        print("✅ СИСТЕМА ГОТОВА К РАБОТЕ!")
        print("=" * 60)
        print("\n🚀 Для запуска веб-интерфейса выполните:")
        print("   python src/app.py")
        print("\n" + "=" * 60 + "\n")

        return True

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False


if __name__ == "__main__":
    import sys

    success = check_model()
    sys.exit(0 if success else 1)