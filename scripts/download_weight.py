#!/usr/bin/env python3
"""Скрипт для автоматической загрузки весов модели с проверкой"""
import sys
import urllib.request
import urllib.error
from pathlib import Path

# ⚠️ ВСТАВЬТЕ СЮДА ПРАВИЛЬНУЮ ССЫЛКУ (без точки после v!):
MODEL_URL = "https://github.com/romaplaton1995-alt/blood-cell-classifier/releases/download/v2.0.0/best_medical_model.pth"

MODELS_DIR = Path("/app/models")
MODEL_PATH = MODELS_DIR / "model.pth"


def download_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10_000_000:  # >10 MB
        print(f"✅ Модель уже загружена: {MODEL_PATH}")
        print(f"   Размер: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")
        return True

    print(f"\n📥 Загрузка модели:")
    print(f"   URL: {MODEL_URL}")
    print(f"   Путь: {MODEL_PATH}\n")

    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Проверяем, что ссылка ведёт на файл, а не на HTML-страницу
        with urllib.request.urlopen(MODEL_URL, timeout=30) as response:
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' in content_type:
                print(f"❌ Ошибка: Ссылка ведёт на HTML-страницу, а не на файл!")
                print(f"   Content-Type: {content_type}")
                print(f"💡 Проверьте тег релиза на GitHub (должно быть v1.0.0, а не v.1.0.0)")
                return False

        # Скачиваем с прогрессом
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(downloaded * 100 / total_size, 100)
                sys.stdout.write(f"\r⬇️  Загружено: {percent:.1f}%")
                sys.stdout.flush()

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, report_progress)
        print()  # Новая строка после прогресса

        # Проверяем размер скачанного файла
        file_size = MODEL_PATH.stat().st_size
        print(f"\n📊 Размер файла: {file_size / 1024 / 1024:.1f} MB")

        if file_size < 10_000_000:  # Менее 10 MB — скорее всего, это не модель
            print(f"❌ Ошибка: Файл слишком маленький! Возможно, скачана страница ошибки.")
            print(f"💡 Проверьте ссылку и права доступа к релизу на GitHub")
            MODEL_PATH.unlink()  # Удаляем битый файл
            return False

        print(f"✅ Модель успешно загружена!")
        return True

    except urllib.error.HTTPError as e:
        print(f"\n❌ HTTP ошибка {e.code}: {e.reason}")
        print(f"💡 Проверьте, что релиз публичный и ссылка верная")
        return False
    except urllib.error.URLError as e:
        print(f"\n❌ Ошибка сети: {e.reason}")
        print(f"💡 Проверьте интернет-соединение внутри контейнера")
        return False
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)