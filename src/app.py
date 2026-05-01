#!/usr/bin/env python3
"""
Веб-интерфейс для классификации клеток крови.
Запускается через Gradio.
"""
import torch
import gradio as gr
from pathlib import Path
import torchvision.transforms.v2 as v2
from PIL import Image
from src.model import load_model, predict, CLASS_NAMES, PATHOLOGICAL_CLASSES

# Глобальные переменные (загружаются один раз при старте)
MODEL = None
DEVICE = None
TRANSFORM = None


def init_model():
    """Инициализирует модель при первом запуске"""
    global MODEL, DEVICE, TRANSFORM

    if MODEL is not None:
        return

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("/app/models/model.pth")

    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ Модель не найдена: {model_path}\n"
            f"💡 Запустите сначала: python scripts/download_weight.py"
        )

    print(f"📦 Загрузка модели на {DEVICE}...")
    MODEL = load_model(str(model_path), DEVICE)

    # Те же трансформы, что при обучении (без аугментаций!)
    TRANSFORM = v2.Compose([
        v2.Resize((256, 256), antialias=True),
        v2.CenterCrop((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"✅ Модель готова. Классов: {len(CLASS_NAMES)}")


def classify_image(image: Image.Image) -> tuple:
    """
    Делает предсказание для изображения.

    Returns:
        tuple: (список топ-5 результатов, строка с предупреждением)
    """
    init_model()

    if image is None:
        return [], "⚠️ Загрузите изображение"

    # Конвертируем в RGB на всякий случай
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Предобработка
    tensor = TRANSFORM(image).unsqueeze(0).to(DEVICE)

    # Предсказание
    results = predict(MODEL, tensor, CLASS_NAMES)

    # Формируем вывод для Gradio Label (топ-5)
    output = {}
    for cls, data in list(results.items())[:5]:
        output[cls] = data['probability']  # data['probability'] уже float 0.0-1.0

    # Медицинское предупреждение
    top_class = list(results.keys())[0]
    top_prob = results[top_class]['percentage']

    if top_class in PATHOLOGICAL_CLASSES:
        warning = (
            f"⚠️ ВНИМАНИЕ: Обнаружены потенциально патологические клетки.\n"
            f"Класс: {top_class} (вероятность: {top_prob:.1f}%)\n\n"
            f"🩺 РЕКОМЕНДАЦИЯ: Требуется срочная консультация врача-гематолога.\n"
            f"Не откладывайте очный осмотр!"
        )
    elif top_prob < 50:
        warning = (
            f"⚠️ Низкая уверенность модели ({top_prob:.1f}%).\n"
            f"Рекомендуется повторить анализ или проконсультироваться с врачом."
        )
    else:
        warning = (
            f"✅ Нормальные клетки: {top_class} ({top_prob:.1f}%).\n"
            f"Рекомендуется плановый осмотр согласно графику."
        )

    # Добавляем общий дисклеймер
    warning += "\n\n" + "─" * 50 + "\n"
    warning += "⚠️ Дисклеймер: Система является вспомогательным инструментом.\n"
    warning += "Окончательный диагноз ставит только врач-гематолог."

    return output, warning


def create_interface():
    """Создаёт Gradio интерфейс"""

    with gr.Blocks(
            title="🩸 Blood Cell Classifier",
            theme=gr.themes.Soft(primary_hue="blue"),
            css="""
            .warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 10px; }
            .success { background: #d4edda; border-left: 4px solid #28a745; padding: 10px; }
            .danger { background: #f8d7da; border-left: 4px solid #dc3545; padding: 10px; }
        """
    ) as demo:
        # Заголовок
        gr.Markdown(
            """
            # 🩸 Medical Blood Cell Classifier
            *Классификация клеток крови на основе глубокого обучения (ResNet50)*

            **Характеристики модели:**
            - Точность: 94.83% | F1-Score: 94.80%
            - Классы: 11 типов клеток
            - Приоритет: Максимальная чувствительность к патологическим формам
            """
        )

        # Основная панель
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="📷 Загрузите изображение клетки",
                    height=300
                )
                submit_btn = gr.Button("🔍 Классифицировать", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Очистить", variant="secondary")

            with gr.Column(scale=1):
                output_labels = gr.Label(
                    num_top_classes=5,
                    label="📊 Результаты классификации"
                )
                warning_box = gr.Textbox(
                    label="⚠️ Клиническая рекомендация",
                    interactive=False,
                    lines=8,
                    elem_classes=["warning"]
                )

        # Примеры (опционально)
        gr.Markdown("### 💡 Советы для лучшего результата")
        gr.Markdown(
            """
            - Используйте чёткие изображения клеток в фокусе
            - Избегайте перекрытий и артефактов на снимке
            - Для наилучшей точности используйте изображения ~500×500 px
            """
        )

        # Дисклеймер
        gr.Markdown(
            """
            ---
            > **⚠️ Важное предупреждение**
            > 
            > Данная система предназначена **ТОЛЬКО** для образовательных и исследовательских целей.
            > 
            > **НЕ ИСПОЛЬЗУЙТЕ** для постановки диагнозов без:
            > - Клинической валидации
            > - Одобрения этического комитета  
            > - Сертификации как медицинского изделия
            > 
            > Авторы не несут ответственности за любые последствия использования.
            """
        )

        # Обработчики событий
        submit_btn.click(
            fn=classify_image,
            inputs=image_input,
            outputs=[output_labels, warning_box]
        )

        clear_btn.click(
            fn=lambda: (None, ""),
            inputs=None,
            outputs=[image_input, warning_box]
        )

    return demo


if __name__ == "__main__":
    print("🚀 Запуск веб-интерфейса Blood Cell Classifier...")
    print(f"🌐 Интерфейс будет доступен по адресу: http://localhost:7860")
    print(f"🔗 Для доступа извне используйте: --share или настройте reverse proxy")

    demo = create_interface()

    # Запуск сервера
    demo.launch(
        server_name="0.0.0.0",  # Слушать все интерфейсы (нужно для Docker)
        server_port=7860,
        share=False,  # Установите True для получения публичной ссылки (ngrok)
        quiet=False
    )