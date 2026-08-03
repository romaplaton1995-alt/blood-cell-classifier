#!/usr/bin/env python3
"""Веб-интерфейс классификатора клеток крови (Gradio)"""
import torch, gradio as gr
from pathlib import Path
import torchvision.transforms.v2 as v2
from PIL import Image
from src.model import load_model, predict, CLASSES, PATHOLOGICAL

# Глобалы — модель грузим один раз
_model = _device = _transform = None

def _init():
    """Инициализация модели (ленивая загрузка)"""
    global _model, _device, _transform
    if _model is not None:
        return
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p = Path("/app/models/model.pth")
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")
    _model = load_model(str(p), _device)
    _transform = v2.Compose([
        v2.Resize((256, 256), antialias=True),
        v2.CenterCrop((224, 224)),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def _classify(img):
    """Основная логика классификации"""
    if img is None:
        return {}, "⚠️ Загрузите изображение"
    _init()
    if isinstance(img, dict):  # Gradio quirk
        img = img.get('composite', img.get('image'))
        if img is None:
            return {}, "❌ Ошибка чтения"
    if img.mode != "RGB":
        img = img.convert("RGB")
    tensor = _transform(img).unsqueeze(0).to(_device)
    res = predict(_model, tensor)
    # Формируем вывод для Gradio Label
    out = {cls: data["prob"] for cls, data in list(res.items())[:5]}
    # Медицинское предупреждение
    top = list(res.keys())[0]
    conf = res[top]["pct"]
    if top in PATHOLOGICAL:
        warn = f"⚠️ Обнаружены патологические клетки: {top} ({conf:.1f}%)\nТребуется консультация гематолога"
    elif conf < 50:
        warn = f"⚠️ Низкая уверенность ({conf:.1f}%). Рекомендуется перепроверка"
    else:
        warn = f"✅ Нормальные клетки: {top} ({conf:.1f}%)"
    warn += "\n\n" + "─"*40 + "\n⚠️ Система — вспомогательный инструмент. Диагноз ставит врач."
    return out, warn

def create_ui():
    """Сборка интерфейса Gradio"""
    with gr.Blocks(title="Blood Cell Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## 🩸 Medical Blood Cell Classifier\n*ResNet50 • 11 классов • F1: 92.3% • Recall бластов: 97%*")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Загрузите изображение")
                btn = gr.Button("Классифицировать", variant="primary")
            with gr.Column():
                out = gr.Label(num_top_classes=5, label="Результаты")
                warn = gr.Textbox(label="Рекомендация", interactive=False, lines=6)
        gr.Markdown("> ⚠️ **Дисклеймер**: Система для образовательных целей. Не для постановки диагноза.")
        btn.click(_classify, inp, [out, warn])
    return demo

if __name__ == "__main__":
    print("Запуск на порту 7860...")
    create_ui().launch(server_name="0.0.0.0", server_port=7860)