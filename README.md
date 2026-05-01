# 🩸 Blood Cell Classifier

Медицинская система классификации клеток крови на основе глубокого обучения.

## 📊 Характеристики

- **Модель**: ResNet50
- **Точность**: 94.83% (Accuracy)
- **F1-Score**: 94.80%
- **Классы**: 11 типов клеток крови
- **Приоритет**: Максимальная чувствительность к патологическим клеткам (бласты, миелоциты, метамиелоциты)

## 🚀 Быстрый старт

### Предварительные требования

- [Docker](https://docs.docker.com/get-docker/) (версия 24.0+)
- [Docker Compose](https://docs.docker.com/compose/install/) (версия 2.0+)
- NVIDIA GPU + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (опционально, для ускорения на GPU)

### Установка и запуск

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/romaplaton1995-alt/blood-cell-classifier.git
   cd blood-cell-classifier