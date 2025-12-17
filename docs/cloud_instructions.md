# Инструкция по работе с Google Drive из Google Colab

## Подключение Google Drive

### Шаг 1: Монтирование

```python
from google.colab import drive
drive.mount('/content/drive')
```

После выполнения кода:
1. Появится ссылка для авторизации
2. Перейдите по ссылке и разрешите доступ
3. Скопируйте код авторизации и вставьте в поле ввода

### Шаг 2: Настройка путей к проекту

```python
import os

# Базовый путь к проекту на Google Drive
PROJECT_PATH = '/content/drive/MyDrive/Практикум_3_семестр/handwritten-ocr'
DATA_PATH = os.path.join(PROJECT_PATH, 'data')

# Пути к сырым данным
RAW_DATA_PATH = os.path.join(DATA_PATH, 'raw')
HWR200_RAW = os.path.join(RAW_DATA_PATH, 'hwr200')
SCHOOL_RAW = os.path.join(RAW_DATA_PATH, 'school_notebooks_ru')

# Пути к обработанным данным
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, 'processed')
HWR200_PROCESSED = os.path.join(PROCESSED_DATA_PATH, 'hwr200')
SCHOOL_PROCESSED = os.path.join(PROCESSED_DATA_PATH, 'school_notebooks_ru')

# Проверка существования
print("Проверка путей:")
print(f"Проект: {os.path.exists(PROJECT_PATH)}")
print(f"HWR200 raw: {os.path.exists(HWR200_RAW)}")
print(f"School raw: {os.path.exists(SCHOOL_RAW)}")
```

## Загрузка данных

### Базовый класс для работы с данными

```python
import cv2
import pandas as pd
from pathlib import Path
from typing import List, Tuple

class DatasetLoader:
    """Класс для загрузки датасетов с Google Drive."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / 'images'
        
    def get_image_paths(self) -> List[str]:
        """Получить список путей к изображениям."""
        if not self.images_path.exists():
            print(f"Папка не найдена: {self.images_path}")
            return []
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(self.images_path.glob(f'*{ext}'))
        
        return [str(p) for p in sorted(image_paths)]
    
    def load_image(self, image_path: str):
        """Загрузить одно изображение."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Не удалось загрузить: {image_path}")
        return image
    
    def load_batch(self, image_paths: List[str], max_images: int = None):
        """Загрузить батч изображений."""
        if max_images:
            image_paths = image_paths[:max_images]
        
        images = []
        for path in image_paths:
            try:
                img = self.load_image(path)
                images.append(img)
            except Exception as e:
                print(f"Ошибка при загрузке {path}: {e}")
        
        return images
```

### Загрузка датасета HWR200

```python
# Инициализация загрузчика
hwr200_loader = DatasetLoader(HWR200_PROCESSED)

# Получить список изображений
image_paths = hwr200_loader.get_image_paths()
print(f"Найдено изображений: {len(image_paths)}")

# Загрузить первые 10 изображений
images = hwr200_loader.load_batch(image_paths, max_images=10)
print(f"Загружено изображений: {len(images)}")

# Просмотр первого изображения
import matplotlib.pyplot as plt

if images:
    plt.figure(figsize=(10, 4))
    plt.imshow(cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title('Пример изображения из HWR200')
    plt.show()
```

### Загрузка с прогресс-баром

```python
from tqdm import tqdm

def load_images_with_progress(image_paths: List[str], max_images: int = None):
    """Загрузить изображения с прогресс-баром."""
    if max_images:
        image_paths = image_paths[:max_images]
    
    images = []
    for path in tqdm(image_paths, desc="Загрузка изображений"):
        try:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
        except Exception as e:
            print(f"Ошибка: {e}")
    
    return images

# Использование
images = load_images_with_progress(image_paths, max_images=100)
```

## Работа с метаданными

### Загрузка описания датасета

```python
# Путь к описанию датасета
dataset_description_path = os.path.join(DATA_PATH, 'description_of_the_dataset.md')

# Чтение описания
if os.path.exists(dataset_description_path):
    with open(dataset_description_path, 'r', encoding='utf-8') as f:
        description = f.read()
    print("Описание датасета:")
    print(description[:500])  # Первые 500 символов
else:
    print("Файл описания не найден")
```

### Сохранение метаданных

```python
import json

# Сохранение информации о датасете
dataset_info = {
    'name': 'HWR200',
    'total_images': len(image_paths),
    'image_format': 'jpg',
    'image_size': 'variable',
    'language': 'Russian',
    'created_at': '2024-12-10'
}

# Сохранить на Google Drive
metadata_path = os.path.join(PROCESSED_DATA_PATH, 'dataset_metadata.json')
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_info, f, ensure_ascii=False, indent=2)

print(f"Метаданные сохранены: {metadata_path}")
```

## Сохранение результатов

### Сохранение результатов инференса

```python
def save_inference_results(predictions: List[str], 
                          image_paths: List[str],
                          output_dir: str):
    """Сохранить результаты распознавания."""
    results_path = os.path.join(PROJECT_PATH, 'results', output_dir)
    os.makedirs(results_path, exist_ok=True)
    
    # Сохранить в CSV
    df = pd.DataFrame({
        'image_path': image_paths,
        'predicted_text': predictions
    })
    
    csv_path = os.path.join(results_path, 'predictions.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Результаты сохранены: {csv_path}")
    
    return csv_path

# Использование
predictions = ["Привет", "мир", "текст"]  # Примеры
image_paths_subset = image_paths[:3]
save_inference_results(predictions, image_paths_subset, 'experiment_01')
```

### Сохранение метрик

```python
def save_metrics(metrics: dict, experiment_name: str):
    """Сохранить метрики эксперимента."""
    results_path = os.path.join(PROJECT_PATH, 'results')
    os.makedirs(results_path, exist_ok=True)
    
    metrics_file = os.path.join(results_path, f'{experiment_name}_metrics.json')
    
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    print(f"Метрики сохранены: {metrics_file}")

# Использование
metrics = {
    'model': 'TrOCR',
    'cer': 12.5,
    'wer': 18.3,
    'inference_time_ms': 45.2,
    'num_samples': 100
}
save_metrics(metrics, 'trocr_experiment_01')
```

## Клонирование GitHub репозитория

```python
# Клонировать репозиторий в Colab
!git clone https://github.com/Nadarsa/handwritten-ocr.git
%cd handwritten-ocr

# Установить зависимости
!pip install -r requirements.txt

# Добавить src в Python path
import sys
sys.path.append('/content/handwritten-ocr/src')

# Теперь можно импортировать модули
from metrics import MetricsEvaluator
from data_augmentation import DataAugmentor
```

## Полный workflow

```python
# 1. Подключение Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Настройка путей
PROJECT_PATH = '/content/drive/MyDrive/Практикум_3_семестр/handwritten-ocr'
HWR200_PROCESSED = os.path.join(PROJECT_PATH, 'data/processed/hwr200')

# 3. Клонирование репо
!git clone https://github.com/Nadarsa/handwritten-ocr.git
%cd handwritten-ocr
!pip install -r requirements.txt

# 4. Импорт модулей
import sys
sys.path.append('/content/handwritten-ocr/src')
from metrics import MetricsEvaluator
from data_augmentation import DatasetLoader

# 5. Загрузка данных
loader = DatasetLoader(HWR200_PROCESSED)
image_paths = loader.get_image_paths()
images = loader.load_batch(image_paths, max_images=50)

# 6. Работа с моделью
# ... ваш код ...

# 7. Сохранение результатов
save_inference_results(predictions, image_paths, 'my_experiment')
save_metrics(metrics, 'my_experiment')
```

## Важные замечания

**Лимиты Google Colab:**
- Сессия отключается после 12 часов работы
- 90 минут бездействия приводят к отключению
- Стандартная RAM: ~12 GB
- GPU RAM: ~15 GB (если включен GPU)

**Советы по работе:**
1. Сохраняйте промежуточные результаты на Drive
2. Используйте батчи при работе с большими данными
3. Периодически сохраняйте прогресс
4. Для больших экспериментов используйте Kaggle Notebooks (больше RAM)

**Синхронизация с GitHub:**
```python
# Сохранить изменения в GitHub
!git add .
!git commit -m "Обновление ноутбука"
!git push
```

**Примечание:** При push может потребоваться авторизация через Personal Access Token.

