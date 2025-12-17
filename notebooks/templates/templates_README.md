# Шаблоны ноутбуков

Эта папка содержит шаблоны для работы команды.

## Доступные шаблоны

### 1. inference_template.ipynb
Шаблон для тестирования моделей распознавания текста.

**Что делает:**
- Загружает предобученную модель
- Выполняет инференс на тестовых данных
- Замеряет производительность (время, память)
- Сохраняет результаты на Google Drive

**Когда использовать:**
- Тестирование новой модели
- Бенчмаркинг производительности
- Сбор предсказаний для анализа

**Выходные файлы:**
- `predictions.csv` - предсказания модели
- `performance_metrics.json` - метрики производительности

### 2. metrics_template.ipynb
Шаблон для вычисления метрик качества.

**Что делает:**
- Загружает предсказания модели
- Вычисляет метрики (CER, WER, Accuracy)
- Анализирует типы ошибок
- Визуализирует результаты

**Когда использовать:**
- После запуска inference_template.ipynb
- Когда есть ground truth метки
- Для сравнения моделей

**Требования:**
- Файл predictions.csv от inference_template.ipynb
- Ground truth метки

**Выходные файлы:**
- `quality_metrics.json` - метрики качества
- `combined_metrics.json` - объединенные метрики

## Как использовать

### Шаг 1: Скопируйте шаблон

```bash
# На Google Drive создайте копию шаблона
cp inference_template.ipynb my_experiment.ipynb
```

Или в Google Colab: File → Save a copy in Drive

### Шаг 2: Переименуйте

Дайте ноутбуку понятное имя:
- `01_trocr_hwr200_test.ipynb`
- `02_easyocr_school_test.ipynb`
- `03_comparison_models.ipynb`

### Шаг 3: Заполните метаданные

В первой ячейке укажите:
- Ваше имя
- Дату
- Название модели/эксперимента

### Шаг 4: Измените параметры

**В inference_template.ipynb:**
```python
# Измените имя эксперимента
EXPERIMENT_NAME = 'trocr_hwr200_test'  # ← ЗДЕСЬ

# Измените модель
model = TrOCRInference(device=device)  # ← ЗДЕСЬ

# Измените количество изображений
test_images, paths = load_test_images(DATA_PATH, max_images=50)  # ← ЗДЕСЬ
```

**В metrics_template.ipynb:**
```python
# Измените имя эксперимента (должно совпадать с inference)
EXPERIMENT_NAME = 'trocr_hwr200_test'  # ← ЗДЕСЬ

# Добавьте ground truth метки
ground_truth = [...]  # ← ЗДЕСЬ
```

### Шаг 5: Запустите

Runtime → Run all (Ctrl+F9)

## Типичный workflow

```
1. inference_template.ipynb
   ↓ (создает predictions.csv)
2. metrics_template.ipynb
   ↓ (вычисляет метрики)
3. Сравнение результатов
```

## Структура результатов

После запуска шаблонов на Google Drive создается:

```
handwritten-ocr/
└── results/
    └── [EXPERIMENT_NAME]/
        ├── predictions.csv
        ├── performance_metrics.json
        ├── quality_metrics.json
        └── combined_metrics.json
```

## Советы

**Naming conventions:**
- Используйте префиксы: `01_`, `02_`, `03_`
- Указывайте модель: `trocr_`, `easyocr_`
- Указывайте датасет: `hwr200_`, `school_`

**Примеры названий:**
- `01_trocr_hwr200_baseline.ipynb`
- `02_easyocr_hwr200_baseline.ipynb`
- `03_trocr_school_augmented.ipynb`
- `04_comparison_all_models.ipynb`

**Сохранение прогресса:**
- Сохраняйте ноутбуки после каждого запуска
- Коммитьте результаты в GitHub
- Дублируйте важные результаты на Google Drive

**Документирование:**
- Добавляйте выводы в конце ноутбука
- Комментируйте изменения кода
- Сохраняйте интересные графики

## Общие проблемы

**"Файл predictions.csv не найден"**
→ Сначала запустите inference_template.ipynb

**"Ground truth метки отсутствуют"**
→ Добавьте метки в metrics_template.ipynb или пропустите вычисление метрик

**"Модель не загружается"**
→ Проверьте интернет-соединение, возможно нужно перезапустить Colab

**"Закончилась память"**
→ Уменьшите `max_images` или используйте Runtime → Restart runtime

## Дополнительные ресурсы

- [cloud_instructions.md](../../docs/cloud_instructions.md) - Работа с Google Drive
- [GitHub репозиторий](https://github.com/Nadarsa/handwritten-ocr)
- [Документация src/models](../../src/models/)
