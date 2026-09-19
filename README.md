# Модель интеллектуального поиска аномалий в логах безопасности

## Описание

Система автоматического анализа логов безопасности для выявления подозрительных событий и аномалий с использованием алгоритмов машинного обучения.

## Возможности

- Загрузка и предобработка логов безопасности (CSV, JSON)
- Извлечение признаков из логов (временные, частотные, статистические)
- Детекция аномалий с использованием нескольких алгоритмов:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM
- Кластеризация событий (K-Means, DBSCAN)
- Визуализация результатов анализа
- Генерация отчёта о выявленных аномалиях

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Сгенерируйте тестовые данные (опционально):
```bash
python generate_test_data.py
```

Это создаст файл `data/test_logs.csv` с синтетическими данными, включающими:
- 1000 нормальных событий
- 50 событий брутфорс атаки
- 200 событий DDoS атаки
- 30 событий подозрительной активности

## Использование

### Базовое использование

```bash
python main.py --input data/test_logs.csv --output results/
```

### С визуализацией и кластеризацией

```bash
python main.py --input data/test_logs.csv --output results/ --visualize --cluster
```

### С выбором алгоритма

```bash
python main.py --input data/test_logs.csv --output results/ --algorithm isolation_forest --visualize
```

### Тестирование с оценкой качества

```bash
python test_anomaly_detection.py --input data/test_logs.csv
```

### Параметры командной строки

- `--input` - путь к файлу с логами (CSV или JSON) [обязательно]
- `--output` - директория для сохранения результатов [по умолчанию: results/]
- `--algorithm` - алгоритм детекции: `isolation_forest`, `lof`, `one_class_svm`, `ensemble` [по умолчанию: ensemble]
- `--contamination` - доля аномалий (0.0-1.0) [по умолчанию: 0.1]
- `--visualize` - создавать визуализации
- `--cluster` - выполнять кластеризацию
- `--n-clusters` - количество кластеров для K-Means [по умолчанию: 5]

## Формат входных данных

CSV файл с колонками:
- `timestamp` - время события
- `source_ip` - IP-адрес источника
- `destination_ip` - IP-адрес назначения
- `event_type` - тип события
- `user` - пользователь
- `status` - статус (success/failed)
- `port` - порт (опционально)
- `bytes` - количество байт (опционально)

## Структура проекта

- `main.py` - главный скрипт для запуска анализа
- `data_loader.py` - загрузка и предобработка логов
- `feature_extractor.py` - извлечение признаков
- `anomaly_detector.py` - детекция аномалий
- `clustering.py` - кластеризация событий
- `visualizer.py` - визуализация результатов
- `report_generator.py` - генерация отчёта
- `data/` - директория для входных логов
- `results/` - директория для результатов анализа

## Результаты

После выполнения анализа в директории `results/` будут созданы:
- `anomalies_report.html` - HTML отчёт о выявленных аномалиях
- `anomalies.csv` - список аномалий в CSV формате
- `statistics.json` - статистика по анализу
- `algorithm_comparison.csv` - сравнение алгоритмов (если есть метки)
- `visualizations/` - графики и визуализации (если использован флаг --visualize):
  - `timeline.png` - временная линия событий
  - `anomaly_distribution.png` - распределение аномалий
  - `algorithm_comparison.png` - сравнение алгоритмов
  - `correlation_heatmap.png` - корреляционная матрица
  - `clusters.png` - визуализация кластеров (если использован --cluster)

## Примеры использования

### Пример 1: Полный анализ с визуализацией

```bash
python main.py --input data/test_logs.csv --output results/ --visualize --cluster --algorithm ensemble
```

### Пример 2: Быстрый анализ без визуализации

```bash
python main.py --input data/test_logs.csv --output results/ --algorithm isolation_forest
```

### Пример 3: Тестирование с оценкой качества

```bash
# Сначала сгенерируйте тестовые данные
python generate_test_data.py

# Затем запустите тестирование
python test_anomaly_detection.py --input data/test_logs.csv
```

