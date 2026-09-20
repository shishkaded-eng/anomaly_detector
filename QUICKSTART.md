# Быстрый старт

## Шаг 1: Установка зависимостей

```bash
pip install -r requirements.txt
```

## Шаг 2: Генерация тестовых данных

```bash
python generate_test_data.py
```

Это создаст файл `data/test_logs.csv` с синтетическими данными.

## Шаг 3: Запуск анализа

### Базовый запуск:
```bash
python main.py --input data/test_logs.csv --output results/
```

### С визуализацией и кластеризацией:
```bash
python main.py --input data/test_logs.csv --output results/ --visualize --cluster
```

## Шаг 4: Просмотр результатов

Откройте файл `results/anomalies_report.html` в браузере для просмотра отчёта.

## Тестирование с оценкой качества

Для тестирования системы на данных с известными аномалиями:

```bash
python test_anomaly_detection.py --input data/test_logs.csv
```

Этот скрипт покажет метрики качества (precision, recall, F1-score) для каждого алгоритма.

## Использование своих данных

Подготовьте CSV файл со следующими колонками:
- `timestamp` - время события
- `source_ip` - IP-адрес источника
- `destination_ip` - IP-адрес назначения
- `event_type` - тип события
- `user` - пользователь
- `status` - статус (success/failed)
- `port` - порт (опционально)
- `bytes` - количество байт (опционально)

Затем запустите:
```bash
python main.py --input ваш_файл.csv --output results/ --visualize
```


