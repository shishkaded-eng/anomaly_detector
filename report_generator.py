"""
Модуль генерации отчёта о выявленных аномалиях.
"""
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime


class ReportGenerator:
    """Класс для генерации отчётов о выявленных аномалиях."""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_html_report(self, data, anomaly_indices, detector_results=None, 
                            cluster_labels=None, statistics=None):
        """
        Генерация HTML отчёта о выявленных аномалиях.
        
        Args:
            data: исходные данные
            anomaly_indices: индексы аномалий
            detector_results: результаты детекторов (опционально)
            cluster_labels: метки кластеров (опционально)
            statistics: дополнительная статистика (опционально)
        """
        anomaly_data = data[data.index.isin(anomaly_indices)]
        
        html = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Отчёт о выявленных аномалиях</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            color: #e74c3c;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .anomaly-row {{
            background-color: #fee;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Отчёт о выявленных аномалиях в логах безопасности</h1>
        <p class="timestamp">Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>Общая статистика</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{len(data)}</div>
                <div class="stat-label">Всего событий</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(anomaly_indices)}</div>
                <div class="stat-label">Выявлено аномалий</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(anomaly_indices) / len(data) * 100:.2f}%</div>
                <div class="stat-label">Процент аномалий</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(data) - len(anomaly_indices)}</div>
                <div class="stat-label">Нормальных событий</div>
            </div>
        </div>
"""
        
        # Статистика по алгоритмам
        if detector_results is not None:
            html += """
        <h2>Результаты алгоритмов детекции</h2>
        <table>
            <tr>
                <th>Алгоритм</th>
                <th>Количество аномалий</th>
            </tr>
"""
            for col in detector_results.columns:
                if col not in ['ensemble', 'ensemble_score']:
                    count = (detector_results[col] == -1).sum()
                    html += f"""
            <tr>
                <td>{col.replace('_', ' ').title()}</td>
                <td>{count}</td>
            </tr>
"""
            html += """
        </table>
"""
        
        # Топ подозрительных IP-адресов
        if 'source_ip' in anomaly_data.columns:
            html += """
        <h2>Топ-10 подозрительных IP-адресов</h2>
        <table>
            <tr>
                <th>IP-адрес</th>
                <th>Количество аномалий</th>
            </tr>
"""
            top_ips = anomaly_data['source_ip'].value_counts().head(10)
            for ip, count in top_ips.items():
                html += f"""
            <tr>
                <td>{ip}</td>
                <td>{count}</td>
            </tr>
"""
            html += """
        </table>
"""
        
        # Топ типов событий с аномалиями
        if 'event_type' in anomaly_data.columns:
            html += """
        <h2>Типы событий с аномалиями</h2>
        <table>
            <tr>
                <th>Тип события</th>
                <th>Количество аномалий</th>
            </tr>
"""
            event_counts = anomaly_data['event_type'].value_counts()
            for event_type, count in event_counts.items():
                html += f"""
            <tr>
                <td>{event_type}</td>
                <td>{count}</td>
            </tr>
"""
            html += """
        </table>
"""
        
        # Детальный список аномалий
        html += f"""
        <h2>Детальный список аномалий (первые 100)</h2>
        <table>
            <tr>
"""
        # Заголовки таблицы
        columns_to_show = ['timestamp', 'source_ip', 'destination_ip', 'event_type', 'user', 'status']
        available_columns = [col for col in columns_to_show if col in anomaly_data.columns]
        
        for col in available_columns:
            html += f"<th>{col.replace('_', ' ').title()}</th>"
        html += """
            </tr>
"""
        
        # Данные (первые 100 записей)
        for idx, row in anomaly_data.head(100).iterrows():
            html += '<tr class="anomaly-row">'
            for col in available_columns:
                value = str(row[col]) if pd.notna(row[col]) else 'N/A'
                html += f"<td>{value}</td>"
            html += '</tr>'
        
        html += """
        </table>
"""
        
        # Рекомендации
        html += """
        <h2>Рекомендации</h2>
        <ul>
            <li>Проверить активность подозрительных IP-адресов из топ-10 списка</li>
            <li>Исследовать события, произошедшие в нерабочее время</li>
            <li>Проанализировать типы событий с наибольшим количеством аномалий</li>
            <li>Настроить правила мониторинга для автоматического обнаружения подобных паттернов</li>
            <li>Рассмотреть блокировку IP-адресов с множественными аномалиями</li>
        </ul>
"""
        
        html += """
    </div>
</body>
</html>
"""
        
        filename = os.path.join(self.output_dir, 'anomalies_report.html')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML отчёт сохранён: {filename}")
        return filename
    
    def save_anomalies_csv(self, data, anomaly_indices, filename='anomalies.csv'):
        """
        Сохранение списка аномалий в CSV.
        
        Args:
            data: исходные данные
            anomaly_indices: индексы аномалий
            filename: имя файла
        """
        anomaly_data = data[data.index.isin(anomaly_indices)]
        filepath = os.path.join(self.output_dir, filename)
        anomaly_data.to_csv(filepath, index=False, encoding='utf-8')
        print(f"CSV файл с аномалиями сохранён: {filepath}")
        return filepath
    
    def save_statistics_json(self, statistics, filename='statistics.json'):
        """
        Сохранение статистики в JSON.
        
        Args:
            statistics: словарь со статистикой
            filename: имя файла
        """
        # Преобразуем numpy типы в Python типы для JSON
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        serializable_stats = convert_to_serializable(statistics)
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, ensure_ascii=False, indent=2)
        
        print(f"Статистика сохранена: {filepath}")
        return filepath

