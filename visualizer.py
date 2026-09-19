"""
Модуль визуализации результатов анализа логов безопасности.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime


class LogVisualizer:
    """Класс для визуализации результатов анализа."""
    
    def __init__(self, output_dir='results/visualizations'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10
        
    def plot_timeline(self, data, anomalies=None, title="Временная линия событий"):
        """
        Визуализация временной линии событий.
        
        Args:
            data: DataFrame с данными
            anomalies: индексы аномалий (опционально)
            title: заголовок графика
        """
        if 'timestamp' not in data.columns:
            print("Колонка 'timestamp' не найдена. Пропуск графика временной линии.")
            return
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Сортировка по времени
        data_sorted = data.sort_values('timestamp')
        data_sorted['event_count'] = 1
        data_sorted['cumulative'] = data_sorted['event_count'].cumsum()
        
        # Построение графика
        ax.plot(data_sorted['timestamp'], data_sorted['cumulative'], 
                label='Всего событий', linewidth=2)
        
        # Выделение аномалий
        if anomalies is not None:
            anomaly_data = data_sorted[data_sorted.index.isin(anomalies)]
            if len(anomaly_data) > 0:
                ax.scatter(anomaly_data['timestamp'], 
                          anomaly_data['cumulative'],
                          color='red', s=50, alpha=0.7, 
                          label='Аномалии', zorder=5)
        
        ax.set_xlabel('Время', fontsize=12)
        ax.set_ylabel('Количество событий', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'timeline.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {filename}")
        plt.close()
    
    def plot_anomaly_distribution(self, data, anomaly_indices, title="Распределение аномалий"):
        """
        Визуализация распределения аномалий по различным признакам.
        
        Args:
            data: DataFrame с данными
            anomaly_indices: индексы аномалий
            title: заголовок
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Распределение по часам
        if 'hour' in data.columns:
            ax = axes[0, 0]
            normal_hours = data[~data.index.isin(anomaly_indices)]['hour']
            anomaly_hours = data[data.index.isin(anomaly_indices)]['hour']
            
            ax.hist(normal_hours, bins=24, alpha=0.6, label='Нормальные', color='blue')
            ax.hist(anomaly_hours, bins=24, alpha=0.6, label='Аномалии', color='red')
            ax.set_xlabel('Час дня')
            ax.set_ylabel('Количество событий')
            ax.set_title('Распределение по часам')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Распределение по дням недели
        if 'day_of_week' in data.columns:
            ax = axes[0, 1]
            normal_days = data[~data.index.isin(anomaly_indices)]['day_of_week']
            anomaly_days = data[data.index.isin(anomaly_indices)]['day_of_week']
            
            ax.hist(normal_days, bins=7, alpha=0.6, label='Нормальные', color='blue')
            ax.hist(anomaly_days, bins=7, alpha=0.6, label='Аномалии', color='red')
            ax.set_xlabel('День недели')
            ax.set_ylabel('Количество событий')
            ax.set_title('Распределение по дням недели')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Распределение по IP-адресам (топ-10)
        if 'source_ip' in data.columns:
            ax = axes[1, 0]
            anomaly_data = data[data.index.isin(anomaly_indices)]
            top_ips = anomaly_data['source_ip'].value_counts().head(10)
            
            ax.barh(range(len(top_ips)), top_ips.values, color='red', alpha=0.7)
            ax.set_yticks(range(len(top_ips)))
            ax.set_yticklabels(top_ips.index)
            ax.set_xlabel('Количество аномалий')
            ax.set_title('Топ-10 IP-адресов с аномалиями')
            ax.grid(True, alpha=0.3, axis='x')
        
        # Распределение по типам событий
        if 'event_type' in data.columns:
            ax = axes[1, 1]
            anomaly_data = data[data.index.isin(anomaly_indices)]
            event_counts = anomaly_data['event_type'].value_counts()
            
            ax.bar(range(len(event_counts)), event_counts.values, color='red', alpha=0.7)
            ax.set_xticks(range(len(event_counts)))
            ax.set_xticklabels(event_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Количество аномалий')
            ax.set_title('Распределение по типам событий')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'anomaly_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {filename}")
        plt.close()
    
    def plot_clusters(self, features_reduced, labels, title="Кластеризация событий"):
        """
        Визуализация кластеров в 2D пространстве.
        
        Args:
            features_reduced: DataFrame с уменьшенной размерностью (2D)
            labels: метки кластеров
            title: заголовок
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Уникальные кластеры
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:  # Шум (для DBSCAN)
                col = 'black'
                marker = 'x'
                label = 'Шум'
            else:
                marker = 'o'
                label = f'Кластер {k}'
            
            class_member_mask = (labels == k)
            xy = features_reduced[class_member_mask]
            ax.scatter(xy.iloc[:, 0], xy.iloc[:, 1], 
                      c=[col], marker=marker, s=50, alpha=0.6, label=label)
        
        ax.set_xlabel('Первая главная компонента', fontsize=12)
        ax.set_ylabel('Вторая главная компонента', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'clusters.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {filename}")
        plt.close()
    
    def plot_correlation_heatmap(self, features, title="Корреляционная матрица признаков"):
        """
        Тепловая карта корреляций между признаками.
        
        Args:
            features: DataFrame с признаками
            title: заголовок
        """
        # Выбираем только числовые колонки
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Ограничиваем количество признаков для читаемости
        if len(numeric_features.columns) > 20:
            # Выбираем наиболее важные признаки
            numeric_features = numeric_features.iloc[:, :20]
        
        corr = numeric_features.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=False, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {filename}")
        plt.close()
    
    def plot_algorithm_comparison(self, results_dict, total_events=None, title="Сравнение алгоритмов детекции аномалий"):
        """
        Сравнение результатов различных алгоритмов детекции аномалий.
        
        Args:
            results_dict: словарь с результатами {алгоритм: количество_аномалий}
            total_events: общее количество событий (для расчета процентов)
            title: заголовок
        """
        # Создаём фигуру с двумя подграфиками
        if total_events is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
            ax2 = None
        
        algorithms = list(results_dict.keys())
        counts = list(results_dict.values())
        
        # Цветовая схема для алгоритмов
        colors = {
            'Isolation Forest': '#3498db',
            'isolation_forest': '#3498db',
            'LOF': '#e74c3c',
            'lof': '#e74c3c',
            'Local Outlier Factor': '#e74c3c',
            'One-Class SVM': '#2ecc71',
            'one_class_svm': '#2ecc71',
            'Ensemble': '#f39c12',
            'ensemble': '#f39c12'
        }
        
        # Получаем цвета для каждого алгоритма
        bar_colors = [colors.get(alg, '#95a5a6') for alg in algorithms]
        
        # График 1: Количество аномалий
        bars = ax1.bar(algorithms, counts, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Добавляем значения на столбцы с процентами (если есть общее количество)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            if total_events is not None:
                percentage = (count / total_events) * 100
                label_text = f'{count}\n({percentage:.1f}%)'
            else:
                label_text = f'{count}'
            
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                   label_text,
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Средняя линия (если есть несколько алгоритмов)
        if len(counts) > 1:
            avg_count = np.mean(counts)
            ax1.axhline(y=avg_count, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Среднее: {avg_count:.0f}')
            ax1.legend(loc='upper right', fontsize=10)
        
        ax1.set_ylabel('Количество выявленных аномалий', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Алгоритм детекции', fontsize=13, fontweight='bold')
        ax1.set_title('Количество выявленных аномалий по алгоритмам', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
        ax1.set_axisbelow(True)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
        
        # График 2: Процент аномалий (если есть общее количество)
        if ax2 is not None and total_events is not None:
            percentages = [(count / total_events) * 100 for count in counts]
            bars2 = ax2.bar(algorithms, percentages, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Добавляем значения на столбцы
            for bar, pct in zip(bars2, percentages):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pct:.2f}%',
                       ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Средняя линия
            if len(percentages) > 1:
                avg_pct = np.mean(percentages)
                ax2.axhline(y=avg_pct, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Среднее: {avg_pct:.2f}%')
                ax2.legend(loc='upper right', fontsize=10)
            
            ax2.set_ylabel('Процент аномалий (%)', fontsize=13, fontweight='bold')
            ax2.set_xlabel('Алгоритм детекции', fontsize=13, fontweight='bold')
            ax2.set_title('Процент выявленных аномалий по алгоритмам', fontsize=14, fontweight='bold', pad=15)
            ax2.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            ax2.set_axisbelow(True)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=11)
        
        # Общий заголовок
        if ax2 is not None:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        filename = os.path.join(self.output_dir, 'algorithm_comparison.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {filename}")
        plt.close()



