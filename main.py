"""
Главный скрипт для анализа аномалий в логах безопасности.
"""
import argparse
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

from data_loader import LogDataLoader
from feature_extractor import FeatureExtractor
from anomaly_detector import AnomalyDetector
from clustering import EventClustering
from visualizer import LogVisualizer
from report_generator import ReportGenerator
from evaluator import AnomalyEvaluator


def main():
    """Главная функция для запуска анализа."""
    parser = argparse.ArgumentParser(description='Анализ аномалий в логах безопасности')
    parser.add_argument('--input', type=str, required=True, 
                       help='Путь к файлу с логами (CSV или JSON)')
    parser.add_argument('--output', type=str, default='results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--algorithm', type=str, default='ensemble',
                       choices=['isolation_forest', 'lof', 'one_class_svm', 'ensemble'],
                       help='Алгоритм детекции аномалий')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Доля аномалий (по умолчанию 0.1)')
    parser.add_argument('--visualize', action='store_true',
                       help='Создавать визуализации')
    parser.add_argument('--cluster', action='store_true',
                       help='Выполнять кластеризацию')
    parser.add_argument('--n-clusters', type=int, default=5,
                       help='Количество кластеров для K-Means')
    
    args = parser.parse_args()
    
    # Создание директорий
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'visualizations'), exist_ok=True)
    
    print("=" * 60)
    print("АНАЛИЗ АНОМАЛИЙ В ЛОГАХ БЕЗОПАСНОСТИ")
    print("=" * 60)
    print(f"Входной файл: {args.input}")
    print(f"Выходная директория: {args.output}")
    print(f"Алгоритм: {args.algorithm}")
    print("=" * 60)
    
    # 1. Загрузка данных
    print("\n[1/7] Загрузка данных...")
    loader = LogDataLoader()
    
    if args.input.endswith('.csv'):
        data = loader.load_csv(args.input)
    elif args.input.endswith('.json'):
        data = loader.load_json(args.input)
    else:
        print(f"Ошибка: неподдерживаемый формат файла. Используйте CSV или JSON.")
        sys.exit(1)
    
    if data is None or len(data) == 0:
        print("Ошибка: не удалось загрузить данные или файл пуст.")
        sys.exit(1)
    
    # Проверка наличия меток аномалий для оценки (если есть, сохраняем и удаляем)
    y_true = None
    if 'is_anomaly' in data.columns:
        y_true = data['is_anomaly'].values
        data = data.drop('is_anomaly', axis=1)
        print(f"Обнаружены метки аномалий для оценки: {y_true.sum()} аномалий")
    
    # Предобработка
    data = loader.preprocess(data)
    
    # 2. Извлечение признаков
    print("\n[2/7] Извлечение признаков...")
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(data)
    features_scaled = feature_extractor.scale_features(features, fit=True)
    
    print(f"Извлечено {len(features.columns)} признаков")
    print(f"Размерность данных: {features_scaled.shape}")
    
    # 3. Детекция аномалий
    print("\n[3/7] Детекция аномалий...")
    detector = AnomalyDetector()
    
    # Обучение всех алгоритмов
    detector.fit_isolation_forest(features_scaled, contamination=args.contamination)
    detector.fit_lof(features_scaled, contamination=args.contamination)
    detector.fit_one_class_svm(features_scaled, nu=args.contamination)
    
    # Детекция аномалий
    if args.algorithm == 'ensemble':
        detector_results = detector.detect_anomalies_ensemble(features_scaled)
        anomaly_indices = detector.get_anomaly_indices('ensemble')
    else:
        detector_results = detector.detect_anomalies_ensemble(features_scaled, methods=[args.algorithm])
        anomaly_indices = detector.get_anomaly_indices(args.algorithm)
    
    print(f"Выявлено аномалий: {len(anomaly_indices)} ({len(anomaly_indices)/len(data)*100:.2f}%)")
    
    # Оценка качества (если есть метки)
    if y_true is not None:
        print("\nОценка качества детекции...")
        evaluator = AnomalyEvaluator()
        
        # Собираем только те алгоритмы, которые есть в результатах
        predictions_dict = {}
        if 'isolation_forest' in detector_results.columns:
            predictions_dict['Isolation Forest'] = detector_results['isolation_forest']
        if 'lof' in detector_results.columns:
            predictions_dict['LOF'] = detector_results['lof']
        if 'one_class_svm' in detector_results.columns:
            predictions_dict['One-Class SVM'] = detector_results['one_class_svm']
        if 'ensemble' in detector_results.columns:
            predictions_dict['Ensemble'] = detector_results['ensemble']
        
        if len(predictions_dict) > 0:
            comparison_df = evaluator.compare_algorithms(y_true, predictions_dict)
            print("\nСравнение алгоритмов:")
            print(comparison_df[['method', 'precision', 'recall', 'f1_score']].to_string(index=False))
            
            # Сохранение сравнения
            evaluator.save_comparison_report(comparison_df, os.path.join(args.output, 'algorithm_comparison.csv'))
        else:
            print("Предупреждение: нет результатов для оценки качества.")
    
    # Статистика по алгоритмам
    algorithm_stats = {}
    for col in detector_results.columns:
        if col not in ['ensemble', 'ensemble_score']:
            count = (detector_results[col] == -1).sum()
            algorithm_stats[col] = count
    
    # 4. Кластеризация (опционально)
    cluster_labels = None
    if args.cluster:
        print("\n[4/7] Кластеризация событий...")
        clustering = EventClustering()
        _, cluster_labels = clustering.fit_kmeans(features_scaled, n_clusters=args.n_clusters)
        
        # DBSCAN для сравнения
        _, dbscan_labels = clustering.fit_dbscan(features_scaled)
        print(f"K-Means: {args.n_clusters} кластеров")
        print(f"DBSCAN: {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)} кластеров")
    else:
        print("\n[4/7] Кластеризация пропущена (используйте --cluster для включения)")
    
    # 5. Визуализация (опционально)
    if args.visualize:
        print("\n[5/7] Создание визуализаций...")
        visualizer = LogVisualizer(output_dir=os.path.join(args.output, 'visualizations'))
        
        # Временная линия
        visualizer.plot_timeline(data, anomalies=anomaly_indices)
        
        # Распределение аномалий
        visualizer.plot_anomaly_distribution(data, anomaly_indices)
        
        # Сравнение алгоритмов
        visualizer.plot_algorithm_comparison(algorithm_stats, total_events=len(data))
        
        # Корреляционная матрица
        visualizer.plot_correlation_heatmap(features)
        
        # Кластеры (если выполнена кластеризация)
        if cluster_labels is not None:
            features_reduced = clustering.reduce_dimensions(features_scaled, n_components=2)
            visualizer.plot_clusters(features_reduced, cluster_labels)
    else:
        print("\n[5/7] Визуализация пропущена (используйте --visualize для включения)")
    
    # 6. Генерация отчёта
    print("\n[6/7] Генерация отчёта...")
    report_gen = ReportGenerator(output_dir=args.output)
    
    # Статистика
    statistics = {
        'total_events': len(data),
        'anomalies_detected': len(anomaly_indices),
        'anomaly_percentage': len(anomaly_indices) / len(data) * 100,
        'algorithm_statistics': algorithm_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    # HTML отчёт
    report_gen.generate_html_report(
        data, anomaly_indices, 
        detector_results=detector_results,
        cluster_labels=cluster_labels,
        statistics=statistics
    )
    
    # CSV с аномалиями
    report_gen.save_anomalies_csv(data, anomaly_indices)
    
    # JSON статистика
    report_gen.save_statistics_json(statistics)
    
    # 7. Итоговая информация
    print("\n[7/7] Анализ завершён!")
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ:")
    print(f"  - Всего событий: {len(data)}")
    print(f"  - Выявлено аномалий: {len(anomaly_indices)}")
    print(f"  - Процент аномалий: {len(anomaly_indices)/len(data)*100:.2f}%")
    print("\nСОЗДАННЫЕ ФАЙЛЫ:")
    print(f"  - {os.path.join(args.output, 'anomalies_report.html')}")
    print(f"  - {os.path.join(args.output, 'anomalies.csv')}")
    print(f"  - {os.path.join(args.output, 'statistics.json')}")
    if args.visualize:
        print(f"  - {os.path.join(args.output, 'visualizations', '*.png')}")
    print("=" * 60)


if __name__ == '__main__':
    main()






