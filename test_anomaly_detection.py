"""
Скрипт для тестирования системы детекции аномалий на синтетических данных.
"""
import pandas as pd
import sys
import os

from data_loader import LogDataLoader
from feature_extractor import FeatureExtractor
from anomaly_detector import AnomalyDetector
from evaluator import AnomalyEvaluator


def test_anomaly_detection(input_file='data/test_logs.csv', contamination=0.1):
    """
    Тестирование системы детекции аномалий.
    
    Args:
        input_file: путь к тестовым данным
        contamination: доля аномалий для алгоритмов
    """
    print("=" * 60)
    print("ТЕСТИРОВАНИЕ СИСТЕМЫ ДЕТЕКЦИИ АНОМАЛИЙ")
    print("=" * 60)
    
    # 1. Загрузка данных
    print("\n[1/5] Загрузка тестовых данных...")
    loader = LogDataLoader()
    data = loader.load_csv(input_file)
    
    if data is None or len(data) == 0:
        print("Ошибка: не удалось загрузить тестовые данные.")
        print("Сначала запустите: python generate_test_data.py")
        sys.exit(1)
    
    # Проверка наличия меток аномалий
    if 'is_anomaly' not in data.columns:
        print("Предупреждение: в данных нет меток аномалий (колонка 'is_anomaly').")
        print("Оценка качества невозможна.")
        y_true = None
    else:
        y_true = data['is_anomaly'].values
        n_anomalies_true = y_true.sum()
        print(f"Истинное количество аномалий: {n_anomalies_true} ({n_anomalies_true/len(data)*100:.2f}%)")
        data = data.drop('is_anomaly', axis=1)  # Удаляем метки из данных
    
    # Предобработка
    data = loader.preprocess(data)
    
    # 2. Извлечение признаков
    print("\n[2/5] Извлечение признаков...")
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(data)
    features_scaled = feature_extractor.scale_features(features, fit=True)
    print(f"Извлечено {len(features.columns)} признаков")
    
    # 3. Обучение и тестирование алгоритмов
    print("\n[3/5] Обучение алгоритмов детекции...")
    detector = AnomalyDetector()
    
    # Обучение всех алгоритмов
    detector.fit_isolation_forest(features_scaled, contamination=contamination)
    detector.fit_lof(features_scaled, contamination=contamination)
    detector.fit_one_class_svm(features_scaled, nu=contamination)
    
    # Получение предсказаний
    detector_results = detector.detect_anomalies_ensemble(features_scaled)
    
    # 4. Оценка качества (если есть метки)
    if y_true is not None:
        print("\n[4/5] Оценка качества детекции...")
        evaluator = AnomalyEvaluator()
        
        # Сравнение всех алгоритмов
        predictions_dict = {
            'Isolation Forest': detector_results['isolation_forest'],
            'LOF': detector_results['lof'],
            'One-Class SVM': detector_results['one_class_svm'],
            'Ensemble': detector_results['ensemble']
        }
        
        comparison_df = evaluator.compare_algorithms(y_true, predictions_dict)
        
        # Вывод результатов
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ АЛГОРИТМОВ")
        print("=" * 60)
        print(comparison_df[['method', 'precision', 'recall', 'f1_score', 'accuracy']].to_string(index=False))
        
        # Детальные отчёты
        for method_name in predictions_dict.keys():
            metrics = evaluator.evaluate(y_true, predictions_dict[method_name], method_name)
            evaluator.print_evaluation_report(metrics)
        
        # Сохранение отчёта
        os.makedirs('results', exist_ok=True)
        evaluator.save_comparison_report(comparison_df, 'results/algorithm_comparison.csv')
        
        # Определение лучшего алгоритма
        best_method = comparison_df.loc[comparison_df['f1_score'].idxmax(), 'method']
        print(f"\nЛучший алгоритм по F1-Score: {best_method}")
        print(f"F1-Score: {comparison_df.loc[comparison_df['f1_score'].idxmax(), 'f1_score']:.4f}")
    else:
        print("\n[4/5] Оценка качества пропущена (нет меток аномалий)")
    
    # 5. Статистика по выявленным аномалиям
    print("\n[5/5] Статистика по выявленным аномалиям:")
    print("=" * 60)
    for method in ['isolation_forest', 'lof', 'one_class_svm', 'ensemble']:
        if method in detector_results.columns:
            count = (detector_results[method] == -1).sum() if method != 'ensemble' else detector_results[method].sum()
            percentage = count / len(data) * 100
            print(f"  {method.replace('_', ' ').title()}: {count} ({percentage:.2f}%)")
    
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Тестирование системы детекции аномалий')
    parser.add_argument('--input', type=str, default='data/test_logs.csv',
                       help='Путь к тестовым данным')
    parser.add_argument('--contamination', type=float, default=0.1,
                       help='Доля аномалий для алгоритмов')
    
    args = parser.parse_args()
    test_anomaly_detection(args.input, args.contamination)

