"""
Модуль оценки качества детекции аномалий.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report


class AnomalyEvaluator:
    """Класс для оценки качества детекции аномалий."""
    
    def __init__(self):
        pass
    
    def evaluate(self, y_true, y_pred, method_name=''):
        """
        Оценка качества детекции.
        
        Args:
            y_true: истинные метки (1 - аномалия, 0 - норма)
            y_pred: предсказанные метки (1 - аномалия, 0 - норма, или -1/1)
            method_name: название метода
            
        Returns:
            Словарь с метриками
        """
        # Преобразуем предсказания в бинарный формат, если нужно
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        # Если предсказания в формате -1/1, преобразуем в 0/1
        if set(np.unique(y_pred)) == {-1, 1}:
            y_pred_binary = (y_pred == -1).astype(int)
        else:
            y_pred_binary = y_pred
        
        # Убеждаемся, что y_true тоже в правильном формате
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        
        # Метрики
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Матрица ошибок
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        elif cm.size == 1:
            # Только один класс предсказан
            if y_pred_binary.sum() == 0:
                tn, fp, fn, tp = (len(y_true) - y_true.sum(), 0, y_true.sum(), 0)
            else:
                tn, fp, fn, tp = (0, y_pred_binary.sum() - y_true.sum(), 0, y_true.sum())
        else:
            tn, fp, fn, tp = (0, 0, 0, 0)
        
        # Дополнительные метрики
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics = {
            'method': method_name,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'specificity': specificity,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def compare_algorithms(self, y_true, predictions_dict):
        """
        Сравнение нескольких алгоритмов.
        
        Args:
            y_true: истинные метки
            predictions_dict: словарь {название_алгоритма: предсказания}
            
        Returns:
            DataFrame с результатами сравнения
        """
        results = []
        
        for method_name, y_pred in predictions_dict.items():
            metrics = self.evaluate(y_true, y_pred, method_name)
            results.append(metrics)
        
        df = pd.DataFrame(results)
        return df
    
    def print_evaluation_report(self, metrics):
        """
        Вывод отчёта об оценке.
        
        Args:
            metrics: словарь с метриками
        """
        print(f"\n{'='*60}")
        print(f"ОЦЕНКА КАЧЕСТВА: {metrics.get('method', 'Unknown')}")
        print(f"{'='*60}")
        print(f"Precision (Точность):  {metrics['precision']:.4f}")
        print(f"Recall (Полнота):      {metrics['recall']:.4f}")
        print(f"F1-Score:              {metrics['f1_score']:.4f}")
        print(f"Accuracy (Точность):   {metrics['accuracy']:.4f}")
        print(f"Specificity:           {metrics['specificity']:.4f}")
        print(f"\nМатрица ошибок:")
        print(f"  True Positives (TP):  {metrics['true_positives']}")
        print(f"  True Negatives (TN):  {metrics['true_negatives']}")
        print(f"  False Positives (FP): {metrics['false_positives']}")
        print(f"  False Negatives (FN): {metrics['false_negatives']}")
        print(f"{'='*60}\n")
    
    def save_comparison_report(self, comparison_df, filename='algorithm_comparison.csv'):
        """
        Сохранение отчёта о сравнении алгоритмов.
        
        Args:
            comparison_df: DataFrame с результатами сравнения
            filename: имя файла
        """
        comparison_df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Отчёт о сравнении алгоритмов сохранён: {filename}")

