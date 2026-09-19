"""
Модуль детекции аномалий в логах безопасности.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score


class AnomalyDetector:
    """Класс для детекции аномалий с использованием различных алгоритмов."""
    
    def __init__(self):
        self.isolation_forest = None
        self.lof = None
        self.one_class_svm = None
        self.ensemble_predictions = None
        
    def fit_isolation_forest(self, features, contamination=0.1, random_state=42):
        """
        Обучение Isolation Forest.
        
        Args:
            features: DataFrame с признаками
            contamination: доля аномалий (по умолчанию 0.1 = 10%)
            random_state: seed для воспроизводимости
            
        Returns:
            Обученная модель
        """
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.isolation_forest.fit(features)
        print(f"Isolation Forest обучен (contamination={contamination})")
        return self.isolation_forest
    
    def predict_isolation_forest(self, features):
        """
        Предсказание аномалий с помощью Isolation Forest.
        
        Args:
            features: DataFrame с признаками
            
        Returns:
            Массив предсказаний: 1 для нормальных, -1 для аномалий
        """
        if self.isolation_forest is None:
            raise ValueError("Isolation Forest не обучен. Сначала вызовите fit_isolation_forest.")
        
        predictions = self.isolation_forest.predict(features)
        return predictions
    
    def fit_lof(self, features, n_neighbors=20, contamination=0.1):
        """
        Обучение Local Outlier Factor.
        
        Args:
            features: DataFrame с признаками
            n_neighbors: количество соседей
            contamination: доля аномалий
            
        Returns:
            Обученная модель
        """
        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True
        )
        self.lof.fit(features)
        print(f"LOF обучен (n_neighbors={n_neighbors}, contamination={contamination})")
        return self.lof
    
    def predict_lof(self, features):
        """
        Предсказание аномалий с помощью LOF.
        
        Args:
            features: DataFrame с признаками
            
        Returns:
            Массив предсказаний: 1 для нормальных, -1 для аномалий
        """
        if self.lof is None:
            raise ValueError("LOF не обучен. Сначала вызовите fit_lof.")
        
        predictions = self.lof.predict(features)
        return predictions
    
    def fit_one_class_svm(self, features, nu=0.1, kernel='rbf', gamma='scale'):
        """
        Обучение One-Class SVM.
        
        Args:
            features: DataFrame с признаками
            nu: верхняя граница доли аномалий
            kernel: тип ядра ('rbf', 'linear', 'poly')
            gamma: параметр для RBF ядра
            
        Returns:
            Обученная модель
        """
        self.one_class_svm = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        self.one_class_svm.fit(features)
        print(f"One-Class SVM обучен (nu={nu}, kernel={kernel})")
        return self.one_class_svm
    
    def predict_one_class_svm(self, features):
        """
        Предсказание аномалий с помощью One-Class SVM.
        
        Args:
            features: DataFrame с признаками
            
        Returns:
            Массив предсказаний: 1 для нормальных, -1 для аномалий
        """
        if self.one_class_svm is None:
            raise ValueError("One-Class SVM не обучен. Сначала вызовите fit_one_class_svm.")
        
        predictions = self.one_class_svm.predict(features)
        return predictions
    
    def detect_anomalies_ensemble(self, features, methods=['isolation_forest', 'lof', 'one_class_svm']):
        """
        Детекция аномалий с использованием ансамбля методов.
        
        Args:
            features: DataFrame с признаками
            methods: список методов для использования
            
        Returns:
            DataFrame с результатами всех методов
        """
        results = pd.DataFrame(index=features.index)
        
        if 'isolation_forest' in methods and self.isolation_forest is not None:
            results['isolation_forest'] = self.predict_isolation_forest(features)
        
        if 'lof' in methods and self.lof is not None:
            results['lof'] = self.predict_lof(features)
        
        if 'one_class_svm' in methods and self.one_class_svm is not None:
            results['one_class_svm'] = self.predict_one_class_svm(features)
        
        # Объединение результатов: аномалия, если хотя бы один метод определил
        if len(results.columns) > 0:
            # Преобразуем: -1 -> 1 (аномалия), 1 -> 0 (норма)
            results_binary = (results == -1).astype(int)
            results['ensemble'] = (results_binary.sum(axis=1) > 0).astype(int)
            results['ensemble_score'] = results_binary.sum(axis=1) / len(results.columns)
        
        self.ensemble_predictions = results
        return results
    
    def get_anomaly_indices(self, method='ensemble'):
        """
        Получение индексов аномалий.
        
        Args:
            method: метод для использования ('isolation_forest', 'lof', 'one_class_svm', 'ensemble')
            
        Returns:
            Массив индексов аномалий
        """
        if self.ensemble_predictions is None:
            raise ValueError("Сначала выполните detect_anomalies_ensemble.")
        
        if method == 'ensemble':
            anomalies = self.ensemble_predictions[self.ensemble_predictions['ensemble'] == 1].index
        elif method in self.ensemble_predictions.columns:
            anomalies = self.ensemble_predictions[self.ensemble_predictions[method] == -1].index
        else:
            raise ValueError(f"Метод {method} не найден.")
        
        return anomalies.tolist()
    
    def evaluate(self, y_true, y_pred):
        """
        Оценка качества детекции аномалий.
        
        Args:
            y_true: истинные метки (1 - аномалия, 0 - норма)
            y_pred: предсказанные метки (1 - аномалия, 0 - норма)
            
        Returns:
            Словарь с метриками
        """
        # Преобразуем предсказания в бинарный формат, если нужно
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values
        
        # Если предсказания в формате -1/1, преобразуем в 0/1
        if set(np.unique(y_pred)) == {-1, 1}:
            y_pred = (y_pred == -1).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        return metrics

