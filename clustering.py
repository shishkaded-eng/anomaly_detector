"""
Модуль кластеризации событий в логах безопасности.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class EventClustering:
    """Класс для кластеризации событий в логах."""
    
    def __init__(self):
        self.kmeans = None
        self.dbscan = None
        self.scaler = StandardScaler()
        self.pca = None
        
    def fit_kmeans(self, features, n_clusters=5, random_state=42):
        """
        Кластеризация с помощью K-Means.
        
        Args:
            features: DataFrame с признаками
            n_clusters: количество кластеров
            random_state: seed для воспроизводимости
            
        Returns:
            Обученная модель и метки кластеров
        """
        # Масштабирование признаков
        features_scaled = self.scaler.fit_transform(features)
        
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = self.kmeans.fit_predict(features_scaled)
        
        # Оценка качества кластеризации
        silhouette = silhouette_score(features_scaled, labels)
        print(f"K-Means обучен: {n_clusters} кластеров, silhouette score = {silhouette:.3f}")
        
        return self.kmeans, labels
    
    def predict_kmeans(self, features):
        """
        Предсказание кластеров с помощью K-Means.
        
        Args:
            features: DataFrame с признаками
            
        Returns:
            Метки кластеров
        """
        if self.kmeans is None:
            raise ValueError("K-Means не обучен. Сначала вызовите fit_kmeans.")
        
        features_scaled = self.scaler.transform(features)
        labels = self.kmeans.predict(features_scaled)
        return labels
    
    def fit_dbscan(self, features, eps=0.5, min_samples=5):
        """
        Кластеризация с помощью DBSCAN.
        
        Args:
            features: DataFrame с признаками
            eps: максимальное расстояние между точками одного кластера
            min_samples: минимальное количество точек для формирования кластера
            
        Returns:
            Обученная модель и метки кластеров
        """
        # Масштабирование признаков
        features_scaled = self.scaler.fit_transform(features)
        
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan.fit_predict(features_scaled)
        
        # Подсчёт кластеров (исключая шум, помеченный как -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"DBSCAN обучен: {n_clusters} кластеров, {n_noise} точек шума (eps={eps}, min_samples={min_samples})")
        
        return self.dbscan, labels
    
    def get_cluster_statistics(self, data, labels, cluster_id):
        """
        Получение статистики по кластеру.
        
        Args:
            data: исходные данные
            labels: метки кластеров
            cluster_id: ID кластера
            
        Returns:
            Словарь со статистикой
        """
        cluster_data = data[labels == cluster_id]
        
        stats = {
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(data) * 100
        }
        
        # Статистика по колонкам
        if 'source_ip' in cluster_data.columns:
            stats['top_source_ips'] = cluster_data['source_ip'].value_counts().head(5).to_dict()
        
        if 'event_type' in cluster_data.columns:
            stats['top_event_types'] = cluster_data['event_type'].value_counts().head(5).to_dict()
        
        if 'status' in cluster_data.columns:
            stats['status_distribution'] = cluster_data['status'].value_counts().to_dict()
        
        if 'timestamp' in cluster_data.columns:
            stats['time_range'] = {
                'start': str(cluster_data['timestamp'].min()),
                'end': str(cluster_data['timestamp'].max())
            }
        
        return stats
    
    def reduce_dimensions(self, features, n_components=2):
        """
        Снижение размерности для визуализации с помощью PCA.
        
        Args:
            features: DataFrame с признаками
            n_components: количество компонент
            
        Returns:
            DataFrame с уменьшенной размерностью
        """
        features_scaled = self.scaler.fit_transform(features)
        self.pca = PCA(n_components=n_components)
        features_reduced = self.pca.fit_transform(features_scaled)
        
        columns = [f'PC{i+1}' for i in range(n_components)]
        return pd.DataFrame(features_reduced, columns=columns, index=features.index)
    
    def find_optimal_k(self, features, max_k=10):
        """
        Поиск оптимального количества кластеров для K-Means.
        
        Args:
            features: DataFrame с признаками
            max_k: максимальное количество кластеров для проверки
            
        Returns:
            Словарь с результатами (k, silhouette_score)
        """
        features_scaled = self.scaler.fit_transform(features)
        results = {}
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            silhouette = silhouette_score(features_scaled, labels)
            results[k] = silhouette
            print(f"K={k}: silhouette score = {silhouette:.3f}")
        
        optimal_k = max(results, key=results.get)
        print(f"Оптимальное количество кластеров: {optimal_k} (silhouette = {results[optimal_k]:.3f})")
        
        return results, optimal_k

