"""
Модуль извлечения признаков из логов безопасности.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter


class FeatureExtractor:
    """Класс для извлечения признаков из логов."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def extract_features(self, data):
        """
        Извлечение признаков из данных логов.
        
        Args:
            data: DataFrame с логами
            
        Returns:
            DataFrame с признаками
        """
        features = pd.DataFrame()
        
        # Временные признаки (уже должны быть в data после предобработки)
        if 'hour' in data.columns:
            features['hour'] = data['hour']
        if 'day_of_week' in data.columns:
            features['day_of_week'] = data['day_of_week']
        if 'is_weekend' in data.columns:
            features['is_weekend'] = data['is_weekend']
        if 'is_night' in data.columns:
            features['is_night'] = data['is_night']
        
        # Частотные признаки по IP-адресам
        if 'source_ip' in data.columns:
            source_ip_counts = data['source_ip'].value_counts().to_dict()
            features['source_ip_freq'] = data['source_ip'].map(source_ip_counts)
            
            # Кодирование IP-адресов
            if 'source_ip' not in self.label_encoders:
                self.label_encoders['source_ip'] = LabelEncoder()
                features['source_ip_encoded'] = self.label_encoders['source_ip'].fit_transform(data['source_ip'])
            else:
                # Обработка новых IP-адресов
                known_classes = set(self.label_encoders['source_ip'].classes_)
                new_ips = data['source_ip'].apply(lambda x: x if x in known_classes else 'unknown')
                features['source_ip_encoded'] = self.label_encoders['source_ip'].transform(new_ips)
        
        if 'destination_ip' in data.columns:
            dest_ip_counts = data['destination_ip'].value_counts().to_dict()
            features['destination_ip_freq'] = data['destination_ip'].map(dest_ip_counts)
            
            if 'destination_ip' not in self.label_encoders:
                self.label_encoders['destination_ip'] = LabelEncoder()
                features['destination_ip_encoded'] = self.label_encoders['destination_ip'].fit_transform(data['destination_ip'])
            else:
                known_classes = set(self.label_encoders['destination_ip'].classes_)
                new_ips = data['destination_ip'].apply(lambda x: x if x in known_classes else 'unknown')
                features['destination_ip_encoded'] = self.label_encoders['destination_ip'].transform(new_ips)
        
        # Частотные признаки по пользователям
        if 'user' in data.columns:
            user_counts = data['user'].value_counts().to_dict()
            features['user_freq'] = data['user'].map(user_counts)
            
            if 'user' not in self.label_encoders:
                self.label_encoders['user'] = LabelEncoder()
                features['user_encoded'] = self.label_encoders['user'].fit_transform(data['user'])
            else:
                known_classes = set(self.label_encoders['user'].classes_)
                new_users = data['user'].apply(lambda x: x if x in known_classes else 'unknown')
                features['user_encoded'] = self.label_encoders['user'].transform(new_users)
        
        # Частотные признаки по типам событий
        if 'event_type' in data.columns:
            event_counts = data['event_type'].value_counts().to_dict()
            features['event_type_freq'] = data['event_type'].map(event_counts)
            
            if 'event_type' not in self.label_encoders:
                self.label_encoders['event_type'] = LabelEncoder()
                features['event_type_encoded'] = self.label_encoders['event_type'].fit_transform(data['event_type'])
            else:
                known_classes = set(self.label_encoders['event_type'].classes_)
                new_events = data['event_type'].apply(lambda x: x if x in known_classes else 'unknown')
                features['event_type_encoded'] = self.label_encoders['event_type'].transform(new_events)
        
        # Статус события
        if 'status_success' in data.columns:
            features['status_success'] = data['status_success']
        if 'status_failed' in data.columns:
            features['status_failed'] = data['status_failed']
        
        # Числовые признаки
        if 'port' in data.columns:
            features['port'] = data['port']
        if 'bytes' in data.columns:
            features['bytes'] = data['bytes']
            # Логарифмическое преобразование для bytes (если есть большие значения)
            if features['bytes'].max() > 1000:
                features['bytes_log'] = np.log1p(features['bytes'])
        
        # Статистические признаки по временным окнам
        if 'timestamp' in data.columns and len(data) > 1:
            data_sorted = data.sort_values('timestamp')
            # Количество событий за последний час (скользящее окно)
            if len(data_sorted) > 10:
                features['events_last_hour'] = self._calculate_rolling_count(data_sorted, window='1h')
                # Среднее количество событий за час
                features['avg_events_per_hour'] = features['events_last_hour'].mean()
            else:
                features['events_last_hour'] = 1
                features['avg_events_per_hour'] = 1
        
        # Признаки аномальности частоты
        for col in ['source_ip_freq', 'destination_ip_freq', 'user_freq', 'event_type_freq']:
            if col in features.columns:
                mean_freq = features[col].mean()
                std_freq = features[col].std()
                if std_freq > 0:
                    features[f'{col}_zscore'] = (features[col] - mean_freq) / std_freq
        
        # Заполнение пропусков
        features = features.fillna(0)
        
        # Сохранение названий признаков
        self.feature_names = features.columns.tolist()
        
        return features
    
    def _calculate_rolling_count(self, data, window='1h'):
        """Вычисление скользящего количества событий."""
        try:
            data_indexed = data.set_index('timestamp')
            if 'source_ip' not in data_indexed.columns:
                return np.ones(len(data))
            
            # Используем count() на первой доступной колонке вместо size()
            # Это работает с RollingGroupby
            if len(data_indexed.columns) > 0:
                first_col = data_indexed.columns[0]
                rolling = data_indexed.groupby('source_ip').rolling(window=window, min_periods=1)[first_col].count()
            else:
                # Если нет колонок, создаём простой счётчик
                rolling = pd.Series(1, index=data_indexed.index)
            
            rolling = rolling.reset_index(level=0, drop=True)
            rolling = rolling.reindex(data_indexed.index).fillna(1)
            return rolling.values
        except Exception as e:
            # В случае ошибки возвращаем единицы
            return np.ones(len(data))
    
    def scale_features(self, features, fit=True):
        """
        Масштабирование признаков.
        
        Args:
            features: DataFrame с признаками
            fit: если True, обучает scaler, иначе использует обученный
            
        Returns:
            Масштабированные признаки
        """
        if fit:
            scaled = self.scaler.fit_transform(features)
        else:
            scaled = self.scaler.transform(features)
        
        return pd.DataFrame(scaled, columns=features.columns, index=features.index)
    
    def get_feature_names(self):
        """Возвращает список названий признаков."""
        return self.feature_names



