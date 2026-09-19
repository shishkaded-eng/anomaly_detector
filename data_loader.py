"""
Модуль загрузки и предобработки логов безопасности.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os


class LogDataLoader:
    """Класс для загрузки и предобработки логов безопасности."""
    
    def __init__(self):
        self.data = None
        
    def load_csv(self, file_path):
        """
        Загрузка логов из CSV файла.
        
        Args:
            file_path: путь к CSV файлу
            
        Returns:
            DataFrame с логами
        """
        try:
            self.data = pd.read_csv(file_path)
            print(f"Загружено {len(self.data)} записей из {file_path}")
            return self.data
        except Exception as e:
            print(f"Ошибка при загрузке CSV: {e}")
            return None
    
    def load_json(self, file_path):
        """
        Загрузка логов из JSON файла.
        
        Args:
            file_path: путь к JSON файлу
            
        Returns:
            DataFrame с логами
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.data = pd.DataFrame(data)
            print(f"Загружено {len(self.data)} записей из {file_path}")
            return self.data
        except Exception as e:
            print(f"Ошибка при загрузке JSON: {e}")
            return None
    
    def preprocess(self, data=None):
        """
        Предобработка данных: очистка, нормализация, обработка пропусков.
        
        Args:
            data: DataFrame для обработки (если None, используется self.data)
            
        Returns:
            Обработанный DataFrame
        """
        if data is None:
            data = self.data.copy()
        else:
            data = data.copy()
        
        # Обработка timestamp
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            data['is_night'] = ((data['hour'] >= 22) | (data['hour'] < 6)).astype(int)
        
        # Обработка пропусков
        # Для числовых колонок заполняем медианой
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isna().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Для категориальных колонок заполняем модой
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isna().sum() > 0:
                mode_value = data[col].mode()[0] if len(data[col].mode()) > 0 else 'unknown'
                data[col].fillna(mode_value, inplace=True)
        
        # Нормализация статуса
        if 'status' in data.columns:
            data['status'] = data['status'].str.lower()
            data['status_success'] = (data['status'] == 'success').astype(int)
            data['status_failed'] = (data['status'] == 'failed').astype(int)
        
        print(f"Предобработка завершена. Обработано {len(data)} записей")
        return data
    
    def get_data(self):
        """Возвращает загруженные данные."""
        return self.data

