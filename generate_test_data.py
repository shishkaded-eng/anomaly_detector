"""
Генерация синтетических данных логов безопасности с известными аномалиями для тестирования.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random


def generate_normal_events(n_events=1000, start_date=None):
    """
    Генерация нормальных событий.
    
    Args:
        n_events: количество событий
        start_date: начальная дата (если None, используется текущая дата)
        
    Returns:
        DataFrame с нормальными событиями
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=7)
    
    events = []
    
    # Нормальные IP-адреса
    normal_ips = [f"192.168.1.{i}" for i in range(10, 50)]
    normal_users = ['user1', 'user2', 'user3', 'admin', 'guest']
    normal_event_types = ['login', 'logout', 'access', 'read', 'write']
    
    for i in range(n_events):
        # Время события (рабочие часы, будние дни)
        hours = random.choice(range(9, 18))
        minutes = random.randint(0, 59)
        days_offset = random.randint(0, 5)  # Будние дни
        timestamp = start_date + timedelta(days=days_offset, hours=hours, minutes=minutes)
        
        event = {
            'timestamp': timestamp,
            'source_ip': random.choice(normal_ips),
            'destination_ip': random.choice(normal_ips),
            'event_type': random.choice(normal_event_types),
            'user': random.choice(normal_users),
            'status': random.choice(['success', 'success', 'success', 'failed']),  # 75% успешных
            'port': random.choice([80, 443, 22, 21, 3306]),
            'bytes': random.randint(100, 10000)
        }
        events.append(event)
    
    return pd.DataFrame(events)


def generate_bruteforce_attack(n_events=50, start_date=None):
    """
    Генерация брутфорс атаки (множественные неудачные попытки входа).
    
    Args:
        n_events: количество событий атаки
        start_date: начальная дата
        
    Returns:
        DataFrame с событиями атаки
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=3)
    
    events = []
    
    # Атакующий IP
    attacker_ip = "10.0.0.100"
    target_ip = "192.168.1.1"
    
    # Атака происходит в короткий промежуток времени
    attack_start = start_date + timedelta(hours=2)
    
    for i in range(n_events):
        # Множественные попытки в короткое время
        timestamp = attack_start + timedelta(seconds=i*2)
        
        event = {
            'timestamp': timestamp,
            'source_ip': attacker_ip,
            'destination_ip': target_ip,
            'event_type': 'login',
            'user': random.choice(['admin', 'root', 'user1', 'test']),
            'status': 'failed',  # Все попытки неудачные
            'port': 22,
            'bytes': random.randint(50, 200)
        }
        events.append(event)
    
    return pd.DataFrame(events)


def generate_ddos_attack(n_events=200, start_date=None):
    """
    Генерация DDoS атаки (аномально высокий трафик).
    
    Args:
        n_events: количество событий атаки
        start_date: начальная дата
        
    Returns:
        DataFrame с событиями атаки
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=2)
    
    events = []
    
    # Множество атакующих IP
    attacker_ips = [f"172.16.0.{i}" for i in range(1, 20)]
    target_ip = "192.168.1.1"
    
    # Атака происходит в короткий промежуток времени
    attack_start = start_date + timedelta(hours=14)
    
    for i in range(n_events):
        # Очень быстрая последовательность запросов
        timestamp = attack_start + timedelta(seconds=i*0.1)
        
        event = {
            'timestamp': timestamp,
            'source_ip': random.choice(attacker_ips),
            'destination_ip': target_ip,
            'event_type': 'access',
            'user': 'anonymous',
            'status': 'success',
            'port': 80,
            'bytes': random.randint(1000, 50000)  # Большой объём данных
        }
        events.append(event)
    
    return pd.DataFrame(events)


def generate_suspicious_activity(n_events=30, start_date=None):
    """
    Генерация подозрительной активности (необычные паттерны).
    
    Args:
        n_events: количество событий
        start_date: начальная дата
        
    Returns:
        DataFrame с подозрительными событиями
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=1)
    
    events = []
    
    # Подозрительный IP
    suspicious_ip = "203.0.113.50"
    
    # Активность в нерабочее время (ночь)
    attack_start = start_date + timedelta(hours=23)
    
    for i in range(n_events):
        # События в ночное время
        timestamp = attack_start + timedelta(hours=i*0.5)
        
        event = {
            'timestamp': timestamp,
            'source_ip': suspicious_ip,
            'destination_ip': random.choice([f"192.168.1.{i}" for i in range(1, 10)]),
            'event_type': random.choice(['write', 'delete', 'modify']),  # Подозрительные операции
            'user': 'unknown',
            'status': 'success',
            'port': random.choice([8080, 4444, 9999]),  # Нестандартные порты
            'bytes': random.randint(5000, 50000)
        }
        events.append(event)
    
    return pd.DataFrame(events)


def generate_test_dataset(output_file='data/test_logs.csv', n_normal=1000):
    """
    Генерация полного тестового набора данных с аномалиями.
    
    Args:
        output_file: путь к выходному файлу
        n_normal: количество нормальных событий
    """
    print("Генерация тестовых данных...")
    
    # Генерация нормальных событий
    print(f"  - Генерация {n_normal} нормальных событий...")
    normal_events = generate_normal_events(n_normal)
    
    # Генерация аномалий
    print("  - Генерация брутфорс атаки (50 событий)...")
    bruteforce = generate_bruteforce_attack(50)
    
    print("  - Генерация DDoS атаки (200 событий)...")
    ddos = generate_ddos_attack(200)
    
    print("  - Генерация подозрительной активности (30 событий)...")
    suspicious = generate_suspicious_activity(30)
    
    # Объединение всех событий
    all_events = pd.concat([normal_events, bruteforce, ddos, suspicious], ignore_index=True)
    
    # Сортировка по времени
    all_events = all_events.sort_values('timestamp').reset_index(drop=True)
    
    # Добавление метки аномалии для оценки
    anomaly_indices = set(range(len(normal_events), len(all_events)))
    all_events['is_anomaly'] = all_events.index.isin(anomaly_indices).astype(int)
    
    # Сохранение
    all_events.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\nТестовые данные сохранены: {output_file}")
    print(f"  - Всего событий: {len(all_events)}")
    print(f"  - Нормальных: {len(normal_events)}")
    print(f"  - Аномалий: {len(anomaly_indices)}")
    print(f"  - Процент аномалий: {len(anomaly_indices)/len(all_events)*100:.2f}%")
    
    return all_events


if __name__ == '__main__':
    import os
    os.makedirs('data', exist_ok=True)
    generate_test_dataset('data/test_logs.csv', n_normal=1000)

