import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# 创建目录
os.makedirs("data/raw/hotel-booking-demand", exist_ok=True)

# 创建随机数据的函数
def generate_mock_hotel_data(num_samples=5000):
    # 创建日期范围
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    days = (end_date - start_date).days
    
    # 酒店类型
    hotel_types = ['Resort Hotel', 'City Hotel']
    
    # 创建数据
    data = {
        'hotel': np.random.choice(hotel_types, num_samples),
        'is_canceled': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'lead_time': np.random.randint(0, 365, num_samples),
        'arrival_date_year': np.random.choice([2022, 2023], num_samples),
        'arrival_date_month': np.random.choice(['January', 'February', 'March', 'April', 'May', 'June', 
                                              'July', 'August', 'September', 'October', 'November', 'December'], num_samples),
        'arrival_date_month_num': np.random.randint(1, 13, num_samples),
        'arrival_date_week_number': np.random.randint(1, 53, num_samples),
        'arrival_date_day_of_month': np.random.randint(1, 31, num_samples),
        'stays_in_weekend_nights': np.random.randint(0, 5, num_samples),
        'stays_in_week_nights': np.random.randint(0, 15, num_samples),
        'adults': np.random.randint(1, 5, num_samples),
        'children': np.random.randint(0, 4, num_samples),
        'babies': np.random.randint(0, 2, num_samples),
        'meal': np.random.choice(['BB', 'HB', 'FB', 'SC'], num_samples),
        'country': np.random.choice(['PRT', 'GBR', 'USA', 'ESP', 'ITA', 'FRA', 'DEU', 'CHN'], num_samples),
        'market_segment': np.random.choice(['Direct', 'Corporate', 'Online TA', 'Offline TA', 'Groups'], num_samples),
        'distribution_channel': np.random.choice(['Direct', 'Corporate', 'TA/TO'], num_samples),
        'is_repeated_guest': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
        'previous_cancellations': np.random.randint(0, 3, num_samples),
        'previous_bookings_not_canceled': np.random.randint(0, 5, num_samples),
        'reserved_room_type': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], num_samples),
        'assigned_room_type': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], num_samples),
        'booking_changes': np.random.randint(0, 5, num_samples),
        'deposit_type': np.random.choice(['No Deposit', 'Refundable', 'Non Refund'], num_samples, p=[0.8, 0.15, 0.05]),
        'agent': np.random.choice(list(range(1, 21)) + [None], num_samples),
        'company': np.random.choice(list(range(1, 11)) + [None], num_samples, p=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.9]),
        'days_in_waiting_list': np.random.randint(0, 30, num_samples),
        'customer_type': np.random.choice(['Transient', 'Contract', 'Transient-Party', 'Group'], num_samples, p=[0.7, 0.1, 0.15, 0.05]),
        'adr': np.random.uniform(50, 500, num_samples),  # 平均每日房价
        'required_car_parking_spaces': np.random.randint(0, 3, num_samples),
        'total_of_special_requests': np.random.randint(0, 5, num_samples),
        'reservation_status': np.random.choice(['Check-Out', 'Canceled', 'No-Show'], num_samples, p=[0.6, 0.35, 0.05]),
    }
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 处理一些相关性和约束
    df.loc[df['is_canceled'] == 1, 'reservation_status'] = 'Canceled'
    df.loc[(df['is_canceled'] == 0) & (np.random.random(num_samples) < 0.05), 'reservation_status'] = 'No-Show'
    df.loc[(df['is_canceled'] == 0) & (df['reservation_status'] != 'No-Show'), 'reservation_status'] = 'Check-Out'
    
    # 生成预订日期
    df['reservation_status_date'] = df.apply(
        lambda row: (start_date + timedelta(days=random.randint(0, days))).strftime('%Y-%m-%d'),
        axis=1
    )
    
    # 创建经纬度信息（用于POI数据集成）
    df['latitude'] = np.random.uniform(31.1, 31.3, num_samples)
    df['longitude'] = np.random.uniform(121.4, 121.6, num_samples)
    
    return df

# 生成数据
print("Generating mock hotel booking data...")
hotel_data = generate_mock_hotel_data(5000)

# 保存数据
output_path = "data/raw/hotel-booking-demand/hotel_bookings.csv"
hotel_data.to_csv(output_path, index=False)
print(f"Mock hotel booking data saved to: {output_path}")

print("\nData generation complete!") 