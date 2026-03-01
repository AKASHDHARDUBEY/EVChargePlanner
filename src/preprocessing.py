import pandas as pd
import numpy as np


def load_volume_data(file_path):
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    return df


def get_total_volume(df):
    zone_cols = [col for col in df.columns if col != 'time']
    df_processed = df.copy()
    df_processed['total_volume'] = df_processed[zone_cols].sum(axis=1)
    return df_processed


def add_time_features(df):
    df_features = df.copy()
    df_features['hour'] = df_features['time'].dt.hour
    df_features['day_of_week'] = df_features['time'].dt.dayofweek
    df_features['day'] = df_features['time'].dt.day
    df_features['month'] = df_features['time'].dt.month
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)

    # sin/cos encoding so model knows hour 23 and 0 are close
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    return df_features


def add_lag_features(df, target_col='total_volume', lags=[1, 2, 3, 6, 12, 24]):
    df_lag = df.copy()
    for lag in lags:
        df_lag[f'lag_{lag}'] = df_lag[target_col].shift(lag)
    return df_lag


def add_rolling_features(df, target_col='total_volume', windows=[6, 12, 24]):
    df_roll = df.copy()
    for window in windows:
        df_roll[f'rolling_mean_{window}'] = df_roll[target_col].rolling(window=window).mean()
        df_roll[f'rolling_std_{window}'] = df_roll[target_col].rolling(window=window).std()
    return df_roll


def prepare_features(df):
    df_prep = get_total_volume(df)
    df_prep = add_time_features(df_prep)
    df_prep = add_lag_features(df_prep)
    df_prep = add_rolling_features(df_prep)
    df_prep = df_prep.dropna()
    return df_prep


def get_feature_columns():
    return [
        'hour', 'day_of_week', 'day', 'month', 'is_weekend',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12', 'lag_24',
        'rolling_mean_6', 'rolling_mean_12', 'rolling_mean_24',
        'rolling_std_6', 'rolling_std_12', 'rolling_std_24'
    ]


def split_data(df, train_ratio=0.8):
    train_size = int(len(df) * train_ratio)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    return train_df, test_df


def get_peak_hours(df, top_n=5):
    hourly_avg = df.groupby('hour')['total_volume'].mean()
    return hourly_avg.nlargest(top_n)


def get_peak_days(df, top_n=3):
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = df.groupby('day_of_week')['total_volume'].mean()
    daily_avg.index = daily_avg.index.map(lambda x: day_names[x])
    return daily_avg.nlargest(top_n)


def get_summary_stats(df):
    return {
        'total_records': len(df),
        'date_range_start': df['time'].min().strftime('%Y-%m-%d'),
        'date_range_end': df['time'].max().strftime('%Y-%m-%d'),
        'avg_volume': df['total_volume'].mean(),
        'max_volume': df['total_volume'].max(),
        'min_volume': df['total_volume'].min(),
        'std_volume': df['total_volume'].std()
    }
