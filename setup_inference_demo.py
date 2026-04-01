import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_and_save_model():
    # Load the mock January data we created earlier
    if not os.path.exists("taxi_jan.csv"):
        print("Error: taxi_jan.csv not found. Please run setup_mock_data.py first.")
        return

    df = pd.read_csv("taxi_jan.csv")
    
    # Feature Engineering (mimicking the requirements)
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['pickup_cluster'] = df['PULocationID'] % 10  # Mock clusters
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 10 or 17 <= x <= 20) else 0)
    
    # Mock lag features
    df['lag_1'] = df['total_amount'].shift(1).fillna(df['total_amount'].mean())
    df['lag_24'] = df['total_amount'].shift(24).fillna(df['total_amount'].mean())
    
    # Define features and target
    FEATURES = ["hour", "day_of_week", "pickup_cluster", "is_peak_hour", "lag_1", "lag_24"]
    TARGET = "total_amount"
    
    X = df[FEATURES]
    y = df[TARGET]
    
    # Train model
    print("Training Random Forest model for inference demo...")
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    # Save model
    joblib.dump(model, "best_model.pkl")
    print("Model saved as best_model.pkl")
    
    # Save a processed version of taxi_feb.csv to simulate "march_2016_with_lags.csv"
    if os.path.exists("taxi_feb.csv"):
        feb_df = pd.read_csv("taxi_feb.csv")
        feb_df['tpep_pickup_datetime'] = pd.to_datetime(feb_df['tpep_pickup_datetime'])
        feb_df['hour'] = feb_df['tpep_pickup_datetime'].dt.hour
        feb_df['day_of_week'] = feb_df['tpep_pickup_datetime'].dt.dayofweek
        feb_df['pickup_cluster'] = feb_df['PULocationID'] % 10
        feb_df['is_peak_hour'] = feb_df['hour'].apply(lambda x: 1 if (8 <= x <= 10 or 17 <= x <= 20) else 0)
        feb_df['lag_1'] = feb_df['total_amount'].shift(1).fillna(feb_df['total_amount'].mean())
        feb_df['lag_24'] = feb_df['total_amount'].shift(24).fillna(feb_df['total_amount'].mean())
        
        # Save as the expected filename in the user request
        feb_df.to_csv("march_2016_with_lags.csv", index=False)
        print("Saved march_2016_with_lags.csv (mocked from February data)")

if __name__ == "__main__":
    train_and_save_model()
