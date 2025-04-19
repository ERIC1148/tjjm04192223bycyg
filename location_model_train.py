import os
import numpy as np
import pandas as pd
import pickle

print("Starting location model training")

# Create necessary directories
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/models", exist_ok=True)

# Create mock location data
location_data_path = "data/processed/location_features.csv"
if not os.path.exists(location_data_path):
    print("Creating mock location data...")
    n_locations = 100
    np.random.seed(42)
    
    location_data = {
        'location_id': [f'LOC_{i:03d}' for i in range(n_locations)],
        'latitude': np.random.uniform(31.1, 31.3, n_locations),
        'longitude': np.random.uniform(121.4, 121.6, n_locations),
        'poi_restaurant_count': np.random.randint(5, 50, n_locations),
        'poi_shopping_count': np.random.randint(3, 30, n_locations),
        'poi_entertainment_count': np.random.randint(2, 20, n_locations),
        'poi_transport_count': np.random.randint(1, 15, n_locations),
        'distance_to_city_center': np.random.uniform(0.5, 15.0, n_locations),
        'distance_to_airport': np.random.uniform(5.0, 50.0, n_locations),
        'distance_to_subway': np.random.uniform(0.1, 3.0, n_locations),
        'population_density': np.random.uniform(5000, 25000, n_locations),
        'income_per_capita': np.random.uniform(80000, 150000, n_locations),
        'unemployment_rate': np.random.uniform(2.0, 8.0, n_locations),
        'competitor_count': np.random.randint(0, 10, n_locations),
        'area_km2': np.random.uniform(0.5, 5.0, n_locations),
        'historical_score': np.random.uniform(20, 95, n_locations)
    }
    
    pd.DataFrame(location_data).to_csv(location_data_path, index=False)
    print(f"Mock location data created at {location_data_path}")
else:
    print(f"Using existing location data at {location_data_path}")

# Define a simple XGBoost model class
class SimpleLocationModel:
    def __init__(self):
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
        except ImportError:
            from sklearn.ensemble import RandomForestRegressor
            print("XGBoost not available, using RandomForest instead")
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {path}")
        return path

# Load the location data
location_df = pd.read_csv(location_data_path)
print(f"Loaded location data: {location_df.shape}")

# Prepare features and target
target_col = 'historical_score'
feature_cols = [col for col in location_df.columns 
                if col not in [target_col, 'location_id']]

X = location_df[feature_cols].values
y = location_df[target_col].values

# Train the model
print("Training model...")
model = SimpleLocationModel()
model.fit(X, y)

# Save the model
model_path = "data/models/simple_location_model.pkl"
model.save(model_path)

print("Training complete!") 