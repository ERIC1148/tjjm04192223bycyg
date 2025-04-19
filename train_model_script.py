import os
import sys
import pandas as pd
import logging
import numpy as np
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # 1. Create necessary directories
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/models", exist_ok=True)
        os.makedirs("data/results", exist_ok=True)
        
        logger.info("Starting model training process...")
        
        # 2. Load/create location data for location model
        location_data_path = "data/processed/location_features.csv"
        if not os.path.exists(location_data_path):
            logger.info("Creating mock location data...")
            
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
            logger.info(f"Mock location data created and saved to {location_data_path}")
        
        # 3. Train location selection model
        logger.info("Training location selection model...")
        
        # Import models module
        from src.models.location_selection_model import LocationSelectionModel
        
        # Load the location data
        location_df = pd.read_csv(location_data_path)
        logger.info(f"Loaded location data, shape: {location_df.shape}")
        
        # Prepare features and target
        target_col = 'historical_score'
        feature_cols = [col for col in location_df.columns 
                        if col not in [target_col, 'location_id']]
        
        X = location_df[feature_cols].values
        y = location_df[target_col].values
        
        # Create and train the model
        model = LocationSelectionModel(model_type='xgboost')
        model.fit(X, y, feature_names=feature_cols)
        
        # Save the model
        location_model_path = "data/models/location_model.pkl"
        model.save_model(location_model_path)
        logger.info(f"Location selection model trained and saved to {location_model_path}")
        
        # 4. Create spatial-temporal model if hotel features exist
        features_path = "data/processed/hotel_features.csv"
        if os.path.exists(features_path):
            logger.info("Training spatial-temporal model...")
            
            # Import required modules
            from src.models.spatial_temporal_model import (
                SpatialTemporalModel, 
                SpatialTemporalTrainer, 
                SpatialTemporalDataset,
                prepare_sequence_data
            )
            from sklearn.model_selection import train_test_split
            
            # Load hotel features
            features_df = pd.read_csv(features_path)
            logger.info(f"Loaded hotel features, shape: {features_df.shape}")
            
            # Define feature and target columns
            st_feature_cols = [
                'lead_time', 'arrival_date_month_num', 'stays_in_weekend_nights', 
                'stays_in_week_nights', 'adults', 'children', 'is_repeated_guest', 
                'previous_cancellations', 'previous_bookings_not_canceled', 
                'booking_changes', 'required_car_parking_spaces', 'total_of_special_requests'
            ]
            st_target_cols = ['adr']  # Average Daily Rate
            
            # Filter out non-existing columns
            st_feature_cols = [col for col in st_feature_cols if col in features_df.columns]
            st_target_cols = [col for col in st_target_cols if col in features_df.columns]
            
            if not st_feature_cols or not st_target_cols:
                logger.warning("Not enough features or targets for spatial-temporal model")
            else:
                # Prepare sequence data
                seq_len = 7
                X, y, _ = prepare_sequence_data(
                    features_df, st_feature_cols, st_target_cols, seq_len=seq_len
                )
                logger.info(f"Sequence data prepared, shapes: X={X.shape}, y={y.shape}")
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Create datasets and dataloaders
                train_dataset = SpatialTemporalDataset(X_train, y_train)
                test_dataset = SpatialTemporalDataset(X_test, y_test)
                
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=32, shuffle=True
                )
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=32, shuffle=False
                )
                
                # Create and train model
                input_dim = X_train.shape[2]  # Feature dimension
                hidden_dim = 64
                output_dim = y_train.shape[1]  # Target dimension
                num_nodes = 100
                
                model = SpatialTemporalModel(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    num_nodes=num_nodes,
                    seq_len=seq_len
                )
                
                trainer = SpatialTemporalTrainer(model, learning_rate=0.001)
                
                # Train model
                st_model_path = "data/models/spatial_temporal_model.pt"
                history = trainer.train(
                    train_loader, test_loader, 
                    epochs=20, 
                    patience=5,
                    model_path=st_model_path
                )
                
                logger.info(f"Spatial-temporal model trained and saved to {st_model_path}")
        else:
            logger.warning(f"Hotel features file not found at {features_path}, skipping spatial-temporal model")
        
        logger.info("Model training completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 