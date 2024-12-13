import pandas as pd
import os
from insurance.common.utils import rmsle
from insurance.common.feature_engineering import engineer_features
from insurance.common.training import train_and_predict

def main():
    """Main training pipeline."""
    # Set number of threads/jobs
    n_jobs = os.cpu_count() - 1  # Leave one CPU free
    n_gpu_threads = 4  # Adjust based on GPU memory
    
    print("Loading data...")
    train_data = pd.read_csv('insurance/data/train.csv')
    test_data = pd.read_csv('insurance/data/test.csv')
    
    print("\nData columns:")
    print("Train columns:", train_data.columns.tolist())
    print("Test columns:", test_data.columns.tolist())
    
    print("\nEngineering features...")
    (X_nan, X_complete, y, test_nan, test_complete, 
     numerical_features_nan, numerical_features_complete, 
     categorical_features) = engineer_features(train_data, test_data)
    
    print("\nFeature sets created:")
    print(f"NaN features (for CatBoost, LightGBM, NGBoost): {X_nan.shape[1]} features")
    print(f"Complete features (for TabNet, clustering): {X_complete.shape[1]} features")
    
    print("\nTraining models and generating predictions...")
    oof_predictions, test_predictions = train_and_predict(
        X_nan=X_nan,
        X_complete=X_complete,
        y=y,
        test_nan=test_nan,
        test_complete=test_complete,
        numerical_features_nan=numerical_features_nan,
        numerical_features_complete=numerical_features_complete,
        categorical_features=categorical_features,
        n_jobs=n_jobs,
        n_gpu_threads=n_gpu_threads
    )
    
    # Save predictions
    print("\nSaving predictions...")
    submission = pd.DataFrame({
        'id': test_data['id'],
        'target': test_predictions['ensemble']
    })
    submission.to_csv('insurance/submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    # Print final validation scores
    print("\nFinal Validation Scores:")
    for model_name, predictions in oof_predictions.items():
        score = rmsle(y, predictions)
        print(f"{model_name}: {score:.4f}")

if __name__ == '__main__':
    main()
