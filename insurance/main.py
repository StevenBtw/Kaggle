import pandas as pd
import os
from datetime import datetime
import json
from insurance.common.utils import rmsle
from insurance.common.feature_engineering import engineer_features
from insurance.common.training import train_and_predict

def get_next_run_number():
    """Get the next run number by checking existing run files."""
    log_dir = 'insurance/logs'
    os.makedirs(log_dir, exist_ok=True)
    existing_runs = [f for f in os.listdir(log_dir) if f.startswith('run') and f.endswith('.json')]
    if not existing_runs:
        return 1
    run_numbers = [int(f[3:-5]) for f in existing_runs]  # Extract numbers from 'run{n}.json'
    return max(run_numbers) + 1

def log_step(run_log, step_name):
    """Log a step with its timestamp."""
    run_log['steps'].append({
        'step': step_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

def main():
    """Main training pipeline."""
    # Initialize run log
    run_number = get_next_run_number()
    run_log = {
        'run_number': run_number,
        'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'steps': []
    }
    
    # Set number of threads/jobs
    n_jobs = os.cpu_count() - 1  # Leave one CPU free
    n_gpu_threads = 4  # Adjust based on GPU memory
    
    log_step(run_log, "Loading data")
    print("Loading data...")
    train_data = pd.read_csv('insurance/data/train.csv')
    test_data = pd.read_csv('insurance/data/test.csv')
    
    print("\nData columns:")
    print("Train columns:", train_data.columns.values)
    print("Test columns:", test_data.columns.values)
    
    log_step(run_log, "Engineering features")
    print("\nEngineering features...")
    (X_nan, X_complete, y, test_nan, test_complete, 
     numerical_features_nan, numerical_features_complete, 
     categorical_features) = engineer_features(train_data, test_data)
    
    print("\nFeature sets created:")
    print(f"NaN features (for CatBoost, LightGBM, NGBoost): {X_nan.shape[1]} features")
    print(f"Complete features (for TabNet, clustering): {X_complete.shape[1]} features")
    
    log_step(run_log, "Training models")
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
    
    log_step(run_log, "Saving predictions")
    print("\nSaving predictions...")
    submission = pd.DataFrame({
        'id': test_data['id'],
        'target': test_predictions['ensemble']
    })
    submission.to_csv('insurance/submission.csv', index=False)
    print("Predictions saved to submission.csv")
    
    # Print final validation scores
    log_step(run_log, "Computing final scores")
    print("\nFinal Validation Scores:")
    scores = {}
    for model_name, predictions in oof_predictions.items():
        score = rmsle(y, predictions)
        scores[model_name] = float(score)  # Convert numpy float to Python float for JSON
        print(f"{model_name}: {score:.4f}")
    
    # Add scores to run log
    run_log['scores'] = scores
    run_log['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save run log
    log_dir = 'insurance/logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = f'{log_dir}/run{run_number}.json'
    with open(log_path, 'w') as f:
        json.dump(run_log, f, indent=2)
    print(f"\nRun log saved to: {log_path}")

if __name__ == '__main__':
    main()
