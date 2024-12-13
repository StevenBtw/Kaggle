import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

class ModelLogger:
    def __init__(self):
        self.current_run = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models': {},
            'ensemble_score': None
        }
        self.log_dir = 'insurance/logs'
        os.makedirs(self.log_dir, exist_ok=True)
    
    def log_model_score(self, model_name, score, fold_scores, feature_importances=None):
        """Log model performance metrics."""
        self.current_run['models'][model_name] = {
            'score': float(score),
            'fold_scores': [float(s) for s in fold_scores],
            'fold_std': float(np.std(fold_scores)),
            'feature_importances': feature_importances.tolist() if feature_importances is not None else None
        }
    
    def log_model_weight(self, model_name, weight):
        """Log model weight in ensemble."""
        if model_name in self.current_run['models']:
            self.current_run['models'][model_name]['weight'] = float(weight)
    
    def log_ensemble_score(self, score):
        """Log ensemble model score."""
        self.current_run['ensemble_score'] = float(score)
    
    def save_run(self):
        """Save current run to log file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'run_{timestamp}.json')
        
        with open(log_file, 'w') as f:
            json.dump(self.current_run, f, indent=2)
        
        return log_file
    
    def compare_runs(self, n_runs=5):
        """Compare current run with previous runs."""
        # Get list of log files
        log_files = sorted([
            f for f in os.listdir(self.log_dir) 
            if f.startswith('run_') and f.endswith('.json')
        ], reverse=True)
        
        # Load recent runs
        runs = []
        for log_file in log_files[:n_runs]:
            with open(os.path.join(self.log_dir, log_file), 'r') as f:
                run = json.load(f)
                runs.append({
                    'timestamp': run['timestamp'],
                    'ensemble_score': run['ensemble_score'],
                    **{f"{name}_score": info['score']
                       for name, info in run['models'].items()}
                })
        
        return pd.DataFrame(runs)
