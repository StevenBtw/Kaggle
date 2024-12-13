import xgboost as xgb

# Model version - increment when changing model architecture or hyperparameters
MODEL_VERSION = "1.0"

def create_xgb_models(n_gpu_threads=4):
    """Create XGBoost models with different hyperparameters."""
    models = []
    
    # Base model
    models.append(xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        tree_method='gpu_hist',  # Use GPU
        gpu_id=0,
        max_bin=256,  # Optimize for GPU memory
        early_stopping_rounds=50,
        verbosity=1
    ))
    
    # Deeper model
    models.append(xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        min_child_weight=4,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=43,
        tree_method='gpu_hist',  # Use GPU
        gpu_id=0,
        max_bin=256,  # Optimize for GPU memory
        early_stopping_rounds=50,
        verbosity=1
    ))
    
    # More regularized model
    models.append(xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        min_child_weight=5,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=44,
        tree_method='gpu_hist',  # Use GPU
        gpu_id=0,
        max_bin=256,  # Optimize for GPU memory
        early_stopping_rounds=50,
        verbosity=1
    ))
    
    return models
