import lightgbm as lgb

# Model version - increment when changing model architecture or hyperparameters
MODEL_VERSION = "1.0"

def create_lgbm_models(n_jobs=-1):
    """Create LightGBM models with different hyperparameters."""
    models = []
    
    # Base model
    models.append(lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=n_jobs,
        device='gpu'
    ))
    
    # Deeper model
    models.append(lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=63,
        max_depth=8,
        min_child_samples=30,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=43,
        n_jobs=n_jobs,
        device='gpu'
    ))
    
    # More regularized model
    models.append(lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=40,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=44,
        n_jobs=n_jobs,
        device='gpu'
    ))
    
    return models
