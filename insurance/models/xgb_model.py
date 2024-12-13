import xgboost as xgb

def create_xgb_models(n_gpu_threads):
    """Create XGBoost models with different configurations."""
    models = {
        'xgb1': xgb.XGBRegressor(
            n_estimators=6000,
            learning_rate=0.002,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=0,
            n_jobs=n_gpu_threads
        ),
        'xgb2': xgb.XGBRegressor(
            n_estimators=6000,
            learning_rate=0.002,
            max_depth=10,
            min_child_weight=2,
            subsample=0.7,
            colsample_bytree=0.7,
            gamma=0.2,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=43,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=0,
            n_jobs=n_gpu_threads
        ),
        'xgb3': xgb.XGBRegressor(
            n_estimators=6000,
            learning_rate=0.002,
            max_depth=9,
            min_child_weight=1.5,
            subsample=0.75,
            colsample_bytree=0.75,
            gamma=0.15,
            reg_alpha=0.15,
            reg_lambda=0.15,
            random_state=44,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=0,
            n_jobs=n_gpu_threads
        )
    }
    return models
