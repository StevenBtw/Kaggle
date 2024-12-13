import lightgbm as lgb

def create_lgbm_models(n_gpu_threads):
    """Create LightGBM models with different configurations."""
    models = {
        'lgb1': lgb.LGBMRegressor(
            n_estimators=6000,
            learning_rate=0.002,
            num_leaves=31,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            n_jobs=n_gpu_threads,
            verbose=-1
        ),
        'lgb2': lgb.LGBMRegressor(
            n_estimators=6000,
            learning_rate=0.002,
            num_leaves=127,
            max_depth=12,
            min_child_samples=10,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=43,
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            n_jobs=n_gpu_threads,
            verbose=-1
        ),
        'lgb3': lgb.LGBMRegressor(
            n_estimators=6000,
            learning_rate=0.002,
            num_leaves=63,
            max_depth=10,
            min_child_samples=15,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.15,
            reg_lambda=0.15,
            random_state=44,
            device='gpu',
            gpu_platform_id=0,
            gpu_device_id=0,
            n_jobs=n_gpu_threads,
            verbose=-1
        )
    }
    return models

def create_meta_model():
    """Create meta model for stacking."""
    return lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.005,
        num_leaves=31,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        device='gpu'
    )
