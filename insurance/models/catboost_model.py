from catboost import CatBoostRegressor

def create_catboost_models(n_gpu_threads=4):
    """Create CatBoost models with different hyperparameters."""
    models = []
    
    # Base model
    models.append(CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=42,
        task_type='GPU',
        devices='0:' + str(n_gpu_threads),
        early_stopping_rounds=50,
        verbose=100
    ))
    
    # Deeper model
    models.append(CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=5,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=43,
        task_type='GPU',
        devices='0:' + str(n_gpu_threads),
        early_stopping_rounds=50,
        verbose=100
    ))
    
    # More regularized model
    models.append(CatBoostRegressor(
        iterations=3000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=10,
        loss_function='RMSE',
        eval_metric='RMSE',
        random_seed=44,
        task_type='GPU',
        devices='0:' + str(n_gpu_threads),
        early_stopping_rounds=50,
        verbose=100
    ))
    
    return models
