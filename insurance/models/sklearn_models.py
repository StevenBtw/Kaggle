from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor

def create_hist_gradient_models():
    """Create HistGradientBoosting models with different hyperparameters."""
    models = []
    
    # Base model
    models.append(HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=42,
        verbose=1
    ))
    
    # Deeper model
    models.append(HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=8,
        l2_regularization=2.0,
        early_stopping=True,
        random_state=43,
        verbose=1
    ))
    
    # More regularized model
    models.append(HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.03,
        max_depth=6,
        l2_regularization=5.0,
        early_stopping=True,
        random_state=44,
        verbose=1
    ))
    
    return models

def create_extra_trees_models(n_jobs=-1):
    """Create ExtraTrees models with different hyperparameters."""
    models = []
    
    # Base model
    models.append(ExtraTreesRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        max_features='sqrt',
        n_jobs=n_jobs,
        random_state=42,
        verbose=1
    ))
    
    # More trees, less depth
    models.append(ExtraTreesRegressor(
        n_estimators=800,
        max_depth=8,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=n_jobs,
        random_state=43,
        verbose=1
    ))
    
    # Deeper trees, more regularization
    models.append(ExtraTreesRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=8,
        min_samples_leaf=4,
        max_features='sqrt',
        n_jobs=n_jobs,
        random_state=44,
        verbose=1
    ))
    
    return models
