import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import joblib
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
import torch
from joblib import parallel_backend

from ..models import (
    create_lgbm_models,
    create_xgb_models,
    create_catboost_models,
    create_hist_gradient_models,
    create_extra_trees_models,
    create_tabnet_models,
    create_ngboost_models
)
from .preprocessing import create_preprocessor
from .utils import rmsle
from .logging import ModelLogger

def train_and_predict(
    X_nan,
    X_complete,
    y,
    test_nan,
    test_complete,
    numerical_features_nan,
    numerical_features_complete,
    categorical_features,
    n_jobs=-1,
    n_gpu_threads=4,
    n_splits=5,
    random_state=42
):
    """Train models and generate predictions using cross-validation."""
    
    # Initialize cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize logger
    logger = ModelLogger()
    
    # Create model dictionaries
    models = {
        'lgbm': create_lgbm_models(n_jobs),
        'catboost': create_catboost_models(n_gpu_threads),
        'ngboost': create_ngboost_models(),
        'hist_gradient': create_hist_gradient_models(),
        'extra_trees': create_extra_trees_models(n_jobs)
    }
    
    tabnet_models = {
        'tabnet': create_tabnet_models(n_gpu_threads)
    }
    
    # Initialize predictions dictionaries
    oof_predictions = {}
    test_predictions = {}
    meta_features = np.zeros((len(X_nan), len(models) + len(tabnet_models)))
    meta_test = np.zeros((len(test_nan), len(models) + len(tabnet_models)))
    model_weights = {}
    
    # Create preprocessors
    preprocessor_nan = create_preprocessor(numerical_features_nan, categorical_features, n_jobs)
    preprocessor_complete = create_preprocessor(numerical_features_complete, categorical_features, n_jobs)
    
    # Standard scaler for TabNet
    scaler = StandardScaler()
    
    with parallel_backend('threading', n_jobs=n_jobs):
        # Train models that can handle NaN values
        for i, (name, model) in enumerate(models.items()):
            if name in ['lgbm', 'catboost', 'ngboost']:
                X = X_nan
                test_features = test_nan
                preprocessor = preprocessor_nan
            else:
                X = X_complete
                test_features = test_complete
                preprocessor = preprocessor_complete
            
            print(f"\nTraining {name}...")
            oof_pred = np.zeros(len(X))
            test_pred = np.zeros(len(test_features))
            fold_scores = []
            feature_imps = None
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                print(f"Fold {fold}/{n_splits}")
                
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Log transform target
                y_train_log = np.log1p(y_train)
                
                # Preprocess features
                preprocessor.fit(X_train)
                X_train_processed = preprocessor.transform(X_train)
                X_val_processed = preprocessor.transform(X_val)
                
                # Model-specific feature selection
                if 'lgb' in name or 'cat' in name:
                    selector = lgb.LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=42,
                        device='gpu'
                    )
                elif 'ngb' in name:
                    selector = lgb.LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        num_leaves=63,
                        random_state=42,
                        device='gpu'
                    )
                else:
                    selector = lgb.LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        num_leaves=15,
                        max_depth=5,
                        random_state=42,
                        device='gpu'
                    )
                
                selector.fit(X_train_processed, y_train_log)
                
                # Store feature importances (only from first fold)
                if fold == 1:
                    feature_imps = selector.feature_importances_
                
                # Get feature importances and select top features
                importances = selector.feature_importances_
                if 'hist' in name or 'et' in name:
                    importance_threshold = np.percentile(importances, 30)
                else:
                    importance_threshold = np.percentile(importances, 20)
                
                selected_features = importances >= importance_threshold
                
                # Select features
                X_train_selected = X_train_processed[:, selected_features]
                X_val_selected = X_val_processed[:, selected_features]
                
                # Train model
                model.fit(X_train_selected, y_train_log)
                
                # Generate predictions
                fold_pred = np.expm1(model.predict(X_val_selected))
                oof_pred[val_idx] = fold_pred
                
                # Calculate fold score
                fold_score = rmsle(y_val, fold_pred)
                fold_scores.append(fold_score)
                
                # Generate test predictions
                test_processed = preprocessor.transform(test_features)
                test_selected = test_processed[:, selected_features]
                test_pred += np.expm1(model.predict(test_selected)) / n_splits
            
            # Store predictions
            oof_predictions[name] = np.maximum(oof_pred, 0)
            test_predictions[name] = np.maximum(test_pred, 0)
            
            # Store meta-features
            meta_features[:, i] = oof_predictions[name]
            meta_test[:, i] = test_predictions[name]
            
            # Calculate model score and log it
            model_score = rmsle(y, oof_predictions[name])
            model_weights[name] = 1.0 / (model_score ** 2)
            
            # Log model performance
            logger.log_model_score(name, model_score, fold_scores, feature_imps)
            logger.log_model_weight(name, model_weights[name])
            
            print(f"{name} RMSLE: {model_score:.4f}")
        
        # Train TabNet models
        for i, (name, model) in enumerate(tabnet_models.items(), start=len(models)):
            print(f"\nTraining {name}...")
            oof_pred = np.zeros(len(X_complete))
            test_pred = np.zeros(len(test_complete))
            fold_scores = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_complete), 1):
                print(f"Fold {fold}/{n_splits}")
                
                X_train, X_val = X_complete.iloc[train_idx], X_complete.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Log transform target
                y_train_log = np.log1p(y_train)
                
                # Preprocess features
                preprocessor_complete.fit(X_train)
                X_train_processed = preprocessor_complete.transform(X_train)
                X_val_processed = preprocessor_complete.transform(X_val)
                
                # Scale data for TabNet
                X_train_scaled = scaler.fit_transform(X_train_processed)
                X_val_scaled = scaler.transform(X_val_processed)
                
                # Convert to torch tensors
                X_train_tensor = torch.FloatTensor(X_train_scaled)
                y_train_tensor = torch.FloatTensor(y_train_log.values)
                X_val_tensor = torch.FloatTensor(X_val_scaled)
                
                # Train TabNet
                model.fit(
                    X_train_tensor, y_train_tensor,
                    eval_set=[(X_val_tensor, torch.FloatTensor(np.log1p(y_val.values)))],
                    patience=5,
                    max_epochs=200,
                    eval_metric=['rmse']
                )
                
                # Generate predictions
                fold_pred = np.expm1(model.predict(X_val_tensor).numpy())
                oof_pred[val_idx] = fold_pred
                
                # Calculate fold score
                fold_score = rmsle(y_val, fold_pred)
                fold_scores.append(fold_score)
                
                # Generate test predictions
                test_processed = preprocessor_complete.transform(test_complete)
                test_scaled = scaler.transform(test_processed)
                test_tensor = torch.FloatTensor(test_scaled)
                test_pred += np.expm1(model.predict(test_tensor).numpy()) / n_splits
            
            # Store predictions
            oof_predictions[name] = np.maximum(oof_pred, 0)
            test_predictions[name] = np.maximum(test_pred, 0)
            
            # Store meta-features
            meta_features[:, i] = oof_predictions[name]
            meta_test[:, i] = test_predictions[name]
            
            # Calculate model score and log it
            model_score = rmsle(y, oof_predictions[name])
            model_weights[name] = 1.0 / (model_score ** 2)
            
            # Log model performance
            logger.log_model_score(name, model_score, fold_scores)
            logger.log_model_weight(name, model_weights[name])
            
            print(f"{name} RMSLE: {model_score:.4f}")
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    model_weights = {k: v/total_weight for k, v in model_weights.items()}
    
    # Create enhanced meta-features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    meta_poly = poly.fit_transform(meta_features)
    meta_test_poly = poly.transform(meta_test)
    
    # Add weighted averages
    weighted_pred = sum(pred * model_weights[name] for name, pred in oof_predictions.items())
    weighted_test = sum(pred * model_weights[name] for name, pred in test_predictions.items())
    
    # Add pairwise differences
    for i, name1 in enumerate(list(models.keys()) + list(tabnet_models.keys())):
        for j, name2 in enumerate(list(models.keys()) + list(tabnet_models.keys())):
            if i < j:
                meta_features = np.column_stack([
                    meta_features,
                    np.abs(oof_predictions[name1] - oof_predictions[name2])
                ])
                meta_test = np.column_stack([
                    meta_test,
                    np.abs(test_predictions[name1] - test_predictions[name2])
                ])
    
    # Train meta-model with Ridge regression
    meta_model = Ridge(alpha=1.0, random_state=42)
    meta_model.fit(meta_poly, np.log1p(y))
    
    # Generate final predictions
    final_oof = np.maximum(np.expm1(meta_model.predict(meta_poly)), 0)
    final_test = np.maximum(np.expm1(meta_model.predict(meta_test_poly)), 0)
    
    # Blend with weighted average for robustness
    blend_weight = 0.7  # Meta-model gets 70% weight
    final_oof = blend_weight * final_oof + (1 - blend_weight) * weighted_pred
    final_test = blend_weight * final_test + (1 - blend_weight) * weighted_test
    
    oof_predictions['ensemble'] = final_oof
    test_predictions['ensemble'] = final_test
    
    # Calculate and log final ensemble score
    ensemble_score = rmsle(y, final_oof)
    logger.log_ensemble_score(ensemble_score)
    print(f"\nEnsemble RMSLE: {ensemble_score:.4f}")
    
    # Save run log
    log_file = logger.save_run()
    print(f"\nRun log saved to: {log_file}")
    
    # Show comparison with previous runs
    print("\nComparison with previous runs:")
    print(logger.compare_runs().to_string())
    
    return oof_predictions, test_predictions
