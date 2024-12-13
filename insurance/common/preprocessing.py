import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import lightgbm as lgb

def create_preprocessor(numerical_features, categorical_features, n_jobs=-1):
    """Create a preprocessing pipeline."""
    
    # Numerical features pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical features pipeline - using -1 as fill value since categories are encoded as non-negative integers
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        n_jobs=1,  # Force single job to avoid parallel processing issues
        verbose=0,
        sparse_threshold=0
    )
    
    # Wrap in a pipeline that preserves feature names
    class FeatureNamePreservingPipeline(Pipeline):
        def fit_transform(self, X, y=None, **fit_params):
            # Get feature names before transformation
            self.feature_names_ = self._get_feature_names(X)
            result = super().fit_transform(X, y, **fit_params)
            return pd.DataFrame(result, columns=self.feature_names_, index=X.index)
        
        def transform(self, X):
            result = super().transform(X)
            return pd.DataFrame(result, columns=self.feature_names_, index=X.index)
        
        def _get_feature_names(self, X):
            # Get numeric feature names (unchanged)
            numeric_features = [f for f in numerical_features if f != 'Premium Amount']
            
            # Get categorical feature names after one-hot encoding
            cat_features = []
            for feature in categorical_features:
                unique_values = sorted(X[feature].unique())
                cat_features.extend([f"{feature}_{val}" for val in unique_values])
            
            return numeric_features + cat_features
    
    pipeline = FeatureNamePreservingPipeline([
        ('preprocessor', preprocessor)
    ])
    
    return pipeline
