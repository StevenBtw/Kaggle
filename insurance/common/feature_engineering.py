import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from itertools import combinations_with_replacement
import hashlib
import joblib

def compute_input_hash(train_data, test_data):
    """Compute a hash of the input data to detect changes."""
    train_hash = str(pd.util.hash_pandas_object(train_data).values.sum())
    test_hash = str(pd.util.hash_pandas_object(test_data).values.sum())
    data_str = train_hash + test_hash
    return hashlib.md5(data_str.encode()).hexdigest()

def load_cached_features(data_hash):
    """Load cached feature sets if they exist."""
    cache_path = f'insurance/output/features/feature_sets_{data_hash}.joblib'
    if os.path.exists(cache_path):
        print("\nLoading cached feature sets...")
        return joblib.load(cache_path)
    return None

def save_features(features, name, subset='train'):
    """Save features to disk."""
    os.makedirs('insurance/output/features', exist_ok=True)
    path = f'insurance/output/features/{name}_{subset}.parquet'
    features.to_parquet(path)

def load_features(name, subset='train'):
    """Load features from disk if they exist."""
    path = f'insurance/output/features/{name}_{subset}.parquet'
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

def save_feature_sets(feature_sets, data_hash):
    """Save feature sets to cache."""
    os.makedirs('insurance/output/features', exist_ok=True)
    cache_path = f'insurance/output/features/feature_sets_{data_hash}.joblib'
    joblib.dump(feature_sets, cache_path)

def save_model_predictions(predictions, model_name, data_hash, fold=None):
    """Save model predictions to disk."""
    os.makedirs('insurance/output/models', exist_ok=True)
    if fold is not None:
        path = f'insurance/output/models/{model_name}_fold{fold}_{data_hash}.joblib'
    else:
        path = f'insurance/output/models/{model_name}_{data_hash}.joblib'
    joblib.dump(predictions, path)

def load_model_predictions(model_name, data_hash, fold=None):
    """Load model predictions from disk if they exist."""
    if fold is not None:
        path = f'insurance/output/models/{model_name}_fold{fold}_{data_hash}.joblib'
    else:
        path = f'insurance/output/models/{model_name}_{data_hash}.joblib'
    
    if os.path.exists(path):
        try:
            predictions = joblib.load(path)
            # Check if predictions have version information
            if 'version' not in predictions:
                return None
            return predictions
        except (OSError, EOFError):
            # If there's an error loading the file, return None
            return None
    return None

def create_or_load_features(name, create_fn, *args, **kwargs):
    """Create features or load from disk if they exist."""
    train_features = load_features(name, 'train')
    test_features = load_features(name, 'test')
    
    if train_features is None or test_features is None:
        print(f"Creating {name} features...")
        train_features, test_features = create_fn(*args, **kwargs)
        save_features(train_features, name, 'train')
        save_features(test_features, name, 'test')
    else:
        print(f"Loading {name} features from disk...")
    
    return train_features, test_features

def create_interactions(df_train, df_test, features, degree=3, handle_nan=False):
    """Create interaction features up to specified degree."""
    def process_df(df):
        if handle_nan:
            df = df.copy()
        else:
            # Fill NaN for models that can't handle them
            imputer = SimpleImputer(strategy='median')
            df = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)
        
        interactions = []
        for i in range(2, degree + 1):
            for combo in combinations_with_replacement(features, i):
                interaction = df[list(combo)].prod(axis=1)
                name = '_'.join(combo)
                interactions.append((name, interaction))
        return pd.DataFrame(dict(interactions))
    
    return process_df(df_train), process_df(df_test)

def create_statistical_features(df_train, df_test, base_features, handle_nan=False):
    """Create statistical aggregation features."""
    def process_df(df, is_train=True):
        if handle_nan:
            df_calc = df.copy()
        else:
            # Fill NaN for models that can't handle them
            imputer = SimpleImputer(strategy='median')
            df_calc = pd.DataFrame(imputer.fit_transform(df[base_features]), columns=base_features)
        
        stats = {}
        
        # Mean ratios
        for f1 in base_features:
            for f2 in base_features:
                if f1 != f2:
                    stats[f'{f1}_div_{f2}'] = df_calc[f1] / (df_calc[f2] + 1e-6)
        
        # Rolling statistics - only for numeric features that are not the target
        numeric_cols = df_calc.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'Premium Amount']
        for col in numeric_cols:
            stats[f'{col}_rolling_mean'] = df_calc[col].rolling(window=2, min_periods=1).mean()
            stats[f'{col}_rolling_std'] = df_calc[col].rolling(window=2, min_periods=1).std()
        
        return pd.DataFrame(stats)
    
    return process_df(df_train), process_df(df_test)

def create_cluster_features(df_train, df_test, features, n_clusters=5):
    """Create clustering-based features (always needs imputation)."""
    # Create imputer for handling missing values
    imputers = {}  # Store imputers for each feature set
    kmeans_models = {}  # Store KMeans models for each feature set
    
    def process_df(df, fit=True):
        cluster_features = {}
        
        # Cluster on different feature combinations
        feature_sets = [
            ['Age', 'Health Score'],
            ['Age', 'Health Score', 'Number of Dependents'],
            ['Health Score', 'Number of Dependents']
        ]
        
        for i, feature_set in enumerate(feature_sets):
            # Get or create imputer and kmeans for this feature set
            if fit:
                imputers[i] = SimpleImputer(strategy='median')
                kmeans_models[i] = KMeans(n_clusters=n_clusters, random_state=42)
            
            # Impute missing values
            data = imputers[i].fit_transform(df[feature_set]) if fit else imputers[i].transform(df[feature_set])
            
            # Fit or predict clusters
            if fit:
                cluster_labels = kmeans_models[i].fit_predict(data)
            else:
                cluster_labels = kmeans_models[i].predict(data)
            
            cluster_features[f'cluster_{i}'] = cluster_labels
            
            # Distance to cluster centers
            distances = kmeans_models[i].transform(data)
            for j in range(n_clusters):
                cluster_features[f'cluster_{i}_dist_{j}'] = distances[:, j]
        
        return pd.DataFrame(cluster_features)
    
    return process_df(df_train, fit=True), process_df(df_test, fit=False)

def create_domain_features(df_train, df_test, handle_nan=False):
    """Create domain-specific features."""
    def process_df(df):
        if handle_nan:
            df_calc = df.copy()
        else:
            # Fill NaN for models that can't handle them
            df_calc = df.fillna({
                'Health Score': df['Health Score'].median(),
                'Age': df['Age'].median(),
                'Number of Dependents': df['Number of Dependents'].median()
            })
        
        features = {}
        
        # Health Score-related features
        features['health_category'] = pd.cut(df_calc['Health Score'], 
                                    bins=[0, 50, 70, 85, 100],
                                    labels=['poor', 'fair', 'good', 'excellent'])
        features['health_category'] = LabelEncoder().fit_transform(features['health_category'])
        
        # Age-related features
        features['age_decade'] = df_calc['Age'] // 10
        features['is_senior'] = (df_calc['Age'] >= 50).astype(int)
        features['is_young_adult'] = ((df_calc['Age'] >= 20) & (df_calc['Age'] < 35)).astype(int)
        features['is_middle_age'] = ((df_calc['Age'] >= 35) & (df_calc['Age'] < 50)).astype(int)
        
        # Family size features
        features['has_dependents'] = (df_calc['Number of Dependents'] > 0).astype(int)
        features['large_family'] = (df_calc['Number of Dependents'] >= 3).astype(int)
        features['single_parent'] = (df_calc['Number of Dependents'] > 0) & (df_calc['Age'] < 35)
        
        # Risk factors
        features['high_risk'] = ((df_calc['Health Score'] < 50) & (df_calc['Smoking Status'] == 'Yes')).astype(int)
        features['very_high_risk'] = ((df_calc['Health Score'] < 40) & (df_calc['Smoking Status'] == 'Yes') & (df_calc['Age'] > 50)).astype(int)
        features['multiple_risk_factors'] = ((df_calc['Health Score'] < 60) & (df_calc['Smoking Status'] == 'Yes') & (df_calc['Age'] > 40)).astype(int)
        
        # Location-specific risk combinations
        features['location_risk'] = pd.factorize(df_calc['Location'])[0] * features['high_risk']
        
        return pd.DataFrame(features)
    
    return process_df(df_train), process_df(df_test)

def create_polynomial_features(df_train, df_test, features, degree=3, handle_nan=False):
    """Create polynomial features up to specified degree."""
    # Initialize transformers
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    imputer = SimpleImputer(strategy='median') if not handle_nan else None
    
    def process_df(df, fit=True):
        data = df[features].copy()  # Create a copy to avoid modifying original
        
        if not handle_nan:
            # Fill missing values for models that can't handle them
            if fit:
                data = pd.DataFrame(imputer.fit_transform(data), columns=features)
            else:
                data = pd.DataFrame(imputer.transform(data), columns=features)
        
        # For models that can handle NaN, we still need to fill NaN for polynomial features
        # but we'll mark the locations to restore them later
        mask = data.isna()
        data = data.fillna(data.median())
        
        if fit:
            poly_features = poly.fit_transform(data)
        else:
            poly_features = poly.transform(data)
        
        feature_names = [f"poly_{i}" for i in range(poly_features.shape[1])]
        result = pd.DataFrame(poly_features, columns=feature_names)
        
        if handle_nan:
            # Restore NaN values where they were originally
            for col in result.columns:
                # If any of the original features used in this polynomial term had NaN,
                # the result should be NaN
                should_be_nan = mask.any(axis=1)
                result.loc[should_be_nan, col] = np.nan
        
        return result
    
    return process_df(df_train, fit=True), process_df(df_test, fit=False)

def engineer_features(train_data, test_data):
    """Engineer features for model training."""
    # Compute hash of input data
    data_hash = compute_input_hash(train_data, test_data)
    print(f"\nInput data hash: {data_hash}")
    
    # Try to load cached feature sets
    cached = load_cached_features(data_hash)
    if cached is not None:
        print("Using cached feature sets")
        return cached
    
    print("\nEngineering features...")
    
    # List of features
    numerical_features = [
        'Age', 'Health Score', 'Number of Dependents', 'Annual Income',
        'Credit Score', 'Insurance Duration', 'Previous Claims'
    ]
    
    categorical_features = [
        'Gender', 'Marital Status', 'Education Level', 'Occupation',
        'Location', 'Policy Type', 'Smoking Status', 'Exercise Frequency',
        'Property Type'
    ]
    
    # Create copies to avoid modifying original data
    train = train_data.copy()
    test = test_data.copy()
    
    # Label encode categorical features
    label_encoders = {}
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        train[feature] = label_encoders[feature].fit_transform(train[feature])
        test[feature] = label_encoders[feature].transform(test[feature])
    
    # Create or load features for models that can handle NaN
    train_domain_nan, test_domain_nan = create_or_load_features(
        'domain_nan', create_domain_features, train, test, handle_nan=True
    )
    
    train_stats_nan, test_stats_nan = create_or_load_features(
        'stats_nan', create_statistical_features, train, test, numerical_features, handle_nan=True
    )
    
    train_poly_nan, test_poly_nan = create_or_load_features(
        'poly_nan', create_polynomial_features, train, test, numerical_features, handle_nan=True
    )
    
    train_interactions_nan, test_interactions_nan = create_or_load_features(
        'interactions_nan', create_interactions, train, test, 
        numerical_features + categorical_features, handle_nan=True
    )
    
    # Create or load features for models that need imputation
    train_domain, test_domain = create_or_load_features(
        'domain', create_domain_features, train, test, handle_nan=False
    )
    
    train_stats, test_stats = create_or_load_features(
        'stats', create_statistical_features, train, test, numerical_features, handle_nan=False
    )
    
    train_clusters, test_clusters = create_or_load_features(
        'clusters', create_cluster_features, train, test, numerical_features
    )
    
    train_poly, test_poly = create_or_load_features(
        'poly', create_polynomial_features, train, test, numerical_features, handle_nan=False
    )
    
    train_interactions, test_interactions = create_or_load_features(
        'interactions', create_interactions, train, test, 
        numerical_features + categorical_features, handle_nan=False
    )
    
    # Combine features for models that can handle NaN
    train_nan = pd.concat([
        train,
        train_domain_nan,
        train_stats_nan,
        train_poly_nan,
        train_interactions_nan
    ], axis=1)
    
    test_nan = pd.concat([
        test,
        test_domain_nan,
        test_stats_nan,
        test_poly_nan,
        test_interactions_nan
    ], axis=1)
    
    # Combine features for models that need complete data
    train_complete = pd.concat([
        train,
        train_domain,
        train_stats,
        train_clusters,
        train_poly,
        train_interactions
    ], axis=1)
    
    test_complete = pd.concat([
        test,
        test_domain,
        test_stats,
        test_clusters,
        test_poly,
        test_interactions
    ], axis=1)
    
    # Update feature lists
    numerical_features_complete = numerical_features.copy()
    numerical_features_complete.extend(train_domain.columns)
    numerical_features_complete.extend(train_stats.columns)
    numerical_features_complete.extend(train_clusters.columns)
    numerical_features_complete.extend(train_poly.columns)
    numerical_features_complete.extend(train_interactions.columns)
    
    numerical_features_nan = numerical_features.copy()
    numerical_features_nan.extend(train_domain_nan.columns)
    numerical_features_nan.extend(train_stats_nan.columns)
    numerical_features_nan.extend(train_poly_nan.columns)
    numerical_features_nan.extend(train_interactions_nan.columns)
    
    # Split features and target
    X_nan = train_nan.drop('Premium Amount', axis=1)
    X_complete = train_complete.drop('Premium Amount', axis=1)
    y = train['Premium Amount']
    
    # Create feature sets tuple
    feature_sets = (X_nan, X_complete, y, test_nan, test_complete, 
                   numerical_features_nan, numerical_features_complete, categorical_features)
    
    # Cache the feature sets
    save_feature_sets(feature_sets, data_hash)
    
    return feature_sets
