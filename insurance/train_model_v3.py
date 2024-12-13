import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import warnings
from joblib import parallel_backend
warnings.filterwarnings('ignore')

# Enable parallel processing
n_jobs = -1  # Use all available CPU cores
n_gpu_threads = 4  # Number of GPU threads to use

# Load data with optimized dtypes
print("Loading data...")
dtype_map = {
    'id': 'int32',
    'Age': 'int8',
    'Gender': 'category',
    'Credit Score': 'int16',
    'Annual Income': 'float32',
    'Marital Status': 'category',
    'Number of Dependents': 'int8',
    'Education Level': 'category',
    'Occupation': 'category',
    'Location': 'category',
    'Vehicle Age': 'int8',
    'Insurance Duration': 'int8',
    'Previous Claims': 'int8',
    'Policy Type': 'category',
    'Policy Start Date': 'str',
    'Smoking Status': 'category',
    'Exercise Frequency': 'category',
    'Health Score': 'int8',
    'Property Type': 'category',
    'Customer Feedback': 'category',
    'Premium Amount': 'float32'
}

train_df = pd.read_csv('data/train.csv', dtype=dtype_map)
test_df = pd.read_csv('data/test.csv', dtype=dtype_map)

def engineer_features(df):
    # Convert Policy Start Date to datetime and extract features
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Policy_Month'] = df['Policy Start Date'].dt.month.astype('int8')
    df['Policy_Quarter'] = df['Policy Start Date'].dt.quarter.astype('int8')
    df['Policy_Year'] = df['Policy Start Date'].dt.year.astype('int16')
    
    # Create interaction features (with optimized dtypes)
    df['Age_Income_Ratio'] = (df['Age'].astype('float32') / (df['Annual Income'] + 1)).astype('float32')
    df['Income_per_Dependent'] = (df['Annual Income'] / (df['Number of Dependents'] + 1)).astype('float32')
    df['Health_Age_Ratio'] = (df['Health Score'].astype('float32') / df['Age']).astype('float32')
    
    # Risk scores
    df['Risk_Score'] = (
        (df['Previous Claims'] * 2) + 
        (df['Age'] * 0.5) + 
        (100 - df['Health Score']) + 
        (df['Vehicle Age'] * 1.5)
    ).astype('float32')
    
    # Lifestyle score
    df['Lifestyle_Score'] = df.apply(lambda x: 
        (2 if x['Exercise Frequency'] == 'Daily' else
         1.5 if x['Exercise Frequency'] == 'Weekly' else
         1 if x['Exercise Frequency'] == 'Monthly' else 0.5) *
        (0.7 if x['Smoking Status'] == 'No' else 1.3) *
        (x['Health Score'] / 50), axis=1
    ).astype('float32')
    
    # Financial stability score
    df['Financial_Score'] = (
        (df['Credit Score'] / 800) * 
        (df['Annual Income'] / df['Annual Income'].mean()) * 
        (df['Insurance Duration'] + 1)
    ).astype('float32')
    
    # Property risk score
    df['Property_Risk'] = df.apply(lambda x:
        (1.2 if x['Property Type'] == 'House' else
         1.0 if x['Property Type'] == 'Apartment' else 1.1) *
        (x['Vehicle Age'] + 1), axis=1
    ).astype('float32')
    
    return df

print("Engineering features...")
X = train_df.drop(['Premium Amount', 'id'], axis=1)
X = engineer_features(X)
test_features = test_df.drop(['id'], axis=1)
test_features = engineer_features(test_features)
y = train_df['Premium Amount']

# Split categorical and numerical columns
categorical_features = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 
                       'Location', 'Policy Type', 'Smoking Status', 'Exercise Frequency', 
                       'Property Type', 'Customer Feedback']
numerical_features = [col for col in X.columns if col not in categorical_features and col != 'Policy Start Date']

# Create base models with GPU support
hist_gb = HistGradientBoostingRegressor(
    max_iter=500,  # Increased iterations
    learning_rate=0.03,  # Reduced learning rate for better convergence
    max_depth=12,  # Increased depth
    l2_regularization=0.5,
    random_state=42,
    max_bins=255,  # Optimized for GPU
)

lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=12,
    random_state=42,
    device='gpu',  # Enable GPU
    gpu_platform_id=0,
    gpu_device_id=0,
    n_jobs=n_gpu_threads,
    verbose=-1
)

cat_model = cb.CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=12,
    random_state=42,
    verbose=False,
    task_type='GPU',  # Enable GPU
    devices='0',
    thread_count=n_gpu_threads
)

xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=12,
    random_state=42,
    tree_method='gpu_hist',  # Enable GPU
    predictor='gpu_predictor',
    gpu_id=0,
    n_jobs=n_gpu_threads
)

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5, n_jobs=n_jobs)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5, n_jobs=n_jobs)),
    ('onehot', OneHotEncoder(drop='first', sparse=False, handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    n_jobs=n_jobs
)

# Create ensemble model
ensemble = VotingRegressor(
    estimators=[
        ('hist_gb', hist_gb),
        ('lgb', lgb_model),
        ('cat', cat_model),
        ('xgb', xgb_model)
    ],
    weights=[1, 1.2, 1.1, 1],  # Slightly higher weight to LightGBM and CatBoost
    n_jobs=n_jobs
)

# Create final pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ensemble)
])

# Prepare cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_scores = []

# Drop Policy Start Date after feature engineering
X = X.drop('Policy Start Date', axis=1)
test_features = test_features.drop('Policy Start Date', axis=1)

print("Starting cross-validation training...")
# Use parallel backend for scikit-learn operations
with parallel_backend('threading', n_jobs=n_jobs):
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        val_pred = model.predict(X_val)
        val_rmsle = np.sqrt(mean_squared_log_error(y_val, val_pred))
        cv_scores.append(val_rmsle)
        print(f"Fold {fold} RMSLE: {val_rmsle:.4f}")

print(f"\nMean CV RMSLE: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# Train final model on full dataset
print("\nTraining final model on full dataset...")
model.fit(X, y)

# Make predictions on test set
print("Making predictions on test set...")
test_pred = model.predict(test_features)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Premium Amount': test_pred
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
