import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

def engineer_features(df):
    # Convert Policy Start Date to datetime and extract features
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Policy_Month'] = df['Policy Start Date'].dt.month
    df['Policy_Quarter'] = df['Policy Start Date'].dt.quarter
    df['Policy_Year'] = df['Policy Start Date'].dt.year
    df['Policy_DayOfWeek'] = df['Policy Start Date'].dt.dayofweek
    
    # Basic ratios and interactions
    df['Age_Income_Ratio'] = df['Age'] / (df['Annual Income'] + 1)
    df['Income_per_Dependent'] = df['Annual Income'] / (df['Number of Dependents'] + 1)
    df['Health_Age_Ratio'] = df['Health Score'] / df['Age']
    
    # Advanced feature combinations
    df['Risk_Score'] = (
        (df['Previous Claims'] * 2) + 
        (df['Age'] * 0.5) + 
        (100 - df['Health Score']) + 
        (df['Vehicle Age'] * 1.5)
    )
    
    # Lifestyle score with more granular exercise impact
    exercise_map = {
        'Daily': 2.5,
        'Weekly': 2.0,
        'Monthly': 1.5,
        'Never': 1.0
    }
    smoking_map = {'No': 0.6, 'Yes': 1.4}
    
    df['Exercise_Factor'] = df['Exercise Frequency'].map(exercise_map)
    df['Smoking_Factor'] = df['Smoking Status'].map(smoking_map)
    df['Lifestyle_Score'] = df['Exercise_Factor'] * df['Smoking_Factor'] * (df['Health Score'] / 40)
    
    # Enhanced financial stability score
    df['Claims_Factor'] = np.where(df['Previous Claims'].isna(), 0.8, 1.0 / (df['Previous Claims'] + 1))
    df['Financial_Score'] = (
        (df['Credit Score'] / 850) * 
        np.log1p(df['Annual Income']) * 
        (df['Insurance Duration'] + 1) *
        df['Claims_Factor']
    )
    
    # Location-based risk factor
    location_map = {'Urban': 1.2, 'Suburban': 1.0, 'Rural': 0.9}
    df['Location_Risk'] = df['Location'].map(location_map)
    
    # Education-based risk modifier
    education_map = {
        'PhD': 0.8,
        "Master's": 0.9,
        "Bachelor's": 1.0,
        'High School': 1.1
    }
    df['Education_Risk'] = df['Education Level'].map(education_map)
    
    # Combined risk score
    df['Total_Risk_Score'] = (
        df['Risk_Score'] * 
        df['Location_Risk'] * 
        df['Education_Risk'] * 
        (2 - df['Lifestyle_Score'] / df['Lifestyle_Score'].max())
    )
    
    # Drop temporary columns
    df = df.drop(['Exercise_Factor', 'Smoking_Factor', 'Claims_Factor'], axis=1)
    
    return df

print("Engineering features...")
X = train_df.drop(['Premium Amount', 'id'], axis=1)
X = engineer_features(X)
test_features = test_df.drop(['id'], axis=1)
test_features = engineer_features(test_features)
y = train_df['Premium Amount']

# Drop datetime column after feature engineering
X = X.drop('Policy Start Date', axis=1)
test_features = test_features.drop('Policy Start Date', axis=1)

# Split categorical and numerical columns
categorical_features = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 
                       'Location', 'Policy Type', 'Smoking Status', 'Exercise Frequency', 
                       'Property Type', 'Customer Feedback']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Create base models
def create_models():
    models = {
        'lgb1': lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            random_state=42
        ),
        'lgb2': lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=127,
            max_depth=12,
            random_state=43
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        ),
        'hist_gb': HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        )
    }
    return models

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Prepare cross-validation
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# Generate out-of-fold predictions
def generate_oof_predictions():
    models = create_models()
    oof_predictions = {}
    test_predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        oof_pred = np.zeros(len(X))
        test_pred = np.zeros(len(test_features))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            print(f"Fold {fold}/{n_splits}")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Preprocess
            preprocessor.fit(X_train)
            X_train_processed = preprocessor.transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            
            # Train and predict
            model.fit(X_train_processed, y_train)
            oof_pred[val_idx] = model.predict(X_val_processed)
            
            # Predict test
            test_processed = preprocessor.transform(test_features)
            test_pred += model.predict(test_processed) / n_splits
        
        oof_predictions[name] = oof_pred
        test_predictions[name] = test_pred
        
        # Calculate RMSLE for this model
        rmsle = np.sqrt(mean_squared_log_error(y, oof_pred))
        print(f"{name} RMSLE: {rmsle:.4f}")
    
    return oof_predictions, test_predictions

# Generate predictions
print("Generating out-of-fold predictions...")
oof_predictions, test_predictions = generate_oof_predictions()

# Optimize weights using Bayesian Optimization
def objective(**weights):
    # Normalize weights to sum to 1
    total = sum(weights.values())
    normalized_weights = {k: v/total for k, v in weights.items()}
    
    # Compute weighted prediction
    weighted_pred = sum(normalized_weights[name] * pred 
                       for name, pred in oof_predictions.items())
    
    # Return negative RMSLE (for maximization)
    return -np.sqrt(mean_squared_log_error(y, weighted_pred))

# Define parameter space for weights
pbounds = {name: (0, 1) for name in oof_predictions.keys()}

# Run Bayesian Optimization
print("\nOptimizing ensemble weights...")
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=42
)

optimizer.maximize(init_points=5, n_iter=20)

# Get best weights
best_weights = optimizer.max['params']
total = sum(best_weights.values())
best_weights = {k: v/total for k, v in best_weights.items()}

print("\nBest weights found:")
for name, weight in best_weights.items():
    print(f"{name}: {weight:.4f}")

# Generate final predictions using best weights
final_predictions = sum(best_weights[name] * pred 
                       for name, pred in test_predictions.items())

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Premium Amount': final_predictions
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")

# Calculate final RMSLE on OOF predictions
final_oof_predictions = sum(best_weights[name] * pred 
                          for name, pred in oof_predictions.items())
final_rmsle = np.sqrt(mean_squared_log_error(y, final_oof_predictions))
print(f"\nFinal OOF RMSLE: {final_rmsle:.4f}")
