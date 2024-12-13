import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Feature engineering
def engineer_features(df):
    # Convert Policy Start Date to datetime and extract features
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Policy_Month'] = df['Policy Start Date'].dt.month
    df['Policy_Quarter'] = df['Policy Start Date'].dt.quarter
    
    # Create interaction features
    df['Age_Income_Ratio'] = df['Age'] / (df['Annual Income'] + 1)  # Add 1 to avoid division by zero
    df['Income_per_Dependent'] = df['Annual Income'] / (df['Number of Dependents'] + 1)
    df['Health_Age_Ratio'] = df['Health Score'] / df['Age']
    
    # Create risk score
    df['Risk_Score'] = (
        (df['Previous Claims'] * 2) + 
        (df['Age'] * 0.5) + 
        (100 - df['Health Score']) + 
        (df['Vehicle Age'] * 1.5)
    )
    
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

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),  # Use KNN imputer for better missing value handling
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))  # Add polynomial features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create model pipeline with HistGradientBoostingRegressor (faster and better handling of missing values)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_depth=8,
        l2_regularization=1.0,
        random_state=42
    ))
])

# Split data
print("Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop Policy Start Date after feature engineering
X_train = X_train.drop('Policy Start Date', axis=1)
X_val = X_val.drop('Policy Start Date', axis=1)
test_features = test_features.drop('Policy Start Date', axis=1)

# Train model
print("Training model...")
model.fit(X_train, y_train)

# Evaluate
print("\nEvaluating model...")
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)

# Calculate RMSLE
train_rmsle = np.sqrt(mean_squared_log_error(y_train, train_pred))
val_rmsle = np.sqrt(mean_squared_log_error(y_val, val_pred))

print(f"Train RMSLE: {train_rmsle:.4f}")
print(f"Validation RMSLE: {val_rmsle:.4f}")

# Make predictions on test set
print("\nMaking predictions on test set...")
test_pred = model.predict(test_features)

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Premium Amount': test_pred
})

submission.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")

# Print feature importance summary
if hasattr(model['regressor'], 'feature_importances_'):
    feature_names = (
        numerical_features + 
        [f"{feat}_{val}" for feat, vals in 
         model['preprocessor'].named_transformers_['cat'].named_steps['onehot'].categories_ 
         for val in vals[1:]]
    )
    importances = model['regressor'].feature_importances_
    for feat, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{feat}: {imp:.4f}")
