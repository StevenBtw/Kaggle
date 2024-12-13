import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_log_error
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Separate features and target
y = train_df['Premium Amount']
X = train_df.drop(['Premium Amount', 'id', 'Policy Start Date'], axis=1)
test_features = test_df.drop(['id', 'Policy Start Date'], axis=1)

# Split categorical and numerical columns
categorical_features = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 
                       'Location', 'Policy Type', 'Smoking Status', 'Exercise Frequency', 
                       'Property Type', 'Customer Feedback']
numerical_features = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score',
                     'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration']

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
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

# Create model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

# Split data
print("Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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
