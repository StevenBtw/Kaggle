import pandas as pd
import numpy as np

# Read the datasets
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

# Display basic information
print("\n=== Dataset Shapes ===")
print("Training Data:", train_df.shape)
print("Test Data:", test_df.shape)

# Check for missing values
print("\n=== Missing Values ===")
print("\nTraining Data:")
print(train_df.isnull().sum())
print("\nTest Data:")
print(test_df.isnull().sum())

# Display unique values in categorical columns
categorical_cols = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 
                   'Location', 'Policy Type', 'Smoking Status', 'Exercise Frequency', 
                   'Property Type', 'Customer Feedback']

print("\n=== Categorical Features Analysis ===")
for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(train_df[col].value_counts())

# Basic statistics of numerical columns
numerical_cols = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score',
                 'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration',
                 'Premium Amount']

print("\n=== Numerical Features Statistics ===")
print(train_df[numerical_cols].describe())

# Correlation with target
correlations = train_df[numerical_cols].corr()['Premium Amount'].sort_values(ascending=False)
print("\n=== Correlations with Premium Amount ===")
print(correlations)
