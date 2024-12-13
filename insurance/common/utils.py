import numpy as np
import hashlib
import pandas as pd

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error."""
    return np.sqrt(np.mean(np.square(np.log1p(y_pred) - np.log1p(y_true))))

def compute_input_hash(train_data, test_data):
    """Compute a hash of the input data to detect changes."""
    train_hash = str(pd.util.hash_pandas_object(train_data).values.sum())
    test_hash = str(pd.util.hash_pandas_object(test_data).values.sum())
    data_str = train_hash + test_hash
    return hashlib.md5(data_str.encode()).hexdigest()
