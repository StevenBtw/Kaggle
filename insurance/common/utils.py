import numpy as np
from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred):
    """Calculate Root Mean Squared Logarithmic Error."""
    return np.sqrt(mean_squared_log_error(y_true, y_pred))
