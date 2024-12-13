from pytorch_tabnet.tab_model import TabNetRegressor
from ngboost import NGBRegressor
from ngboost.learners import default_tree_learner
from ngboost.distns import Normal
import torch

def create_tabnet_models(n_gpu_threads=4):
    """Create TabNet models with different hyperparameters."""
    if torch.cuda.is_available():
        device = 'cuda'
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(0.7)  # Prevent OOM
    else:
        device = 'cpu'
    
    models = []
    
    # Base model
    models.append(TabNetRegressor(
        n_d=64,  # Width of the decision prediction layer
        n_a=64,  # Width of the attention embedding
        n_steps=5,  # Number of steps in the architecture
        gamma=1.5,  # Coefficient for feature reusage
        n_independent=2,  # Number of independent GLU layers
        n_shared=2,  # Number of shared GLU layers
        lambda_sparse=1e-3,  # Sparsity regularization
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params=dict(
            mode="min",
            patience=5,
            min_lr=1e-5,
            factor=0.5,
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax',  # "sparsemax" or "entmax"
        device_name=device
    ))
    
    # Deeper model
    models.append(TabNetRegressor(
        n_d=128,
        n_a=128,
        n_steps=7,
        gamma=1.3,
        n_independent=3,
        n_shared=3,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2),
        scheduler_params=dict(
            mode="min",
            patience=5,
            min_lr=1e-5,
            factor=0.5,
        ),
        scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
        mask_type='entmax',
        device_name=device
    ))
    
    return models

def create_ngboost_models():
    """Create NGBoost models with different hyperparameters."""
    models = []
    
    # Base model
    models.append(NGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        natural_gradient=True,
        verbose=True,
        Base=default_tree_learner,  # Already a configured base learner
        Dist=Normal,
        random_state=42
    ))
    
    # More trees, slower learning
    models.append(NGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        natural_gradient=True,
        verbose=True,
        Base=default_tree_learner,  # Already a configured base learner
        Dist=Normal,
        random_state=43
    ))
    
    # Deeper trees
    models.append(NGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        natural_gradient=True,
        verbose=True,
        Base=default_tree_learner,  # Already a configured base learner
        Dist=Normal,
        random_state=44
    ))
    
    return models
