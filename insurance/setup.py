from setuptools import setup, find_packages

setup(
    name="insurance",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.2",
        "pandas>=1.2.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "catboost>=1.0.0",
        "joblib>=1.0.0",
        "pytorch-tabnet>=3.1.1",
        "ngboost>=0.3.12",
        "torch>=1.9.0",
    ],
    python_requires=">=3.8",
)
