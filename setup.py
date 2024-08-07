from setuptools import setup, find_packages

setup(
    name='lstm_forecast',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        'ta',
        'statsmodels',
        'numpy',
        'yfinance',
        'matplotlib',
        'torch',
        'plotly',
        'scikit-learn',
        'xgboost',
        'optuna',
        'onnxruntime',
        'onnx',
        'flask',
    ],
    entry_points={
        'console_scripts': [
            'lstm_forecast=lstm_forecast.cli:main',
        ],
    },
)
