import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lstm_forecast.logger import setup_logger

logger = setup_logger("feature_selection_logger", "logs/feature_selection.log")

def rolling_feature_selection(X, y, window_size, num_features, lag=10):
    """
    Perform rolling window feature selection.
    
    Parameters:
    - X: DataFrame, feature matrix
    - y: DataFrame, target matrix
    - window_size: int, size of the rolling window
    - num_features: int, number of features to select
    
    Returns:
    - selected_features: list of selected feature names
    """
    logger.info(f"Starting rolling feature selection with window size {window_size} and selecting {num_features} features")
    selected_features = []
    for i in range(0, len(X) - window_size + 1, lag):
        X_window = X.iloc[i:i+window_size]
        y_window = y.iloc[i:i+window_size]
        
        model = XGBRegressor(n_estimators=100, random_state=42)
        rfe = RFE(estimator=model, n_features_to_select=num_features)
        
        rfe.fit(X_window, y_window)
        selected_features.extend(X.columns[rfe.support_].tolist())
    
    # Compare most frequent features
    feature_counts = pd.Series(selected_features).value_counts()
    logger.info(f"Rolling feature selection completed. Selected features: {feature_counts.index.tolist()[:num_features]}")
    return feature_counts.index.tolist()[:num_features]

def granger_causality_test(X, y, max_lag=5, threshold=0.05):
    """
    Perform Granger Causality test for feature selection.
    
    Parameters:
    - X: DataFrame, feature matrix
    - y: Series, target variable
    - max_lag: int, maximum number of lags to test
    - threshold: float, p-value threshold for significance
    
    Returns:
    - causal_features: list of features that Granger-cause the target
    """
    logger.info(f"Starting Granger Causality test with max lag {max_lag} and p-value threshold {threshold}")
    causal_features = []
    for column in X.columns:
        data = pd.concat([y, X[column]], axis=1)
        test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        # Check if the feature Granger-causes the target at any lag
        if any(test_result[i+1][0]['ssr_ftest'][1] < threshold for i in range(max_lag)):
            causal_features.append(column)
    
    logger.info(f"Granger Causality test completed. Causal features: {causal_features}")
    return causal_features

def var_causality_test(X, y, max_lag=5, threshold=0.05):
    """
    Perform VAR (Vector Autoregression) Causality test for feature selection.
    
    Parameters:
    - X: DataFrame, feature matrix
    - y: Series, target variable
    - max_lag: int, maximum number of lags to test
    - threshold: float, p-value threshold for significance
    
    Returns:
    - causal_features: list of features that VAR-cause the target
    """
    logger.info(f"Starting VAR Causality test with max lag {max_lag} and p-value threshold {threshold}")
    causal_features = []
    data = pd.concat([y, X], axis=1)
    model = VAR(data)
    result = model.fit(maxlags=max_lag, ic='aic')
    
    for column in X.columns:
        test_result = result.test_causality(causing=column, caused=y.name)
        if test_result.pvalue < threshold:
            causal_features.append(column)
    
    logger.info(f"VAR Causality test completed. Causal features: {causal_features}")
    return causal_features

def correlation_analysis(df, threshold=0.9):
    """
    Perform correlation analysis to remove highly correlated features.
    
    Parameters:
    - df: DataFrame, input feature matrix
    - threshold: float, correlation threshold for feature removal
    
    Returns:
    - selected_features: list of selected feature names
    """
    logger.info(f"Starting correlation analysis with threshold {threshold}")
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_features = df.columns.difference(to_drop).tolist()
    logger.info(f"Correlation analysis completed. Selected features: {selected_features}")
    return selected_features

def time_series_feature_selection(X, y, num_features, window_size=252, max_lag=5):
    """
    Perform feature selection for time series data.
    
    Parameters:
    - X: DataFrame, feature matrix
    - y: DataFrame or Series, target variable(s)
    - num_features: int, number of features to select
    - window_size: int, size of the rolling window
    - max_lag: int, maximum number of lags for causality tests
    
    Returns:
    - final_features: list of selected feature names
    """
    logger.info("Starting time series feature selection process")
    logger.info(f"Parameters: num_features={num_features}, window_size={window_size}, max_lag={max_lag}")
    
    # Step 1: Rolling window feature selection
    rolling_features = rolling_feature_selection(X, y, window_size, num_features, lag=max_lag)
    
    # Step 2: VAR Causality test
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  # Use the first target for causality test
    causal_features = var_causality_test(X[rolling_features], y, max_lag)
    
    # Step 3: Correlation analysis
    uncorrelated_features = correlation_analysis(X[causal_features])
    
    # Step 4: Final feature selection
    final_features = uncorrelated_features[:num_features]
    logger.info(f"Time series feature selection completed. Final selected features: {final_features}")
    
    return final_features
