import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import grangercausalitytests

def rolling_feature_selection(X, y, window_size, num_features):
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
    selected_features = []
    for i in range(len(X) - window_size + 1):
        X_window = X.iloc[i:i+window_size]
        y_window = y.iloc[i:i+window_size]
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        rfe = RFE(estimator=model, n_features_to_select=num_features)
        
        rfe.fit(X_window, y_window)
        selected_features.extend(X.columns[rfe.support_].tolist())
    
    return list(set(selected_features))

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
    causal_features = []
    for column in X.columns:
        data = pd.concat([y, X[column]], axis=1)
        test_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        
        # Check if the feature Granger-causes the target at any lag
        if any(test_result[i+1][0]['ssr_ftest'][1] < threshold for i in range(max_lag)):
            causal_features.append(column)
    
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
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_features = df.columns.difference(to_drop).tolist()
    return selected_features

def time_series_feature_selection(X, y, num_features, window_size=252, max_lag=5):
    """
    Perform feature selection for time series data.
    
    Parameters:
    - X: DataFrame, feature matrix
    - y: DataFrame or Series, target variable(s)
    - num_features: int, number of features to select
    - window_size: int, size of the rolling window
    - max_lag: int, maximum number of lags for Granger Causality test
    
    Returns:
    - final_features: list of selected feature names
    """
    # Step 1: Rolling window feature selection
    rolling_features = rolling_feature_selection(X, y, window_size, num_features)
    
    # Step 2: Granger Causality test
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]  # Use the first target for Granger Causality test
    causal_features = granger_causality_test(X[rolling_features], y, max_lag)
    
    # Step 3: Correlation analysis
    uncorrelated_features = correlation_analysis(X[causal_features])
    
    # Step 4: Final feature selection
    final_features = uncorrelated_features[:num_features]
    
    return final_features
