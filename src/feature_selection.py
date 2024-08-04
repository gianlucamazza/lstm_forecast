import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import numpy as np

def recursive_feature_elimination(X, y, num_features):
    """
    Perform recursive feature elimination for multiple targets.

    Parameters:
    - X: DataFrame, feature matrix
    - y: DataFrame, target matrix with multiple columns
    - num_features: int, number of features to select

    Returns:
    - selected_features: list of selected feature names
    """
    # Ensure y is a DataFrame
    if isinstance(y, pd.Series):
        y = y.to_frame()

    # Initialize model
    model = LinearRegression()
    rfe = RFE(model, num_features)

    # Fit RFE for each target and aggregate results
    support_matrix = np.zeros((X.shape[1], y.shape[1]))

    for i, target in enumerate(y.columns):
        rfe.fit(X, y[target])
        support_matrix[:, i] = rfe.support_

    # Aggregate support results by considering features selected in majority of targets
    support_scores = np.sum(support_matrix, axis=1)
    feature_indices = np.argsort(support_scores)[-num_features:]

    selected_features = X.columns[feature_indices].tolist()
    return selected_features

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