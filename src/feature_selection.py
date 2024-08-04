import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
import numpy as np

def recursive_feature_elimination(X, y, num_features):
    model = LinearRegression()
    rfe = RFE(model, num_features)
    fit = rfe.fit(X, y)
    selected_features = X.columns[fit.support_].tolist()
    return selected_features

def correlation_analysis(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    selected_features = df.columns.difference(to_drop).tolist()
    return selected_features
