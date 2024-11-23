from sklearn.preprocessing import StandardScaler
def scale_features(X):
    """
    Scales the features of the data
    Args:
        data: pd.DataFrame
    Returns:
        data: pd.DataFrame
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
