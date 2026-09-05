def min_max_scale(X):
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denominator = X_max - X_min

    return (X - X_min) / denominator