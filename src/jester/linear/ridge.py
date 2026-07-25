import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


class Ridge(object):

    def __init__(self, alpha, normalize=False):
        self.alpha = alpha  # our tuning / regularization parameter
        self.coefficients = None  # our weights vector, w (in formulae above)
        self.intercept = None  # our intercept parameter, b (in formulae above)
        self.normalize = normalize  # boolean whether to normalize the features or not
        self.scaler = StandardScaler()  # method by which to normalize the features (depends on self.normalize)
        self.model = linear_model.Ridge(alpha=alpha, fit_intercept=True)

    def fit(self, X, y):
        if self.normalize:
            X = self.scaler.fit_transform(X)

        self.model.fit(X, y)
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        num_nonzero_coefs = np.sum(self.coefficients != 0)
        coef_norm = np.sqrt(np.sum(self.coefficients ** 2))

        return num_nonzero_coefs, coef_norm

    def evaluate(self, X, y):
        if self.normalize:
            X = self.scaler.transform(X)    
        y_pred = self.model.predict(X)
        root_mean_squared_error = np.sqrt(np.mean((y - y_pred) ** 2))

        return root_mean_squared_error