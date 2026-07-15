import numpy as np 

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None,
        self.errors_per_epoch = []

    def predict(self, X, y):
        output = np.dot(X, self.weights) + self.bias
        return np.where(output >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0 

        for _ in range(epochs):
            y_pred = predict(X, y)
            error = y_pred - y
            pass
