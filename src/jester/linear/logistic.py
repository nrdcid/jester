import numpy as np

class LogisticRegression:
    """
    Logistic Regression Classifier via Gradient Descent.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size for gradient descent.
    n_iterations : int, default=1000
        Number of iterations for the optimization loop.
    
    Attributes
    ----------
    weights : np.ndarray
        Coefficients of the features.
    bias : float
        Intercept term.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        """
        Compute the sigmoid of z
        """
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        Fit the model using gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training vector
        y : np.ndarray of shape (n_samples,)
            Target vector (0 or 1)
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input features

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
    
    def predict_proba(self, X):
        """
        Predict probability estimates.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        probs : np.ndarray of shape (n_samples,)
            Returns the probability of the sample for each class in the model.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)
