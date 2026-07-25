from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

class LinearKernel(object):
    def compute(self, x1, x2):
        return np.dot(x1, x2.T)


class RadialKernel(object):

    def __init__(self, gamma):
        self.gamma = gamma

    def compute(self, x1, x2):
        return np.exp(-self.gamma * euclidean_distances(x1, x2, squared=True))


class PolynomialKernel(object):

    def __init__(self, c, p):
        self.c = c
        self.p = p

    def compute(self, x1, x2):
        return (np.dot(x1, x2.T) + self.c) ** self.p