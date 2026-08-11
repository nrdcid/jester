"""Tests for the NumPy neural-network layers."""

import numpy as np

from jester.nn import Dense, ReLU, Sigmoid


def test_sigmoid_forward():
    x = np.array([[-0.4838731, 0.08083195], [0.93456167, -0.50316134]])
    expected = np.array([[0.38133797, 0.52019699], [0.71799983, 0.37679803]])
    assert np.allclose(Sigmoid().forward(x), expected)


def test_sigmoid_backward():
    x = np.array([[-0.4838731, 0.08083195], [0.93456167, -0.50316134]])
    grad = np.array([[0.19960269, 0.20993069], [-0.85814751, -0.41418101]])
    layer = Sigmoid()
    layer.forward(x)
    expected = np.array([[0.04709013, 0.05239704], [-0.17375434, -0.09725851]])
    assert np.allclose(layer.backward(grad), expected)


def test_relu_forward_and_backward():
    x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    grad = np.ones_like(x)
    layer = ReLU()
    assert np.array_equal(layer.forward(x), [[0.0, 2.0], [3.0, 0.0]])
    assert np.array_equal(layer.backward(grad), [[0.0, 1.0], [1.0, 0.0]])


def test_dense_forward_and_backward_shapes():
    np.random.seed(42)
    layer = Dense(2, 2)
    x = np.ones((3, 2))
    output = layer.forward(x)
    grad_input = layer.backward(np.ones_like(output))
    assert output.shape == (3, 2)
    assert grad_input.shape == x.shape
    assert layer.weights.gradient.shape == layer.weights.shape
    assert layer.bias.gradient.shape == layer.bias.shape
