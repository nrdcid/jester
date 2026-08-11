"""Tests for the neural-network container."""

from jester.nn import MSE, Network, ReLU, SGD


def test_add_is_chainable_and_keeps_layers_in_order():
    network = Network(optimizer=SGD(learning_rate=0.1), loss=MSE())
    first = ReLU()
    second = ReLU()

    assert network.add(first).add(second) is network
    assert network.layers == [first, second]


def test_add_layer_remains_a_compatibility_alias():
    network = Network(optimizer=SGD(learning_rate=0.1), loss=MSE())
    layer = ReLU()

    network.add_layer(layer)

    assert network.layers == [layer]
