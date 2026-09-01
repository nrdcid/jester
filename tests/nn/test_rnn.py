"""Tests for the recurrent neural-network layer."""

import numpy as np
import pytest

from jester.nn import RNN


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_size": 0, "hidden_size": 2},
        {"input_size": 2, "hidden_size": 0},
        {"input_size": 2, "hidden_size": 2, "n_layers": 0},
    ],
)
def test_constructor_rejects_non_positive_dimensions(kwargs):
    with pytest.raises(ValueError):
        RNN(**kwargs)


def test_forward_returns_sequence_and_saves_final_hidden_state():
    rnn = RNN(input_size=3, hidden_size=4, n_layers=2)
    x = np.zeros((5, 7, 3))

    output = rnn.forward(x)

    assert isinstance(output, np.ndarray)
    assert output.shape == (5, 7, 4)
    assert rnn.last_hidden_state.shape == (2, 5, 4)
    assert np.allclose(rnn.last_hidden_state[1], output[:, -1, :])


def test_forward_accepts_an_initial_hidden_state():
    rnn = RNN(input_size=1, hidden_size=1)
    rnn.W_xh[0][:] = 0.0
    rnn.W_hh[0][:] = 1.0
    rnn.b[0][:] = 0.0
    x = np.zeros((1, 3, 1))

    from_zero = rnn.forward(x, h=np.zeros((1, 1, 1)))
    from_one = rnn.forward(x, h=np.ones((1, 1, 1)))

    assert np.allclose(from_zero, 0.0)
    assert np.all(from_one > 0.0)
    assert np.allclose(rnn.last_hidden_state[0], from_one[:, -1, :])


def test_forward_rejects_an_initial_hidden_state_with_the_wrong_shape():
    rnn = RNN(input_size=2, hidden_size=3, n_layers=2)
    x = np.zeros((4, 5, 2))

    with pytest.raises(ValueError, match="expected"):
        rnn.forward(x, h=np.zeros((2, 4, 2)))
