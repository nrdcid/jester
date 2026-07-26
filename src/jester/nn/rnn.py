import numpy as np

from .layers import Layer
from .parameter import Parameter


def _scale(fan_in, fan_out):
    return (fan_in * fan_out) ** 0.5


class RNN(Layer):
    """
    Multi-layer recurrent neural network (Elman / "vanilla" RNN).

    Compatible with :class:`~jester.nn.network.Network` via the Layer interface:
    ``parameters`` holds all trainable weights so SGD/Adam can update them.

    Attributes:
        input_size (int): Feature dim of each time step
        hidden_size (int): Hidden state size (same for every layer)
        n_layers (int): Number of stacked recurrent layers
        bias (bool): Whether layers include a bias term
        W_xh (list[Parameter]): Input-to-hidden weights per layer
        W_hh (list[Parameter]): Hidden-to-hidden weights per layer
        b (list[Parameter] | None): Bias per layer when ``bias`` is True
    """

    def __init__(self, input_size, hidden_size, n_layers=1, bias=True, name=""):
        """
        Args:
            input_size: Size of the input features
            hidden_size: Size of the hidden state
            n_layers: Number of stacked recurrent layers
            bias: Whether to include bias terms
            name: Optional layer identifier
        """
        super().__init__(name)
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be >= 1")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.bias = bool(bias)

        self.W_xh = []
        self.W_hh = []
        self.b = [] if self.bias else None
        self.parameters = []

        for layer in range(self.n_layers):
            in_dim = self.input_size if layer == 0 else self.hidden_size
            w_xh = Parameter(
                np.random.randn(in_dim, self.hidden_size) / _scale(in_dim, self.hidden_size),
                name=f"{name}.W_xh[{layer}]" if name else f"W_xh[{layer}]",
            )
            w_hh = Parameter(
                np.random.randn(self.hidden_size, self.hidden_size)
                / _scale(self.hidden_size, self.hidden_size),
                name=f"{name}.W_hh[{layer}]" if name else f"W_hh[{layer}]",
            )
            self.W_xh.append(w_xh)
            self.W_hh.append(w_hh)
            self.parameters.extend([w_xh, w_hh])

            if self.bias:
                b = Parameter(
                    np.random.randn(1, self.hidden_size) / self.hidden_size**0.5,
                    name=f"{name}.b[{layer}]" if name else f"b[{layer}]",
                )
                self.b.append(b)
                self.parameters.append(b)

    def init_hidden(self, batch_size):
        """Return zero hidden state of shape (n_layers, batch_size, hidden_size)."""
        return np.zeros((self.n_layers, batch_size, self.hidden_size))

    def forward(self, x, h=None):
        """
        Forward pass through the RNN.

        Args:
            x: Input of shape (batch_size, seq_length, input_size)
            h: Optional initial hidden state of shape
               (n_layers, batch_size, hidden_size). Zeros if omitted.

        Returns:
            tuple:
                output — top-layer hidden states for every time step,
                shape (batch_size, seq_length, hidden_size)
                h_n — final hidden state per layer,
                shape (n_layers, batch_size, hidden_size)
        """
        x = np.asarray(x, dtype=float)
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (batch, seq, input_size), got ndim={x.ndim}"
            )
        batch_size, seq_length, feat = x.shape
        if feat != self.input_size:
            raise ValueError(
                f"x feature dim {feat} != input_size {self.input_size}"
            )

        if h is None:
            h = self.init_hidden(batch_size)
        else:
            h = np.asarray(h, dtype=float)
            expected = (self.n_layers, batch_size, self.hidden_size)
            if h.shape != expected:
                raise ValueError(f"h shape {h.shape} != expected {expected}")

        # Cache: per layer, list of (x_t, h_prev, h_t) over time for BPTT
        cache = [[] for _ in range(self.n_layers)]
        h_n = np.empty_like(h)
        layer_input = x  # (batch, seq, in_dim) for current layer

        for layer in range(self.n_layers):
            h_prev = h[layer]
            outs = []
            W_xh = self.W_xh[layer]
            W_hh = self.W_hh[layer]
            b = self.b[layer] if self.bias else None

            for t in range(seq_length):
                x_t = layer_input[:, t, :]
                pre = x_t @ W_xh + h_prev @ W_hh
                if b is not None:
                    pre = pre + b
                h_t = np.tanh(pre)
                cache[layer].append((x_t, h_prev, h_t))
                outs.append(h_t)
                h_prev = h_t

            h_n[layer] = h_prev
            layer_input = np.stack(outs, axis=1)

        self.saved_arrays = [cache, batch_size, seq_length]
        return layer_input, h_n

    def backward(self, grad_output, grad_h=None):
        """
        Backpropagation through time (and through stacked layers).

        Args:
            grad_output: Grad of loss w.r.t. top-layer outputs,
                shape (batch_size, seq_length, hidden_size)
            grad_h: Optional grad of loss w.r.t. final hidden states,
                shape (n_layers, batch_size, hidden_size)

        Returns:
            Grad of loss w.r.t. input x, shape (batch_size, seq_length, input_size)
        """
        if not self.saved_arrays:
            raise RuntimeError("backward() called before forward()")

        cache, batch_size, seq_length = self.saved_arrays
        grad_output = np.asarray(grad_output, dtype=float)
        if grad_output.shape != (batch_size, seq_length, self.hidden_size):
            raise ValueError(
                f"grad_output shape {grad_output.shape} != "
                f"({batch_size}, {seq_length}, {self.hidden_size})"
            )

        if grad_h is None:
            grad_h = np.zeros((self.n_layers, batch_size, self.hidden_size))
        else:
            grad_h = np.asarray(grad_h, dtype=float)
            expected = (self.n_layers, batch_size, self.hidden_size)
            if grad_h.shape != expected:
                raise ValueError(f"grad_h shape {grad_h.shape} != expected {expected}")

        for p in self.parameters:
            p.zero_gradient()

        # Gradient flowing into the top layer's sequence outputs
        d_layer_out = grad_output.copy()

        for layer in range(self.n_layers - 1, -1, -1):
            W_xh = self.W_xh[layer]
            W_hh = self.W_hh[layer]
            b = self.b[layer] if self.bias else None

            dW_xh = np.zeros_like(W_xh)
            dW_hh = np.zeros_like(W_hh)
            db = np.zeros_like(b) if b is not None else None

            # Grad w.r.t. this layer's inputs over time (becomes grad for layer below)
            d_inputs = []
            dh_next = grad_h[layer]  # from final hidden + BPTT chain

            for t in range(seq_length - 1, -1, -1):
                x_t, h_prev, h_t = cache[layer][t]
                dh = d_layer_out[:, t, :] + dh_next
                dpre = dh * (1.0 - h_t**2)

                dW_xh += x_t.T @ dpre
                dW_hh += h_prev.T @ dpre
                if db is not None:
                    db += np.sum(dpre, axis=0, keepdims=True)

                d_inputs.append(dpre @ W_xh.T)
                dh_next = dpre @ W_hh.T

            d_inputs.reverse()
            d_layer_out = np.stack(d_inputs, axis=1)

            W_xh.gradient = dW_xh
            W_hh.gradient = dW_hh
            if b is not None:
                b.gradient = db

        self.saved_arrays = []
        return d_layer_out
