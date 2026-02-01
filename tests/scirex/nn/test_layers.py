"""
Unit tests for scirex.nn.layers module
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from scirex.nn.layers.convolution import Conv1d, Conv2d, Conv3d
from scirex.nn.layers.dropout import Dropout
from scirex.nn.layers.embeddings import Embedding
from scirex.nn.layers.linear import Linear
from scirex.nn.layers.mlp import MLP
from scirex.nn.layers.normalisation import BatchNorm, LayerNorm, RMSNorm
from scirex.nn.layers.recurrent import GRUCell, LSTMCell
from scirex.nn.layers.sequential import Sequential


class TestLinear:
    """Tests for Linear layer"""

    def test_linear_creation(self):
        """Test Linear layer creation"""
        rngs = nnx.Rngs(0)
        layer = Linear(in_features=10, out_features=5, rngs=rngs)
        assert isinstance(layer, Linear)

    def test_linear_forward(self):
        """Test Linear layer forward pass"""
        rngs = nnx.Rngs(0)
        layer = Linear(in_features=10, out_features=5, rngs=rngs)
        x = jnp.ones((3, 10))
        output = layer(x)
        assert output.shape == (3, 5)

    def test_linear_no_bias(self):
        """Test Linear layer without bias"""
        rngs = nnx.Rngs(0)
        layer = Linear(in_features=10, out_features=5, use_bias=False, rngs=rngs)
        x = jnp.ones((3, 10))
        output = layer(x)
        assert output.shape == (3, 5)


class TestMLP:
    """Tests for MLP layer"""

    def test_mlp_creation(self):
        """Test MLP creation"""
        rngs = nnx.Rngs(0)
        mlp = MLP(in_size=10, out_size=2, hidden_size=16, depth=2, rngs=rngs)
        assert isinstance(mlp, MLP)

    def test_mlp_forward(self):
        """Test MLP forward pass"""
        rngs = nnx.Rngs(0)
        mlp = MLP(in_size=10, out_size=2, hidden_size=16, depth=2, rngs=rngs)
        x = jnp.ones((5, 10))
        output = mlp(x)
        assert output.shape == (5, 2)

    def test_mlp_different_depths(self):
        """Test MLP with different depths"""
        rngs = nnx.Rngs(0)
        for depth in [1, 2, 3, 5]:
            mlp = MLP(in_size=10, out_size=2, hidden_size=16, depth=depth, rngs=rngs)
            x = jnp.ones((3, 10))
            output = mlp(x)
            assert output.shape == (3, 2)


class TestSequential:
    """Tests for Sequential layer"""

    def test_sequential_creation(self):
        """Test Sequential creation"""
        rngs = nnx.Rngs(0)
        layers = [Linear(in_features=10, out_features=5, rngs=rngs), Linear(in_features=5, out_features=2, rngs=rngs)]
        seq = Sequential(layers)
        assert isinstance(seq, Sequential)

    def test_sequential_forward(self):
        """Test Sequential forward pass"""
        rngs = nnx.Rngs(0)
        layers = [Linear(in_features=10, out_features=5, rngs=rngs), Linear(in_features=5, out_features=2, rngs=rngs)]
        seq = Sequential(layers)
        x = jnp.ones((3, 10))
        output = seq(x)
        assert output.shape == (3, 2)


class TestNormalization:
    """Tests for normalization layers"""

    def test_layer_norm(self):
        """Test LayerNorm"""
        rngs = nnx.Rngs(0)
        layer = LayerNorm(num_features=10, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (3, 10))
        output = layer(x)
        assert output.shape == x.shape

    def test_batch_norm(self):
        """Test BatchNorm"""
        rngs = nnx.Rngs(0)
        layer = BatchNorm(num_features=10, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (3, 10))
        output = layer(x)
        assert output.shape == x.shape

    def test_rms_norm(self):
        """Test RMSNorm"""
        rngs = nnx.Rngs(0)
        layer = RMSNorm(num_features=10, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (3, 10))
        output = layer(x)
        assert output.shape == x.shape


class TestConvolution:
    """Tests for convolution layers"""

    def test_conv1d(self):
        """Test Conv1d"""
        rngs = nnx.Rngs(0)
        layer = Conv1d(in_features=3, out_features=8, kernel_size=3, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 10, 3))  # (batch, length, channels)
        output = layer(x)
        assert output.shape[0] == 2
        assert output.shape[2] == 8

    def test_conv2d(self):
        """Test Conv2d"""
        rngs = nnx.Rngs(0)
        layer = Conv2d(in_features=3, out_features=16, kernel_size=(3, 3), rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 28, 28, 3))  # (batch, H, W, channels)
        output = layer(x)
        assert output.shape[0] == 2
        assert output.shape[3] == 16

    def test_conv3d(self):
        """Test Conv3d"""
        rngs = nnx.Rngs(0)
        layer = Conv3d(in_features=3, out_features=8, kernel_size=(3, 3, 3), rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 8, 3))  # (batch, D, H, W, channels)
        output = layer(x)
        assert output.shape[0] == 2
        assert output.shape[4] == 8


class TestDropout:
    """Tests for Dropout layer"""

    def test_dropout_creation(self):
        """Test Dropout creation"""
        rngs = nnx.Rngs(0)
        layer = Dropout(rate=0.5, rngs=rngs)
        assert isinstance(layer, Dropout)

    def test_dropout_forward(self):
        """Test Dropout forward pass"""
        rngs = nnx.Rngs(0)
        layer = Dropout(rate=0.5, rngs=rngs)
        x = jnp.ones((10, 20))
        output = layer(x)
        assert output.shape == x.shape


class TestEmbedding:
    """Tests for Embedding layer"""

    def test_embedding_creation(self):
        """Test Embedding creation"""
        rngs = nnx.Rngs(0)
        layer = Embedding(num_embeddings=100, features=16, rngs=rngs)
        assert isinstance(layer, Embedding)

    def test_embedding_forward(self):
        """Test Embedding forward pass"""
        rngs = nnx.Rngs(0)
        layer = Embedding(num_embeddings=100, features=16, rngs=rngs)
        x = jnp.array([[1, 2, 3], [4, 5, 6]])  # (batch, seq_len)
        output = layer(x)
        assert output.shape == (2, 3, 16)


class TestRecurrent:
    """Tests for recurrent layers"""

    def test_gru_cell(self):
        """Test GRUCell"""
        rngs = nnx.Rngs(0)
        cell = GRUCell(in_features=10, hidden_features=20, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (3, 10))
        carry = jax.random.normal(jax.random.PRNGKey(1), (3, 20))
        new_carry, output = cell(carry, x)
        assert new_carry.shape == (3, 20)
        assert output.shape == (3, 20)

    def test_lstm_cell(self):
        """Test LSTMCell"""
        rngs = nnx.Rngs(0)
        cell = LSTMCell(in_features=10, hidden_features=20, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(0), (3, 10))
        carry = (
            jax.random.normal(jax.random.PRNGKey(1), (3, 20)),  # h
            jax.random.normal(jax.random.PRNGKey(2), (3, 20)),  # c
        )
        new_carry, output = cell(carry, x)
        assert new_carry[0].shape == (3, 20)
        assert new_carry[1].shape == (3, 20)
        assert output.shape == (3, 20)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
