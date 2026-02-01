"""
Unit tests for scirex.nn.base module (Network and Model classes)
"""

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from scirex.nn.base import Model, Network
from scirex.nn.layers import MLP, Linear


class SimpleNetwork(Network):
    """Simple test network"""

    def __init__(self, rngs: nnx.Rngs):
        self.linear = Linear(in_features=10, out_features=1, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


class TestNetwork:
    """Tests for Network base class"""

    def test_network_creation(self):
        """Test that a Network can be created"""
        rngs = nnx.Rngs(0)
        net = SimpleNetwork(rngs)
        assert isinstance(net, Network)
        assert isinstance(net, nnx.Module)

    def test_network_forward_pass(self):
        """Test forward pass through network"""
        rngs = nnx.Rngs(0)
        net = SimpleNetwork(rngs)
        x = jnp.ones((5, 10))
        output = net(x)
        assert output.shape == (5, 1)

    def test_network_with_mlp(self):
        """Test network with MLP layer"""
        rngs = nnx.Rngs(42)
        net = MLP(in_size=10, out_size=2, hidden_size=16, depth=2, rngs=rngs)
        x = jnp.ones((3, 10))
        output = net(x)
        assert output.shape == (3, 2)


class TestModel:
    """Tests for Model class"""

    def test_model_creation(self):
        """Test that a Model can be created"""
        rngs = nnx.Rngs(0)
        net = SimpleNetwork(rngs)

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        model = Model(net=net, optimizer=optax.adam(0.01), loss_fn=mse_loss, metrics=[])
        assert isinstance(model, Model)
        assert model.net is net

    def test_model_training(self):
        """Test that model can train and loss decreases"""
        rngs = nnx.Rngs(123)
        net = SimpleNetwork(rngs)

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        model = Model(net=net, optimizer=optax.adam(0.1), loss_fn=mse_loss, metrics=[])

        # Create simple dataset
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (50, 10))
        y = jax.random.normal(key, (50, 1))

        # Train
        history = model.fit(x, y, num_epochs=3, batch_size=10)

        # Check that loss decreased
        assert len(history) == 3
        assert history[-1]["loss"] < history[0]["loss"]

    def test_model_evaluate(self):
        """Test model evaluation"""
        rngs = nnx.Rngs(456)
        net = SimpleNetwork(rngs)

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        def mae_metric(pred, target):
            return jnp.mean(jnp.abs(pred - target))

        model = Model(net=net, optimizer=optax.adam(0.01), loss_fn=mse_loss, metrics=[mae_metric])

        # Create test data
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (20, 10))
        y = jax.random.normal(key, (20, 1))

        # Evaluate
        loss, metrics = model.evaluate(x, y)

        assert isinstance(loss, (float, jnp.ndarray))
        assert len(metrics) == 1
        assert isinstance(metrics[0], (float, jnp.ndarray))

    def test_model_predict(self):
        """Test model prediction"""
        rngs = nnx.Rngs(789)
        net = SimpleNetwork(rngs)

        def mse_loss(pred, target):
            return jnp.mean((pred - target) ** 2)

        model = Model(net=net, optimizer=optax.adam(0.01), loss_fn=mse_loss, metrics=[])

        # Create test data
        x = jnp.ones((5, 10))

        # Predict
        predictions = model.predict(x)

        assert predictions.shape == (5, 1)
        assert isinstance(predictions, jnp.ndarray)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
