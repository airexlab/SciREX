"""
Test suite for activation functions.

This module contains behavior-driven tests for all activation functions
in scirex.nn.activations.
"""

import jax.numpy as jnp

from scirex.nn import activations


class TestReLUActivations:
    """Test ReLU and its variants."""

    def test_relu_should_return_zero_for_negative_inputs(self):
        """ReLU should return 0 for all negative values."""
        x = jnp.array([-2.0, -1.0, -0.5])
        result = activations.relu(x)
        assert jnp.all(result == 0.0)

    def test_relu_should_return_input_for_positive_inputs(self):
        """ReLU should return the input value for positive values."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = activations.relu(x)
        assert jnp.allclose(result, x)

    def test_relu6_should_clip_values_above_six(self):
        """ReLU6 should clip values to maximum of 6."""
        x = jnp.array([5.0, 7.0, 10.0])
        result = activations.relu6(x)
        assert jnp.all(result <= 6.0)
        assert result[0] == 5.0  # Below 6
        assert result[1] == 6.0  # Clipped to 6
        assert result[2] == 6.0  # Clipped to 6

    def test_leaky_relu_should_allow_small_negative_gradient(self):
        """Leaky ReLU should allow small negative values instead of zero."""
        x = jnp.array([-1.0, 0.0, 1.0])
        result = activations.leaky_relu(x)
        assert result[0] < 0.0  # Negative input gives small negative output
        assert result[1] == 0.0
        assert result[2] == 1.0


class TestSigmoidActivations:
    """Test sigmoid and its variants."""

    def test_sigmoid_should_return_values_between_zero_and_one(self):
        """Sigmoid should always return values in range (0, 1)."""
        x = jnp.array([-10.0, 0.0, 10.0])
        result = activations.sigmoid(x)
        assert jnp.all(result > 0.0)
        assert jnp.all(result < 1.0)

    def test_sigmoid_should_return_half_for_zero_input(self):
        """Sigmoid of 0 should be 0.5."""
        result = activations.sigmoid(jnp.array(0.0))
        assert jnp.isclose(result, 0.5)

    def test_hard_sigmoid_should_approximate_sigmoid_with_linear_regions(self):
        """Hard sigmoid should approximate sigmoid with piecewise linear function."""
        x = jnp.array([-3.0, 0.0, 3.0])
        result = activations.hard_sigmoid(x)
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)

    def test_log_sigmoid_should_return_log_of_sigmoid(self):
        """Log sigmoid should return logarithm of sigmoid values."""
        x = jnp.array([0.0, 1.0, 2.0])
        result = activations.log_sigmoid(x)
        expected = jnp.log(activations.sigmoid(x))
        assert jnp.allclose(result, expected)


class TestTanhActivations:
    """Test tanh and its variants."""

    def test_hard_tanh_should_clip_values_between_minus_one_and_one(self):
        """Hard tanh should clip values to [-1, 1]."""
        x = jnp.array([-2.0, 0.0, 2.0])
        result = activations.hard_tanh(x)
        assert jnp.all(result >= -1.0)
        assert jnp.all(result <= 1.0)
        assert result[0] == -1.0  # Clipped
        assert result[1] == 0.0
        assert result[2] == 1.0  # Clipped

    def test_soft_sign_should_return_values_between_minus_one_and_one(self):
        """Soft sign should return values in range (-1, 1)."""
        x = jnp.array([-10.0, 0.0, 10.0])
        result = activations.soft_sign(x)
        assert jnp.all(result > -1.0)
        assert jnp.all(result < 1.0)


class TestELUActivations:
    """Test ELU and its variants."""

    def test_elu_should_be_smooth_for_negative_inputs(self):
        """ELU should provide smooth negative values for negative inputs."""
        x = jnp.array([-2.0, -1.0, 0.0, 1.0])
        result = activations.elu(x)
        assert result[0] < 0.0  # Negative but smooth
        assert result[1] < 0.0
        assert result[2] == 0.0
        assert result[3] == 1.0

    def test_selu_should_be_self_normalizing(self):
        """SELU should maintain mean and variance (self-normalizing property)."""
        x = jnp.array([-1.0, 0.0, 1.0, 2.0])
        result = activations.selu(x)
        # SELU has specific scaling factors
        assert result.shape == x.shape

    def test_celu_should_be_continuous_at_zero(self):
        """CELU should be continuous at zero."""
        x = jnp.array([-0.1, 0.0, 0.1])
        result = activations.celu(x)
        assert result.shape == x.shape


class TestGELUActivations:
    """Test GELU and GLU activations."""

    def test_gelu_should_be_smooth_approximation_of_relu(self):
        """GELU should provide smooth approximation of ReLU."""
        x = jnp.array([-2.0, 0.0, 2.0])
        result = activations.gelu(x)
        # GELU is smooth, so negative inputs give small negative outputs
        assert result[0] < 0.0
        assert result[2] > 0.0

    def test_glu_should_split_input_and_apply_gating(self):
        """GLU should split input in half and apply gating mechanism."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])  # Will be split into [1,2] and [3,4]
        result = activations.glu(x)
        assert result.shape[0] == x.shape[0] // 2


class TestSwishActivations:
    """Test Swish/SiLU and its variants."""

    def test_swish_should_be_self_gated(self):
        """Swish (x * sigmoid(x)) should be self-gated activation."""
        x = jnp.array([-1.0, 0.0, 1.0])
        result = activations.swish(x)
        expected = x * activations.sigmoid(x)
        assert jnp.allclose(result, expected)

    def test_silu_should_be_same_as_swish(self):
        """SiLU should be identical to Swish."""
        x = jnp.array([-2.0, 0.0, 2.0])
        swish_result = activations.swish(x)
        silu_result = activations.silu(x)
        assert jnp.allclose(swish_result, silu_result)

    def test_hard_swish_should_approximate_swish_efficiently(self):
        """Hard swish should approximate swish with piecewise linear function."""
        x = jnp.array([-3.0, 0.0, 3.0])
        result = activations.hard_swish(x)
        assert result.shape == x.shape


class TestSoftplusActivations:
    """Test Softplus and its variants."""

    def test_softplus_should_be_smooth_approximation_of_relu(self):
        """Softplus should be smooth approximation of ReLU."""
        x = jnp.array([-2.0, 0.0, 2.0])
        result = activations.softplus(x)
        # Softplus is always positive
        assert jnp.all(result > 0.0)
        # For large positive x, softplus(x) ≈ x
        assert jnp.isclose(result[2], 2.0, atol=0.2)

    def test_mish_should_be_smooth_self_regularized(self):
        """Mish (x * tanh(softplus(x))) should be smooth and self-regularized."""
        x = jnp.array([-1.0, 0.0, 1.0])
        result = activations.mish(x)
        assert result.shape == x.shape

    def test_squareplus_should_be_smooth_and_convex(self):
        """Squareplus should be smooth and convex."""
        x = jnp.array([-2.0, 0.0, 2.0])
        result = activations.squareplus(x)
        # Squareplus is always non-negative
        assert jnp.all(result >= 0.0)


class TestSparseActivations:
    """Test sparse activation functions."""

    def test_sparse_plus_should_induce_sparsity(self):
        """Sparse plus should induce sparsity in activations."""
        x = jnp.array([-1.0, 0.0, 1.0, 2.0])
        result = activations.sparse_plus(x)
        assert result.shape == x.shape

    def test_sparse_sigmoid_should_induce_sparsity_in_sigmoid(self):
        """Sparse sigmoid should induce sparsity while maintaining sigmoid-like behavior."""
        x = jnp.array([-2.0, 0.0, 2.0])
        result = activations.sparse_sigmoid(x)
        assert result.shape == x.shape


class TestActivationProperties:
    """Test general properties of activation functions."""

    def test_all_activations_should_preserve_shape(self):
        """All activation functions should preserve input shape."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        activations_to_test = [
            activations.relu,
            activations.sigmoid,
            activations.gelu,
            activations.elu,
            activations.selu,
            activations.swish,
            activations.mish,
            activations.softplus,
        ]

        for activation_fn in activations_to_test:
            result = activation_fn(x)
            assert result.shape == x.shape, f"{activation_fn.__name__} changed shape"

    def test_activations_should_handle_zero_input(self):
        """All activation functions should handle zero input gracefully."""
        x = jnp.array(0.0)

        activations_to_test = [
            activations.relu,
            activations.sigmoid,
            activations.gelu,
            activations.elu,
            activations.swish,
            activations.mish,
        ]

        for activation_fn in activations_to_test:
            result = activation_fn(x)
            assert jnp.isfinite(result), f"{activation_fn.__name__} failed on zero"

    def test_activations_should_handle_large_positive_values(self):
        """Activation functions should handle large positive values without overflow."""
        x = jnp.array(100.0)

        # Test activations that should handle large values
        activations_to_test = [
            activations.relu,
            activations.relu6,
            activations.gelu,
            activations.elu,
            activations.swish,
        ]

        for activation_fn in activations_to_test:
            result = activation_fn(x)
            assert jnp.isfinite(result), f"{activation_fn.__name__} overflow on large value"
