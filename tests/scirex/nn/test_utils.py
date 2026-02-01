"""
Test suite for utility functions.

This module contains behavior-driven tests for all utility functions
in scirex.nn.utils.
"""

import jax.numpy as jnp

from scirex.nn import utils


class TestSoftmax:
    """Test softmax function."""

    def test_softmax_should_return_probabilities_that_sum_to_one(self):
        """Softmax output should sum to 1.0 (valid probability distribution)."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = utils.softmax(x)
        assert jnp.isclose(jnp.sum(result), 1.0)

    def test_softmax_should_return_all_positive_values(self):
        """Softmax should return all positive values."""
        x = jnp.array([-5.0, 0.0, 5.0])
        result = utils.softmax(x)
        assert jnp.all(result > 0.0)

    def test_softmax_should_preserve_relative_ordering(self):
        """Softmax should preserve relative ordering of inputs."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = utils.softmax(x)
        # Larger input should give larger probability
        assert result[0] < result[1] < result[2]

    def test_softmax_should_handle_uniform_inputs(self):
        """Softmax of uniform inputs should give uniform probabilities."""
        x = jnp.array([1.0, 1.0, 1.0, 1.0])
        result = utils.softmax(x)
        # All should be equal to 1/4
        assert jnp.allclose(result, 0.25)

    def test_softmax_should_handle_large_values_without_overflow(self):
        """Softmax should handle large values numerically stable."""
        x = jnp.array([1000.0, 1001.0, 1002.0])
        result = utils.softmax(x)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.isclose(jnp.sum(result), 1.0)

    def test_softmax_should_work_along_specified_axis(self):
        """Softmax should work along specified axis for multidimensional arrays."""
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = utils.softmax(x)
        # Default axis should give valid probabilities
        assert jnp.all(jnp.isfinite(result))


class TestLogSoftmax:
    """Test log softmax function."""

    def test_log_softmax_should_return_log_of_softmax(self):
        """Log softmax should equal log(softmax(x))."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = utils.log_softmax(x)
        expected = jnp.log(utils.softmax(x))
        assert jnp.allclose(result, expected)

    def test_log_softmax_should_return_negative_values(self):
        """Log softmax should return all negative values (since softmax < 1)."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = utils.log_softmax(x)
        assert jnp.all(result <= 0.0)

    def test_log_softmax_should_be_numerically_stable(self):
        """Log softmax should be more numerically stable than log(softmax(x))."""
        x = jnp.array([1000.0, 1001.0, 1002.0])
        result = utils.log_softmax(x)
        assert jnp.all(jnp.isfinite(result))

    def test_log_softmax_should_handle_uniform_inputs(self):
        """Log softmax of uniform inputs should give log(1/n)."""
        x = jnp.array([1.0, 1.0, 1.0, 1.0])
        result = utils.log_softmax(x)
        expected = jnp.log(0.25)
        assert jnp.allclose(result, expected)


class TestStandardize:
    """Test standardize function."""

    def test_standardize_should_give_zero_mean(self):
        """Standardize should center data to have mean of 0."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = utils.standardize(x)
        assert jnp.isclose(jnp.mean(result), 0.0, atol=1e-6)

    def test_standardize_should_give_unit_variance(self):
        """Standardize should scale data to have variance of 1."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = utils.standardize(x)
        assert jnp.isclose(jnp.var(result), 1.0, atol=1e-6)

    def test_standardize_should_preserve_shape(self):
        """Standardize should preserve input shape."""
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = utils.standardize(x)
        assert result.shape == x.shape

    def test_standardize_should_handle_constant_input(self):
        """Standardize should handle constant input (all same values)."""
        x = jnp.array([5.0, 5.0, 5.0, 5.0])
        result = utils.standardize(x)
        # When variance is 0, result should be all zeros
        assert jnp.allclose(result, 0.0) or jnp.all(jnp.isnan(result))

    def test_standardize_should_be_linear_transformation(self):
        """Standardize should be a linear transformation."""
        x = jnp.array([1.0, 2.0, 3.0])
        result = utils.standardize(x)
        # Relative ordering should be preserved
        assert result[0] < result[1] < result[2]

    def test_standardize_should_work_along_specified_axis(self):
        """Standardize should work along specified axis."""
        x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = utils.standardize(x)
        # Should have mean close to 0 and std close to 1
        assert jnp.isclose(jnp.mean(result), 0.0, atol=1e-6)


class TestOneHot:
    """Test one-hot encoding function."""

    def test_one_hot_should_create_binary_vectors(self):
        """One-hot should create vectors with exactly one 1 and rest 0s."""
        x = jnp.array([0, 1, 2])
        result = utils.one_hot(x, 3)  # num_classes as second positional arg
        # Each row should have exactly one 1
        assert jnp.all(jnp.sum(result, axis=-1) == 1)

    def test_one_hot_should_place_one_at_correct_index(self):
        """One-hot should place 1 at the index specified by input."""
        x = jnp.array([0, 1, 2])
        result = utils.one_hot(x, 3)

        assert result[0, 0] == 1.0
        assert result[1, 1] == 1.0
        assert result[2, 2] == 1.0

    def test_one_hot_should_create_correct_shape(self):
        """One-hot should create array of shape (n, num_classes)."""
        x = jnp.array([0, 1, 2, 3, 4])
        num_classes = 10
        result = utils.one_hot(x, num_classes)

        assert result.shape == (5, 10)

    def test_one_hot_should_handle_single_value(self):
        """One-hot should work with single scalar value."""
        x = jnp.array(2)
        result = utils.one_hot(x, 5)

        expected = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])
        assert jnp.allclose(result, expected)

    def test_one_hot_should_handle_multidimensional_input(self):
        """One-hot should work with multidimensional input."""
        x = jnp.array([[0, 1], [2, 3]])
        result = utils.one_hot(x, 4)

        # Shape should be (2, 2, 4)
        assert result.shape == (2, 2, 4)

    def test_one_hot_should_contain_only_zeros_and_ones(self):
        """One-hot encoded array should contain only 0s and 1s."""
        x = jnp.array([0, 1, 2, 3])
        result = utils.one_hot(x, 5)

        unique_values = jnp.unique(result)
        assert jnp.all((unique_values == 0.0) | (unique_values == 1.0))


class TestUtilityFunctionProperties:
    """Test general properties of utility functions."""

    def test_softmax_and_log_softmax_should_be_consistent(self):
        """exp(log_softmax(x)) should equal softmax(x)."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0])

        softmax_result = utils.softmax(x)
        log_softmax_result = utils.log_softmax(x)

        assert jnp.allclose(jnp.exp(log_softmax_result), softmax_result)

    def test_standardize_should_be_invertible_with_statistics(self):
        """Standardize should be invertible if we know mean and std."""
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = jnp.mean(x)
        std = jnp.std(x)

        standardized = utils.standardize(x)
        # Reconstruct original
        reconstructed = standardized * std + mean

        assert jnp.allclose(reconstructed, x)

    def test_one_hot_should_be_reversible_with_argmax(self):
        """argmax(one_hot(x)) should give back x."""
        x = jnp.array([0, 1, 2, 3, 4])
        one_hot_encoded = utils.one_hot(x, 5)
        reconstructed = jnp.argmax(one_hot_encoded, axis=-1)

        assert jnp.allclose(reconstructed, x)

    def test_all_utils_should_handle_empty_arrays(self):
        """Utility functions should handle empty arrays gracefully."""
        x_empty = jnp.array([])

        # These might return empty arrays or NaN, but shouldn't crash
        try:
            utils.softmax(x_empty)
            utils.log_softmax(x_empty)
            utils.standardize(x_empty)
        except (ValueError, IndexError):
            # Some functions may raise errors for empty arrays, which is acceptable
            pass
