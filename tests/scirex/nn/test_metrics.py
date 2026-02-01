"""
Test suite for metrics.

This module contains behavior-driven tests for all metrics
in scirex.nn.metrics.
"""

import pytest
import jax
import jax.numpy as jnp
from scirex.nn import metrics


class TestAccuracyMetric:
    """Test accuracy metric."""
    
    def test_accuracy_should_return_one_for_perfect_predictions(self):
        """Accuracy should return 1.0 when all predictions are correct."""
        predictions = jnp.array([0, 1, 2, 3, 4])
        targets = jnp.array([0, 1, 2, 3, 4])
        
        acc = metrics.accuracy(predictions, targets)
        assert jnp.isclose(acc, 1.0)
    
    def test_accuracy_should_return_zero_for_all_wrong_predictions(self):
        """Accuracy should return 0.0 when all predictions are wrong."""
        predictions = jnp.array([1, 2, 3, 4, 5])
        targets = jnp.array([0, 0, 0, 0, 0])
        
        acc = metrics.accuracy(predictions, targets)
        assert jnp.isclose(acc, 0.0)
    
    def test_accuracy_should_return_fraction_of_correct_predictions(self):
        """Accuracy should return the fraction of correct predictions."""
        predictions = jnp.array([0, 1, 2, 3, 4])
        targets = jnp.array([0, 1, 0, 3, 0])  # 3 out of 5 correct
        
        acc = metrics.accuracy(predictions, targets)
        assert jnp.isclose(acc, 0.6)  # 3/5 = 0.6
    
    def test_accuracy_should_handle_single_sample(self):
        """Accuracy should work with single sample."""
        predictions = jnp.array([1])
        targets = jnp.array([1])
        
        acc = metrics.accuracy(predictions, targets)
        assert jnp.isclose(acc, 1.0)
    
    def test_accuracy_should_handle_multidimensional_arrays(self):
        """Accuracy should work with multidimensional arrays."""
        predictions = jnp.array([[0, 1], [2, 3]])
        targets = jnp.array([[0, 1], [2, 0]])  # 3 out of 4 correct
        
        acc = metrics.accuracy(predictions, targets)
        assert jnp.isclose(acc, 0.75)
    
    def test_accuracy_should_be_between_zero_and_one(self):
        """Accuracy should always be in range [0, 1]."""
        predictions = jnp.array([0, 1, 2, 1, 0])
        targets = jnp.array([0, 0, 2, 1, 1])
        
        acc = metrics.accuracy(predictions, targets)
        assert 0.0 <= acc <= 1.0
    
    def test_accuracy_should_handle_float_predictions(self):
        """Accuracy should handle float predictions (after argmax)."""
        # Simulating class predictions after argmax
        predictions = jnp.array([0.0, 1.0, 2.0, 3.0])
        targets = jnp.array([0.0, 1.0, 2.0, 3.0])
        
        acc = metrics.accuracy(predictions, targets)
        assert jnp.isclose(acc, 1.0)
    
    def test_accuracy_should_be_symmetric_for_binary_classification(self):
        """For binary classification, accuracy should treat both classes equally."""
        # All class 0
        predictions_0 = jnp.array([0, 0, 0, 0])
        targets_0 = jnp.array([0, 0, 0, 0])
        
        # All class 1
        predictions_1 = jnp.array([1, 1, 1, 1])
        targets_1 = jnp.array([1, 1, 1, 1])
        
        acc_0 = metrics.accuracy(predictions_0, targets_0)
        acc_1 = metrics.accuracy(predictions_1, targets_1)
        
        assert jnp.isclose(acc_0, acc_1)
        assert jnp.isclose(acc_0, 1.0)


class TestAccuracyEdgeCases:
    """Test edge cases for accuracy metric."""
    
    def test_accuracy_should_handle_large_class_numbers(self):
        """Accuracy should work with large class numbers."""
        predictions = jnp.array([100, 200, 300])
        targets = jnp.array([100, 200, 300])
        
        acc = metrics.accuracy(predictions, targets)
        assert jnp.isclose(acc, 1.0)
    
    def test_accuracy_should_handle_negative_class_labels(self):
        """Accuracy should work with negative class labels."""
        predictions = jnp.array([-1, 0, 1])
        targets = jnp.array([-1, 0, 1])
        
        acc = metrics.accuracy(predictions, targets)
        assert jnp.isclose(acc, 1.0)
    
    def test_accuracy_should_handle_empty_arrays_gracefully(self):
        """Accuracy should handle empty arrays (though this is edge case)."""
        predictions = jnp.array([])
        targets = jnp.array([])
        
        # Mean of empty array is NaN in JAX
        acc = metrics.accuracy(predictions, targets)
        # This is expected to be NaN for empty arrays
        assert jnp.isnan(acc) or jnp.isclose(acc, 0.0)


class TestAccuracyBehavior:
    """Test behavioral properties of accuracy metric."""
    
    def test_accuracy_should_decrease_with_more_errors(self):
        """Accuracy should decrease as number of errors increases."""
        targets = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        # 1 error
        predictions_1_error = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0])
        # 3 errors
        predictions_3_errors = jnp.array([0, 1, 2, 3, 4, 5, 0, 0, 0, 9])
        
        acc_1 = metrics.accuracy(predictions_1_error, targets)
        acc_3 = metrics.accuracy(predictions_3_errors, targets)
        
        assert acc_1 > acc_3
    
    def test_accuracy_should_be_invariant_to_class_distribution(self):
        """Accuracy calculation should not depend on class distribution."""
        # Balanced classes
        predictions_balanced = jnp.array([0, 0, 1, 1])
        targets_balanced = jnp.array([0, 0, 1, 1])
        
        # Imbalanced classes
        predictions_imbalanced = jnp.array([0, 0, 0, 1])
        targets_imbalanced = jnp.array([0, 0, 0, 1])
        
        acc_balanced = metrics.accuracy(predictions_balanced, targets_balanced)
        acc_imbalanced = metrics.accuracy(predictions_imbalanced, targets_imbalanced)
        
        # Both should be 1.0 (all correct)
        assert jnp.isclose(acc_balanced, 1.0)
        assert jnp.isclose(acc_imbalanced, 1.0)
    
    def test_accuracy_should_handle_random_predictions_appropriately(self):
        """Accuracy for random predictions should be around 1/num_classes."""
        # For binary classification with random 50-50 predictions
        # Expected accuracy should be around 0.5
        key = jax.random.PRNGKey(42)
        targets = jnp.array([0, 1, 0, 1, 0, 1, 0, 1] * 100)  # Balanced
        predictions = jax.random.randint(key, shape=(len(targets),), minval=0, maxval=2)
        
        acc = metrics.accuracy(predictions, targets)
        # Should be around 0.5 for random binary predictions
        assert 0.3 < acc < 0.7  # Allow some variance
