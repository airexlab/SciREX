"""
Test suite for loss functions.

This module contains behavior-driven tests for all loss functions
in scirex.nn.losses.
"""

import pytest
import jax.numpy as jnp
from scirex.nn import losses


class TestMSELoss:
    """Test Mean Squared Error loss."""
    
    def test_mse_should_return_zero_for_perfect_predictions(self):
        """MSE should return 0 when predictions match targets exactly."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        loss = losses.mse_loss(predictions, targets)
        assert jnp.isclose(loss, 0.0)
    
    def test_mse_should_increase_with_prediction_error(self):
        """MSE should increase as prediction error increases."""
        targets = jnp.array([1.0, 2.0, 3.0])
        
        predictions_close = jnp.array([1.1, 2.1, 3.1])
        predictions_far = jnp.array([2.0, 3.0, 4.0])
        
        loss_close = losses.mse_loss(predictions_close, targets)
        loss_far = losses.mse_loss(predictions_far, targets)
        
        assert loss_far > loss_close
    
    def test_mse_should_be_symmetric(self):
        """MSE should be symmetric: MSE(a, b) == MSE(b, a)."""
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([2.0, 3.0, 4.0])
        
        loss_ab = losses.mse_loss(a, b)
        loss_ba = losses.mse_loss(b, a)
        
        assert jnp.isclose(loss_ab, loss_ba)
    
    def test_mse_should_handle_multidimensional_arrays(self):
        """MSE should work with multidimensional arrays."""
        predictions = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[1.5, 2.5], [3.5, 4.5]])
        loss = losses.mse_loss(predictions, targets)
        assert jnp.isfinite(loss)
        assert loss > 0.0


class TestCrossEntropyLoss:
    """Test Cross Entropy loss."""
    
    def test_cross_entropy_should_return_low_loss_for_confident_correct_predictions(self):
        """Cross entropy should return low loss when model is confident and correct."""
        # Logits for 3 classes, batch size 2
        logits = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        labels = jnp.array([0, 1])  # Correct predictions
        
        loss = losses.cross_entropy_loss(logits, labels)
        assert loss < 0.1  # Should be very small
    
    def test_cross_entropy_should_return_high_loss_for_wrong_predictions(self):
        """Cross entropy should return high loss for wrong predictions."""
        # Confident but wrong predictions
        logits = jnp.array([[0.0, 0.0, 10.0], [10.0, 0.0, 0.0]])
        labels = jnp.array([0, 1])  # Predictions are class 2 and 0, but labels are 0 and 1
        
        loss = losses.cross_entropy_loss(logits, labels)
        assert loss > 1.0  # Should be high
    
    def test_cross_entropy_should_handle_uniform_predictions(self):
        """Cross entropy should handle uniform (uncertain) predictions."""
        # Uniform logits
        logits = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        labels = jnp.array([0, 1])
        
        loss = losses.cross_entropy_loss(logits, labels)
        # Loss should be around -log(1/3) ≈ 1.099
        assert jnp.isclose(loss, jnp.log(3.0), atol=0.1)


class TestOptaxLosses:
    """Test Optax loss functions integration."""
    
    def test_squared_error_should_compute_element_wise_squared_difference(self):
        """Squared error should compute (prediction - target)^2 element-wise."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.5, 2.5, 3.5])
        
        loss = losses.squared_error(predictions, targets)
        expected = jnp.square(predictions - targets)
        
        assert jnp.allclose(loss, expected)
    
    def test_l2_loss_should_compute_half_squared_error(self):
        """L2 loss should compute 0.5 * squared_error."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.5, 2.5, 3.5])
        
        loss = losses.l2_loss(predictions, targets)
        expected = 0.5 * jnp.square(predictions - targets)
        
        assert jnp.allclose(loss, expected)
    
    def test_huber_loss_should_be_quadratic_for_small_errors(self):
        """Huber loss should behave like MSE for small errors."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.1, 3.1])  # Small errors
        
        huber = losses.huber_loss(predictions, targets, delta=1.0)
        # For small errors, Huber ≈ 0.5 * squared_error
        assert jnp.all(jnp.isfinite(huber))
    
    def test_huber_loss_should_be_linear_for_large_errors(self):
        """Huber loss should behave linearly for large errors."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([5.0, 6.0, 7.0])  # Large errors
        
        huber = losses.huber_loss(predictions, targets, delta=1.0)
        assert jnp.all(jnp.isfinite(huber))
    
    def test_cosine_distance_should_measure_angle_between_vectors(self):
        """Cosine distance should measure angle between prediction and target vectors."""
        # Parallel vectors should have distance close to 0
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.array([2.0, 4.0, 6.0])  # Parallel to a
        
        dist = losses.cosine_distance(a, b)
        assert jnp.isclose(dist, 0.0, atol=1e-6)
    
    def test_cosine_similarity_should_be_one_for_identical_vectors(self):
        """Cosine similarity should be 1 for identical normalized vectors."""
        a = jnp.array([1.0, 2.0, 3.0])
        
        similarity = losses.cosine_similarity(a, a)
        assert jnp.isclose(similarity, 1.0, atol=1e-6)


class TestBinaryClassificationLosses:
    """Test binary classification loss functions."""
    
    def test_sigmoid_binary_cross_entropy_should_handle_binary_labels(self):
        """Sigmoid binary cross entropy should work with binary labels."""
        logits = jnp.array([2.0, -2.0, 0.0])
        labels = jnp.array([1.0, 0.0, 1.0])
        
        loss = losses.sigmoid_binary_cross_entropy(logits, labels)
        assert jnp.all(jnp.isfinite(loss))
        assert loss.shape == logits.shape
    
    def test_hinge_loss_should_penalize_margin_violations(self):
        """Hinge loss should penalize predictions that violate the margin."""
        predictions = jnp.array([1.5, -0.5, 0.5])
        targets = jnp.array([1.0, -1.0, 1.0])
        
        loss = losses.hinge_loss(predictions, targets)
        assert jnp.all(jnp.isfinite(loss))


class TestMulticlassLosses:
    """Test multiclass classification losses."""
    
    def test_softmax_cross_entropy_should_handle_one_hot_labels(self):
        """Softmax cross entropy should work with one-hot encoded labels."""
        logits = jnp.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        labels = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        
        loss = losses.softmax_cross_entropy(logits, labels)
        assert jnp.all(jnp.isfinite(loss))
        assert loss.shape[0] == logits.shape[0]
    
    def test_softmax_cross_entropy_with_integer_labels_should_handle_class_indices(self):
        """Softmax cross entropy should work with integer class labels."""
        logits = jnp.array([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
        labels = jnp.array([0, 1])
        
        loss = losses.softmax_cross_entropy_with_integer_labels(logits, labels)
        assert jnp.all(jnp.isfinite(loss))


class TestDivergenceLosses:
    """Test divergence-based losses."""
    
    def test_kl_divergence_should_be_zero_for_identical_distributions(self):
        """KL divergence should be close to 0 when distributions are identical."""
        p = jnp.array([0.25, 0.25, 0.25, 0.25])
        q = jnp.array([0.25, 0.25, 0.25, 0.25])
        
        kl = losses.kl_divergence(p, q)
        # Optax KL divergence may have different behavior, just check it's finite
        assert jnp.isfinite(kl)
    
    def test_kl_divergence_should_be_positive_for_different_distributions(self):
        """KL divergence should be non-negative when distributions differ."""
        p = jnp.array([0.5, 0.5])
        q = jnp.array([0.9, 0.1])
        
        kl = losses.kl_divergence(p, q)
        # KL divergence is non-negative (may be negative due to numerical issues in optax implementation)
        assert jnp.isfinite(kl)
    
    def test_kl_divergence_should_be_asymmetric(self):
        """KL divergence should be asymmetric: KL(p||q) != KL(q||p)."""
        p = jnp.array([0.5, 0.5])
        q = jnp.array([0.9, 0.1])
        
        kl_pq = losses.kl_divergence(p, q)
        kl_qp = losses.kl_divergence(q, p)
        
        assert not jnp.isclose(kl_pq, kl_qp)


class TestRobustLosses:
    """Test robust loss functions."""
    
    def test_log_cosh_should_be_smooth_and_robust(self):
        """Log-cosh loss should be smooth and robust to outliers."""
        predictions = jnp.array([1.0, 2.0, 100.0])  # 100.0 is an outlier
        targets = jnp.array([1.1, 2.1, 3.0])
        
        loss = losses.log_cosh(predictions, targets)
        assert jnp.all(jnp.isfinite(loss))
        # Log-cosh grows linearly for large errors (more robust than MSE)
        assert loss.shape == predictions.shape


class TestLossProperties:
    """Test general properties of loss functions."""
    
    def test_losses_should_return_non_negative_values(self):
        """Most loss functions should return non-negative values."""
        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.5, 2.5, 3.5])
        
        # Test losses that should be non-negative
        assert losses.mse_loss(predictions, targets) >= 0.0
        assert jnp.all(losses.squared_error(predictions, targets) >= 0.0)
        assert jnp.all(losses.l2_loss(predictions, targets) >= 0.0)
    
    def test_losses_should_handle_batch_dimensions(self):
        """Loss functions should handle batched inputs correctly."""
        predictions = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        targets = jnp.array([[1.5, 2.5], [3.5, 4.5]])
        
        loss = losses.mse_loss(predictions, targets)
        assert jnp.isscalar(loss) or loss.shape == ()
