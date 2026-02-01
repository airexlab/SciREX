"""
MNIST Classification Example using SciREX NN

This script demonstrates end-to-end training of a neural network on the MNIST dataset
using the scirex.nn module with Flax NNX.

Features:
- Data loading and preprocessing
- Model definition using SciREX layers
- Training loop with JIT compilation
- Metrics tracking and visualization
- Sample predictions display
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, List

# Import SciREX components
from scirex.nn.layers import Linear, Dropout, Sequential, Lambda
from scirex.nn.activations import relu
from scirex.nn.losses import cross_entropy_loss
from scirex.nn.metrics import accuracy
from scirex.nn.utils import softmax


def load_mnist_data(batch_size: int = 128) -> Tuple[tfds.data.Dataset, tfds.data.Dataset]:
    """
    Load and preprocess MNIST dataset.
    
    Args:
        batch_size: Batch size for training
        
    Returns:
        train_ds: Training dataset
        test_ds: Test dataset
    """
    print("Loading MNIST dataset...")
    
    # Load dataset
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    
    train_ds = ds_builder.as_dataset(split='train', batch_size=batch_size)
    test_ds = ds_builder.as_dataset(split='test', batch_size=batch_size)
    
    def preprocess(batch):
        """Normalize images and convert labels."""
        image = jnp.array(batch['image'], dtype=jnp.float32) / 255.0
        image = image.reshape(-1, 784)  # Flatten 28x28 to 784
        label = jnp.array(batch['label'], dtype=jnp.int32)
        return image, label
    
    # Preprocess datasets
    train_ds = train_ds.map(preprocess)
    test_ds = test_ds.map(preprocess)
    
    print(f"✓ Dataset loaded: {ds_builder.info.splits['train'].num_examples} train, "
          f"{ds_builder.info.splits['test'].num_examples} test samples")
    
    return train_ds, test_ds


def create_model(rngs: nnx.Rngs) -> Sequential:
    """
    Create a simple feedforward neural network for MNIST classification.
    
    Args:
        rngs: Random number generators
        
    Returns:
        model: Sequential model
    """
    model = Sequential([
        Linear(784, 256, rngs=rngs),
        Lambda(lambda x: relu(x)),
        Dropout(0.2, rngs=rngs),
        Linear(256, 128, rngs=rngs),
        Lambda(lambda x: relu(x)),
        Dropout(0.2, rngs=rngs),
        Linear(128, 10, rngs=rngs),
    ])
    
    print("\n" + "="*60)
    print("Model Architecture:")
    print("="*60)
    print("Input:  784 features (28×28 flattened)")
    print("Layer 1: Linear(784 → 256) + ReLU + Dropout(0.2)")
    print("Layer 2: Linear(256 → 128) + ReLU + Dropout(0.2)")
    print("Output: Linear(128 → 10)")
    print("="*60 + "\n")
    
    return model


@jax.jit
def train_step(model: Sequential, optimizer_state, x: jax.Array, y: jax.Array):
    """
    Single training step with JIT compilation.
    
    Args:
        model: Neural network model
        optimizer_state: Optimizer state
        x: Input batch
        y: Target labels
        
    Returns:
        loss: Training loss
        acc: Training accuracy
        model: Updated model
        optimizer_state: Updated optimizer state
    """
    def loss_fn(model):
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        return loss, logits
    
    # Compute gradients
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(model)
    
    # Update parameters
    updates, optimizer_state = optimizer_state.update(grads, model)
    model = nnx.apply_updates(model, updates)
    
    # Compute accuracy
    predictions = jnp.argmax(logits, axis=-1)
    acc = accuracy(predictions, y)
    
    return loss, acc, model, optimizer_state


def evaluate(model: Sequential, test_ds) -> Tuple[float, float]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_ds: Test dataset
        
    Returns:
        avg_loss: Average test loss
        avg_acc: Average test accuracy
    """
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    for x, y in test_ds:
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        predictions = jnp.argmax(logits, axis=-1)
        acc = accuracy(predictions, y)
        
        total_loss += loss
        total_acc += acc
        num_batches += 1
    
    return total_loss / num_batches, total_acc / num_batches


def visualize_results(history: Dict[str, List[float]], 
                      model: Sequential, 
                      test_ds,
                      save_path: str = "mnist_results.png"):
    """
    Visualize training results and sample predictions.
    
    Args:
        history: Training history
        model: Trained model
        test_ds: Test dataset
        save_path: Path to save visualization
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Training curves
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['test_acc'], label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 2-7: Sample predictions
    # Get one batch from test set
    for x, y in test_ds.take(1):
        images = x[:12].reshape(-1, 28, 28)
        labels = y[:12]
        logits = model(x[:12])
        predictions = jnp.argmax(logits, axis=-1)
        
        for i in range(6):
            ax = plt.subplot(2, 3, i + 4)
            ax.imshow(images[i], cmap='gray')
            
            true_label = int(labels[i])
            pred_label = int(predictions[i])
            
            color = 'green' if true_label == pred_label else 'red'
            ax.set_title(f'True: {true_label}, Pred: {pred_label}', 
                        color=color, fontsize=12, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    plt.show()


def main():
    """Main training function."""
    print("\n" + "="*60)
    print("MNIST Classification with SciREX NN")
    print("="*60 + "\n")
    
    # Hyperparameters
    learning_rate = 0.001
    batch_size = 128
    num_epochs = 10
    
    print("Hyperparameters:")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Epochs: {num_epochs}\n")
    
    # Set random seed for reproducibility
    seed = 42
    rngs = nnx.Rngs(seed)
    
    # Load data
    train_ds, test_ds = load_mnist_data(batch_size)
    
    # Create model
    model = create_model(rngs)
    
    # Create optimizer
    optimizer = optax.adam(learning_rate)
    optimizer_state = optimizer.init(nnx.state(model))
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    print("Starting training...\n")
    print("="*60)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        # Train on all batches
        for x, y in train_ds:
            loss, acc, model, optimizer_state = train_step(model, optimizer_state, x, y)
            epoch_loss += loss
            epoch_acc += acc
            num_batches += 1
        
        # Average metrics
        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = epoch_acc / num_batches
        
        # Evaluate on test set
        test_loss, test_acc = evaluate(model, test_ds)
        
        # Store history
        history['train_loss'].append(float(avg_train_loss))
        history['train_acc'].append(float(avg_train_acc))
        history['test_loss'].append(float(test_loss))
        history['test_acc'].append(float(test_acc))
        
        # Print progress
        print(f"Epoch {epoch + 1:2d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {avg_train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Test Acc: {test_acc:.4f}")
    
    print("="*60)
    print("\n✓ Training completed!\n")
    
    # Final results
    print("Final Results:")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.2%}")
    print(f"  Test Accuracy:  {history['test_acc'][-1]:.2%}")
    print(f"  Train Loss:     {history['train_loss'][-1]:.4f}")
    print(f"  Test Loss:      {history['test_loss'][-1]:.4f}\n")
    
    # Visualize results
    visualize_results(history, model, test_ds)
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
