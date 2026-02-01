# MNIST Classification with SciREX

This example demonstrates end-to-end training of a neural network on the MNIST dataset using the `scirex.nn` module.

## Overview

The example includes:
- **train.py**: Complete training script with visualization
- **mnist_example.ipynb**: Interactive Jupyter notebook
- Model architecture using SciREX layers
- Training loop with metrics tracking
- Visualization of results and predictions

## Installation

### 1. Install SciREX

```bash
# Clone the repository
git clone https://github.com/your-org/SciREX.git
cd SciREX

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### 2. Install Additional Dependencies

For this example, you'll need matplotlib for visualization:

```bash
uv pip install matplotlib tensorflow-datasets
# or
pip install matplotlib tensorflow-datasets
```

## Quick Start

### Running the Python Script

```bash
cd examples/mnist
python train.py
```

This will:
1. Download the MNIST dataset (if not already cached)
2. Train a neural network for 10 epochs
3. Display training progress with loss and accuracy
4. Save visualizations to `mnist_results.png`
5. Show sample predictions

### Running the Jupyter Notebook

```bash
jupyter notebook mnist_example.ipynb
```

The notebook provides an interactive walkthrough with:
- Step-by-step explanations
- Inline visualizations
- Experiment with different architectures
- Real-time training progress

## Model Architecture

The example uses a simple feedforward neural network:

```python
model = Sequential([
    Linear(784, 256, rngs=rngs),
    Lambda(lambda x: relu(x)),
    Dropout(0.2, rngs=rngs),
    Linear(256, 128, rngs=rngs),
    Lambda(lambda x: relu(x)),
    Dropout(0.2, rngs=rngs),
    Linear(128, 10, rngs=rngs),
])
```

**Architecture Details:**
- Input: 784 features (28×28 flattened images)
- Hidden Layer 1: 256 neurons with ReLU activation
- Dropout: 20% for regularization
- Hidden Layer 2: 128 neurons with ReLU activation
- Dropout: 20% for regularization
- Output: 10 classes (digits 0-9)

## Training Configuration

```python
learning_rate = 0.001
batch_size = 128
num_epochs = 10
optimizer = optax.adam(learning_rate)
```

## Expected Results

After 10 epochs, you should see:
- **Training Accuracy**: ~98-99%
- **Test Accuracy**: ~97-98%
- **Training Loss**: ~0.05-0.10

## Visualizations

The script generates:
1. **Training curves**: Loss and accuracy over epochs
2. **Sample predictions**: Grid of test images with predictions
3. **Confusion patterns**: Examples of misclassified digits

## Features Demonstrated

### SciREX NN Components Used

1. **Layers**:
   - `Linear`: Fully connected layers
   - `Dropout`: Regularization
   - `Sequential`: Model composition
   - `Lambda`: Custom activation functions

2. **Activations**:
   - `relu`: ReLU activation
   - `softmax`: Output probabilities

3. **Losses**:
   - `cross_entropy_loss`: Classification loss

4. **Metrics**:
   - `accuracy`: Classification accuracy

5. **Optimizers** (via Optax):
   - `adam`: Adaptive learning rate

### JAX Features

- **JIT Compilation**: Fast training with `@jax.jit`
- **Automatic Differentiation**: Gradient computation
- **Vectorization**: Efficient batch processing
- **PRNG**: Reproducible random number generation

## Customization

### Modify the Architecture

Edit the model definition in `train.py`:

```python
# Try a deeper network
model = Sequential([
    Linear(784, 512, rngs=rngs),
    Lambda(lambda x: gelu(x)),  # Try GELU activation
    BatchNorm(rngs=rngs),       # Add batch normalization
    Dropout(0.3, rngs=rngs),
    Linear(512, 256, rngs=rngs),
    Lambda(lambda x: gelu(x)),
    Dropout(0.3, rngs=rngs),
    Linear(256, 10, rngs=rngs),
])
```

### Experiment with Hyperparameters

```python
# Try different learning rates
learning_rate = 0.0001  # Lower for more stable training

# Adjust batch size
batch_size = 64  # Smaller for better generalization

# More epochs
num_epochs = 20  # Train longer
```

### Try Different Optimizers

```python
# SGD with momentum
optimizer = optax.sgd(learning_rate, momentum=0.9)

# AdamW with weight decay
optimizer = optax.adamw(learning_rate, weight_decay=1e-4)
```

## Troubleshooting

### Out of Memory

If you encounter memory issues:
```python
batch_size = 64  # Reduce batch size
```

### Slow Training

Enable JIT compilation (already enabled in the example):
```python
@jax.jit
def train_step(model, x, y):
    # Training logic
    pass
```

### Poor Accuracy

Try:
1. Increase number of epochs
2. Adjust learning rate
3. Add more layers or neurons
4. Reduce dropout rate

## File Structure

```
examples/mnist/
├── README.md              # This file
├── train.py              # Training script
├── mnist_example.ipynb   # Jupyter notebook
└── mnist_results.png     # Generated visualizations
```

## Next Steps

After running this example, try:

1. **Experiment with architectures**: Add more layers, try different activations
2. **Advanced features**: Use LoRA for efficient fine-tuning
3. **Other datasets**: Adapt the code for Fashion-MNIST or CIFAR-10
4. **Advanced techniques**: Implement learning rate scheduling, early stopping

## References

- [SciREX Documentation](../../README.md)
- [SciREX NN Module](../../scirex/nn/README.md)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax NNX Documentation](https://flax.readthedocs.io/en/latest/nnx/index.html)

## License

This example is part of the SciREX project and is licensed under the Apache License 2.0.
