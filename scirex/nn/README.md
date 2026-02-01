# SciREX Neural Networks (`scirex.nn`)

A high-performance neural network library built on **Flax.NNX** and **JAX**, providing efficient,
hardware-accelerated deep learning capabilities for scientific computing and research.

## Overview

`scirex.nn` is a modular neural network framework designed for scientific applications, offering:
- **Hardware Acceleration**: Leverages JAX for GPU/TPU support with automatic differentiation
- **JIT Compilation**: Optimized performance through just-in-time compilation
- **Stateful Architecture**: Built on Flax.NNX for intuitive, object-oriented model design
- **Extensible Design**: Easy to customize and extend for research applications

## Installation

```bash
# Install with neural network dependencies
pip install scirex[nn]

# Or install from source
git clone https://github.com/scirex/scirex.git
cd scirex
pip install -e ".[nn]"
```

## Quick Start

```python
import jax.numpy as jnp
from flax import nnx
import optax
from scirex.nn import Model
from scirex.nn.layers import MLP

# Create a neural network
rngs = nnx.Rngs(42)
net = MLP(in_size=10, out_size=2, hidden_size=64, depth=3, rngs=rngs)

# Define loss and optimizer
def mse_loss(pred, target):
    return jnp.mean((pred - target) ** 2)

# Create and train model
model = Model(net=net, optimizer=optax.adam(0.001), loss_fn=mse_loss)
history = model.fit(X_train, y_train, num_epochs=10, batch_size=32)
```

## Module Structure

```
scirex/nn/
├── __init__.py                    # Module exports and public API
├── base.py                        # Base classes (Network, Model) with JIT-optimized training
├── activations.py                 # Activation functions (ReLU, GELU, Swish, etc.)
├── losses.py                      # Loss functions (MSE, cross-entropy, etc.)
├── metrics.py                     # Evaluation metrics (accuracy, precision, recall, etc.)
├── utils.py                       # Utility functions (softmax, standardization, etc.)
└── layers/
    ├── __init__.py                # Layer exports
    ├── linear.py                  # Linear, LinearGeneral, Einsum
    ├── mlp.py                     # Multi-layer perceptron
    ├── sequential.py              # Sequential composition, Lambda layers
    ├── fcnn.py                    # Fully connected neural networks
    ├── convolution.py             # Conv1d, Conv2d, Conv3d, ConvTranspose variants
    ├── pooling.py                 # All pooling operations
    ├── dropout.py                 # Dropout regularization
    ├── embeddings.py              # Embedding layers
    ├── attention.py               # Multi-head attention, positional embeddings
    ├── gcn.py                     # Graph Convolutional Networks
    ├── normalization/             # Normalization layers (organized)
    │   ├── __init__.py
    │   ├── batch.py               # BatchNorm
    │   ├── layer.py               # LayerNorm, RMSNorm
    │   ├── group.py               # GroupNorm, InstanceNorm
    │   └── weight.py              # SpectralNorm, WeightNorm
    ├── recurrent/                 # Recurrent layers (organized)
    │   ├── __init__.py
    │   ├── cells.py               # GRUCell, LSTMCell, OptimizedLSTMCell, SimpleCell
    │   └── wrappers.py            # RNN, Bidirectional
    └── lora/                      # LoRA (Low-Rank Adaptation)
        ├── __init__.py
        └── lora.py                # LoRA, LoRALinear, LoRAParam
```

## New Features (v0.2.0)

### Enhanced Normalization (7 layers)
- **GroupNorm**: Group normalization for small batch sizes
- **InstanceNorm**: Instance normalization for style transfer
- **SpectralNorm**: Spectral normalization wrapper for GANs
- **WeightNorm**: Weight normalization wrapper for faster convergence
- Organized in `layers/normalization/` subdirectory for better modularity

### Advanced Linear Layers (2 layers)
- **LinearGeneral**: Flexible linear transformations over arbitrary axes
- **Einsum**: Einstein summation for complex tensor operations

### Upsampling Layers (4 layers)
- **ConvTranspose**: Base transposed convolution for upsampling
- **ConvTranspose1d**: 1D transposed convolution for sequences
- **ConvTranspose2d**: 2D transposed convolution for images
- **ConvTranspose3d**: 3D transposed convolution for volumes

### Advanced Recurrent Layers (6 layers)
- **OptimizedLSTMCell**: Optimized LSTM implementation for better performance
- **SimpleCell**: Simple RNN cell (Elman RNN)
- **RNN**: Generic RNN wrapper for processing sequences
- **Bidirectional**: Bidirectional RNN wrapper with configurable merge strategies
- Organized in `layers/recurrent/` subdirectory
- Includes existing GRUCell and LSTMCell

### LoRA - Low-Rank Adaptation (3 layers)
- **LoRA**: Parameter-efficient fine-tuning with low-rank matrices
- **LoRALinear**: LoRA-enabled linear layer for efficient adaptation
- **LoRAParam**: LoRA parameter wrapper
- Organized in `layers/lora/` subdirectory
- Enables fine-tuning with ~1% of parameters

**Total: 22 new layers added** (19 unique + 3 moved for organization)

## Common Usage Patterns

### Building Neural Networks

**Multi-Layer Perceptron (MLP)**
```python
from scirex.nn.layers import MLP
from flax import nnx

rngs = nnx.Rngs(42)
net = MLP(in_size=784, out_size=10, hidden_size=128, depth=3, rngs=rngs)
```

**Custom Sequential Network**
```python
from scirex.nn.layers import Linear, Sequential
from scirex.nn.base import Network
import jax.nn as jnn

class CustomNet(Network):
    def __init__(self, rngs: nnx.Rngs):
        self.seq = Sequential([
            Linear(784, 256, rngs=rngs),
            Linear(256, 128, rngs=rngs),
            Linear(128, 10, rngs=rngs)
        ])

    def __call__(self, x):
        for i, layer in enumerate(self.seq.layers):
            x = layer(x)
            if i < len(self.seq.layers) - 1:
                x = jnn.relu(x)
        return x
```

**Convolutional Network**
```python
from scirex.nn.layers import Conv2d, MaxPool2d, Linear, Sequential

class ConvNet(Network):
    def __init__(self, rngs: nnx.Rngs):
        self.conv1 = Conv2d(1, 32, kernel_size=(3, 3), rngs=rngs)
        self.conv2 = Conv2d(32, 64, kernel_size=(3, 3), rngs=rngs)
        self.fc = Linear(64 * 5 * 5, 10, rngs=rngs)
```

### Training Models

**Basic Training Loop**
```python
from scirex.nn import Model
import optax

# Create model
model = Model(
    net=net,
    optimizer=optax.adam(learning_rate=0.001),
    loss_fn=lambda pred, target: jnp.mean((pred - target) ** 2),
    metrics=[]
)

# Train
history = model.fit(X_train, y_train, num_epochs=50, batch_size=32)

# Evaluate
loss, metrics = model.evaluate(X_test, y_test)

# Predict
predictions = model.predict(X_new)
```

**Graph Neural Networks**
```python
from scirex.nn.layers import GCN, GCNModel

# Create GCN
gcn = GCN(
    layers=[input_dim, 64, 32, num_classes],
    activations=['relu', 'relu', None],
    rngs=nnx.Rngs(0)
)

# Train on graph data
model = GCNModel(gcn, loss_fn=cross_entropy_loss, learning_rate=0.01)
model.fit(node_features, adjacency_matrix, degree_array, labels, epochs=100)
```

## Performance

The library is optimized for efficiency:
- **JIT Compilation**: Critical functions are JIT-compiled for speed
- **Vectorization**: Automatic vectorization through JAX
- **Memory Efficiency**: Optimized batch processing and array handling

Benchmark (MLP training on 50 samples, 5 epochs):
- Training time: ~0.97s
- 28% faster than non-optimized implementation

## Testing

```bash
# Run all tests
pytest tests/scirex/nn/ -v

# Run specific test file
pytest tests/scirex/nn/test_layers.py -v
```

## Dependencies

| Package | Purpose |
|---------|---------|
| **JAX** | Numerical computing and automatic differentiation |
| **Flax** | Neural network library (NNX modules) |
| **Optax** | Gradient processing and optimization algorithms |
| **NumPy** | Array operations and numerical computing |
| **Matplotlib** | Visualization and plotting |
