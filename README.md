![Easy Layers](logo.jpg)
Make Models Faster with solely torch-based modules

[![PyPI version](https://img.shields.io/pypi/v/easy_layers.svg?label=pypi%20(stable))](https://pypi.org/project/easy_layers/)
[![Python Versions](https://img.shields.io/pypi/pyversions/easy_layers.svg)](https://pypi.org/project/easy_layers/)

Easy Layers provides a collection of PyTorch-based neural network modules designed for simplicity and efficiency. The library includes einsum operations, activation functions, feed forward layers, and normalization techniques - all optimized for performance.

## Installation

```bash
# install from PyPI
pip install easy_layers
```

Or install from source:

```bash
git clone https://github.com/axion66/easy_layers.git
cd easy_layers
pip install -e .
```

## Usage

The library includes several modules for different neural network components:

### nn.einsum

```python
from easy_layers.nn.einsum import UltEinsum

# Reshape operation (using einops pattern)
reshape_op = UltEinsum("b c -> c b", mode='r')
transposed = reshape_op(tensor)

# Matrix multiplication (using einsum notation)
matmul_op = UltEinsum("ik,kj->ij", mode='m')
result = matmul_op(tensor_a, tensor_b)

# Static methods
transposed = UltEinsum.reshape("h w -> w h", tensor)
result = UltEinsum.multiply("ik,kj->ij", tensor_a, tensor_b)
```

### nn.activations

```python
from easy_layers.nn.activations.acts import Activation, GELU, SiLU, ReLU, GEGLU, SWIGLU

# Using individual activation classes
gelu = GELU()
output = gelu(input_tensor)

# Using the unified Activation interface
activation = Activation("gelu")  # Options: "gelu", "silu", "relu", "geglu", "swiglu"
output = activation(input_tensor)
```

### nn.layers

```python
from easy_layers.nn.layers.layer import FeedForward

# Basic usage
ff_layer = FeedForward(
    in_features=512,
    activation="geglu"  # Default activation
)

# Custom configuration
ff_custom = FeedForward(
    in_features=512,
    out_features=256,  # Different output dimension
    hidden_features=1024,  # Custom hidden dimension
    dropout=0.1,  # Custom dropout rate
    activation="swiglu"  # Different activation
)

output = ff_layer(input_tensor)
```

### nn.norms

```python
from easy_layers.nn.norms.rms import RMSNorm

# Create RMSNorm layer
rms_norm = RMSNorm(heads=8, dim=512)

# Apply normalization
normalized = rms_norm(input_tensor)
```

## Documentation

The full API documentation can be found in the code docstrings. Example scripts are provided in the `examples` directory:

- `einsum_example.py`: Demonstrates the UltEinsum module
- `activations_example.py`: Demonstrates the activation functions
- `feedforward_example.py`: Demonstrates the FeedForward layer
- `rmsnorm_example.py`: Demonstrates the RMSNorm module

Run the examples:

```bash
python examples/einsum_example.py
python examples/activations_example.py
python examples/feedforward_example.py
python examples/rmsnorm_example.py
```

## Modules

### nn.einsum

The `einsum` module provides tools for tensor operations using Einstein summation notation and einops for reshaping.

#### UltEinsum

A versatile module that supports both reshaping operations (using einops.rearrange) and tensor multiplication (using torch.einsum).

### nn.activations

The `activations` module provides various activation functions for neural networks.

#### Activation Classes

- `GELU`: Gaussian Error Linear Unit
- `SiLU`: Sigmoid Linear Unit (also known as Swish)
- `ReLU`: Rectified Linear Unit
- `GEGLU`: Gated GELU
- `SWIGLU`: Gated SiLU (SwiGLU)

### nn.layers

The `layers` module provides neural network layer implementations.

#### FeedForward

A configurable feed forward neural network layer with support for various activation functions.

### nn.norms

The `norms` module provides normalization layers for neural networks.

#### RMSNorm

Root Mean Square Normalization, useful for transformer architectures.

## Development 

### Build the package
```bash
pip install build
python -m build
```

### Install in development mode
```bash
pip install -e .
```
