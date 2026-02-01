# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

"""
    Module: linear.py

    This module implements linear layers for Neural Networks using Flax.NNX.

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 06/01/2025: Initial version
        - 01/02/2026: Migrated from Equinox to Flax.NNX, added LinearGeneral and Einsum

"""
import jax
from flax import nnx


class Linear(nnx.Linear):
    """
    Implements a Linear layer (fully connected layer).
    
    Performs the operation: output = input @ kernel + bias
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        use_bias: Whether to include bias term (default: True)
        rngs: Random number generators
        
    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> layer = Linear(in_features=64, out_features=32, rngs=rngs)
        >>> x = jnp.ones((16, 64))
        >>> output = layer(x)
        >>> output.shape
        (16, 32)
    """
    pass


class LinearGeneral(nnx.LinearGeneral):
    """
    General linear transformation with flexible axis handling.
    
    More flexible than Linear, allowing transformations over arbitrary axes.
    Useful for attention mechanisms and complex tensor operations.
    
    Args:
        in_features: Number or tuple of input feature dimensions
        out_features: Number or tuple of output feature dimensions
        axis: Axis or axes to apply transformation over (default: -1)
        batch_axis: Batch axes to preserve (default: ())
        use_bias: Whether to include bias term (default: True)
        rngs: Random number generators
        
    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> # Transform last two dimensions
        >>> layer = LinearGeneral(in_features=(8, 8), out_features=64, rngs=rngs)
        >>> x = jnp.ones((16, 8, 8))
        >>> output = layer(x)
        >>> output.shape
        (16, 64)
    """
    pass


class Einsum(nnx.Einsum):
    """
    Einstein summation layer for complex tensor operations.
    
    Provides a flexible way to express complex linear transformations
    using Einstein notation.
    
    Args:
        shape: Shape of the weight tensor
        einsum_str: Einstein summation string (e.g., "...i,ij->...j")
        rngs: Random number generators
        
    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> # Matrix multiplication: batch @ weights
        >>> layer = Einsum(shape=(64, 32), einsum_str="...i,ij->...j", rngs=rngs)
        >>> x = jnp.ones((16, 64))
        >>> output = layer(x)
        >>> output.shape
        (16, 32)
    """
    pass


class Identity(nnx.Module):
    """
    Does nothing, useful as a placeholder
    """
    def __call__(self, x: jax.Array) -> jax.Array:
        return x
