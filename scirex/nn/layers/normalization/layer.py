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
    Module: normalization/layer.py

    Layer normalization and RMS normalization using Flax.NNX.

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/02/2026: Initial version (migrated from normalisation.py)

"""

from flax import nnx


class LayerNorm(nnx.LayerNorm):
    """
    Layer Normalization.
    
    Normalizes inputs across the feature dimension, independent of batch size.
    Particularly effective for RNNs and Transformers.
    
    Args:
        num_features: Number of features to normalize
        epsilon: Small constant for numerical stability (default: 1e-5)
        use_bias: Whether to use bias parameter (default: True)
        use_scale: Whether to use scale parameter (default: True)
        rngs: Random number generators
        
    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> ln = LayerNorm(num_features=64, rngs=rngs)
        >>> x = jnp.ones((32, 64))
        >>> output = ln(x)
    """
    pass


class RMSNorm(nnx.RMSNorm):
    """
    Root Mean Square Layer Normalization.
    
    A simplified version of LayerNorm that only uses RMS for normalization,
    without mean centering. More efficient and often performs similarly to LayerNorm.
    
    Args:
        num_features: Number of features to normalize
        epsilon: Small constant for numerical stability (default: 1e-6)
        use_scale: Whether to use scale parameter (default: True)
        rngs: Random number generators
        
    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> rms = RMSNorm(num_features=64, rngs=rngs)
        >>> x = jnp.ones((32, 64))
        >>> output = rms(x)
    """
    pass
