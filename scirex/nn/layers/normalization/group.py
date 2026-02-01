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
    Module: normalization/group.py

    Group normalization and Instance normalization using Flax.NNX.

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/02/2026: Initial version

"""

from flax import nnx


class GroupNorm(nnx.GroupNorm):
    """
    Group Normalization.
    
    Divides channels into groups and normalizes within each group.
    Effective for small batch sizes where BatchNorm struggles.
    
    Args:
        num_groups: Number of groups to divide channels into
        num_features: Total number of features/channels (optional if can be inferred)
        epsilon: Small constant for numerical stability (default: 1e-5)
        use_bias: Whether to use bias parameter (default: True)
        use_scale: Whether to use scale parameter (default: True)
        rngs: Random number generators
        
    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> # 32 channels divided into 8 groups
        >>> gn = GroupNorm(num_groups=8, rngs=rngs)
        >>> x = jnp.ones((4, 32, 32, 32))  # (batch, height, width, channels)
        >>> output = gn(x)
    """
    pass


class InstanceNorm(nnx.Module):
    """
    Instance Normalization.
    
    Normalizes each instance (sample) independently across spatial dimensions.
    Commonly used in style transfer and image generation tasks.
    Implemented as GroupNorm with num_groups = num_features.
    
    Args:
        num_features: Number of features/channels to normalize
        epsilon: Small constant for numerical stability (default: 1e-5)
        use_bias: Whether to use bias parameter (default: True)
        use_scale: Whether to use scale parameter (default: True)
        rngs: Random number generators
        
    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> in_norm = InstanceNorm(num_features=32, rngs=rngs)
        >>> x = jnp.ones((4, 32, 32, 32))  # (batch, height, width, channels)
        >>> output = in_norm(x)
    """
    
    def __init__(
        self,
        num_features: int,
        epsilon: float = 1e-5,
        use_bias: bool = True,
        use_scale: bool = True,
        rngs: nnx.Rngs = None,
    ):
        # InstanceNorm is equivalent to GroupNorm with num_groups = num_features
        self.group_norm = nnx.GroupNorm(
            num_groups=num_features,
            epsilon=epsilon,
            use_bias=use_bias,
            use_scale=use_scale,
            rngs=rngs,
        )
    
    def __call__(self, x):
        return self.group_norm(x)
