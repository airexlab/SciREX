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
Module: normalization/batch.py

Batch normalization layer using Flax.NNX.

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

Version Info:
    - 01/02/2026: Initial version (migrated from normalisation.py)

"""

from flax import nnx


class BatchNorm(nnx.BatchNorm):
    """
    Batch Normalization layer.

    Normalizes inputs across the batch dimension for improved training stability
    and faster convergence.

    Args:
        num_features: Number of features/channels to normalize
        momentum: Momentum for running statistics (default: 0.99)
        epsilon: Small constant for numerical stability (default: 1e-5)
        use_running_average: Whether to use running statistics (default: None, auto-determined)
        rngs: Random number generators

    Example:
        >>> import jax.numpy as jnp
        >>> from flax import nnx
        >>>
        >>> rngs = nnx.Rngs(0)
        >>> bn = BatchNorm(num_features=64, rngs=rngs)
        >>> x = jnp.ones((32, 64))  # batch_size=32, features=64
        >>> output = bn(x)
    """

    pass
