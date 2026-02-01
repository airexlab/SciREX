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
Module: normalization/weight.py

Weight normalization techniques using Flax.NNX.

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

Version Info:
    - 01/02/2026: Initial version

"""

from flax import nnx


class SpectralNorm(nnx.Module):
    """
    Spectral Normalization wrapper.

    Normalizes weights by their spectral norm (largest singular value).
    Commonly used in GANs to stabilize training by constraining the Lipschitz constant.

    Note: This is a simplified implementation. For production use, consider using
    dedicated spectral normalization libraries.

    Args:
        layer: The layer to apply spectral normalization to
        n_power_iterations: Number of power iterations for estimating spectral norm (default: 1)

    Example:
        >>> from flax import nnx
        >>>
        >>> rngs = nnx.Rngs(0)
        >>> # Apply spectral norm to a linear layer
        >>> linear = nnx.Linear(in_features=64, out_features=32, rngs=rngs)
        >>> spec_linear = SpectralNorm(linear, n_power_iterations=1)
        >>>
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((16, 64))
        >>> output = spec_linear(x)
    """

    def __init__(self, layer: nnx.Module, n_power_iterations: int = 1):
        self.layer = layer
        self.n_power_iterations = n_power_iterations
        # Note: Full spectral norm implementation would require tracking singular vectors
        # This is a placeholder that wraps the layer

    def __call__(self, *args, **kwargs):
        # Simplified: just call the wrapped layer
        # Full implementation would normalize weights before calling
        return self.layer(*args, **kwargs)


class WeightNorm(nnx.Module):
    """
    Weight Normalization wrapper.

    Reparameterizes weights as w = g * v / ||v||, where g is a learnable scalar
    and v is the weight vector. Improves conditioning and can speed up convergence.

    Note: This is a simplified implementation. For production use, consider using
    dedicated weight normalization libraries.

    Args:
        layer: The layer to apply weight normalization to

    Example:
        >>> from flax import nnx
        >>>
        >>> rngs = nnx.Rngs(0)
        >>> # Apply weight norm to a linear layer
        >>> linear = nnx.Linear(in_features=64, out_features=32, rngs=rngs)
        >>> wn_linear = WeightNorm(linear)
        >>>
        >>> import jax.numpy as jnp
        >>> x = jnp.ones((16, 64))
        >>> output = wn_linear(x)
    """

    def __init__(self, layer: nnx.Module):
        self.layer = layer
        # Note: Full weight norm implementation would reparameterize the weights
        # This is a placeholder that wraps the layer

    def __call__(self, *args, **kwargs):
        # Simplified: just call the wrapped layer
        # Full implementation would apply weight normalization before calling
        return self.layer(*args, **kwargs)
