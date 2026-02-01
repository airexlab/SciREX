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
Module: recurrent/cells.py

Recurrent cell implementations using Flax.NNX.

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

Version Info:
    - 01/02/2026: Initial version

"""

from flax import nnx


class GRUCell(nnx.GRUCell):
    """
    Gated Recurrent Unit (GRU) cell.

    A simplified variant of LSTM that combines the forget and input gates
    into a single update gate.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features
        rngs: Random number generators

    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>>
        >>> rngs = nnx.Rngs(0)
        >>> cell = GRUCell(in_features=10, hidden_features=20, rngs=rngs)
        >>> x = jnp.ones((5, 10))  # (batch_size, in_features)
        >>> carry = jnp.zeros((5, 20))  # (batch_size, hidden_features)
        >>> new_carry, output = cell(carry, x)
    """

    pass


class LSTMCell(nnx.LSTMCell):
    """
    Long Short-Term Memory (LSTM) cell.

    Standard LSTM cell with input, forget, and output gates.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features
        rngs: Random number generators

    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>>
        >>> rngs = nnx.Rngs(0)
        >>> cell = LSTMCell(in_features=10, hidden_features=20, rngs=rngs)
        >>> x = jnp.ones((5, 10))
        >>> carry = (jnp.zeros((5, 20)), jnp.zeros((5, 20)))  # (h, c)
        >>> new_carry, output = cell(carry, x)
    """

    pass


class OptimizedLSTMCell(nnx.OptimizedLSTMCell):
    """
    Optimized LSTM cell implementation.

    A more efficient implementation of LSTM that combines operations
    for better performance.

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features
        rngs: Random number generators

    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>>
        >>> rngs = nnx.Rngs(0)
        >>> cell = OptimizedLSTMCell(in_features=10, hidden_features=20, rngs=rngs)
        >>> x = jnp.ones((5, 10))
        >>> carry = (jnp.zeros((5, 20)), jnp.zeros((5, 20)))
        >>> new_carry, output = cell(carry, x)
    """

    pass


class SimpleCell(nnx.SimpleCell):
    """
    Simple RNN cell (Elman RNN).

    Basic recurrent cell with tanh activation:
    h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)

    Args:
        in_features: Number of input features
        hidden_features: Number of hidden features
        rngs: Random number generators

    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>>
        >>> rngs = nnx.Rngs(0)
        >>> cell = SimpleCell(in_features=10, hidden_features=20, rngs=rngs)
        >>> x = jnp.ones((5, 10))
        >>> carry = jnp.zeros((5, 20))
        >>> new_carry, output = cell(carry, x)
    """

    pass
