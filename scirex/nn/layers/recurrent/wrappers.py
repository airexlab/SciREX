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
    Module: recurrent/wrappers.py

    RNN wrapper implementations using Flax.NNX.

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/02/2026: Initial version

"""

from flax import nnx
import jax.numpy as jnp
from typing import Callable, Any


class RNN(nnx.RNN):
    """
    Generic RNN wrapper for applying cells over sequences.
    
    Wraps any RNN cell (GRU, LSTM, SimpleCell) to process sequences.
    
    Args:
        cell: RNN cell to use (GRUCell, LSTMCell, etc.)
        
    Example:
        >>> from flax import nnx
        >>> from scirex.nn.layers.recurrent import GRUCell
        >>> import jax.numpy as jnp
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> cell = GRUCell(in_features=10, hidden_features=20, rngs=rngs)
        >>> rnn = RNN(cell)
        >>> 
        >>> # Process sequence
        >>> x = jnp.ones((5, 8, 10))  # (batch, time, features)
        >>> carry = jnp.zeros((5, 20))
        >>> final_carry, outputs = rnn(carry, x)
    """
    pass


class Bidirectional(nnx.Module):
    """
    Bidirectional RNN wrapper.
    
    Processes sequences in both forward and backward directions,
    concatenating the outputs.
    
    Args:
        cell_fn: Function that creates an RNN cell
        merge_fn: How to merge forward/backward outputs ('concat', 'sum', 'avg')
        
    Example:
        >>> from flax import nnx
        >>> from scirex.nn.layers.recurrent import GRUCell, Bidirectional
        >>> import jax.numpy as jnp
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> 
        >>> def create_cell():
        ...     return GRUCell(in_features=10, hidden_features=20, rngs=rngs)
        >>> 
        >>> birnn = Bidirectional(create_cell, merge_fn='concat')
        >>> x = jnp.ones((5, 8, 10))  # (batch, time, features)
        >>> outputs = birnn(x)  # Shape: (5, 8, 40) if concat
    """
    
    def __init__(
        self,
        cell_fn: Callable[[], Any],
        merge_fn: str = 'concat',
    ):
        self.forward_rnn = nnx.RNN(cell_fn())
        self.backward_rnn = nnx.RNN(cell_fn())
        self.merge_fn = merge_fn
    
    def __call__(self, inputs, initial_carry=None):
        """
        Process sequence bidirectionally.
        
        Args:
            inputs: Input sequence (batch, time, features)
            initial_carry: Optional initial carry state
            
        Returns:
            Merged outputs from forward and backward passes
        """
        # Forward pass
        if initial_carry is None:
            batch_size = inputs.shape[0]
            hidden_size = self.forward_rnn.cell.hidden_features
            initial_carry = jnp.zeros((batch_size, hidden_size))
        
        _, forward_outputs = self.forward_rnn(initial_carry, inputs)
        
        # Backward pass (reverse sequence)
        reversed_inputs = jnp.flip(inputs, axis=1)
        _, backward_outputs = self.backward_rnn(initial_carry, reversed_inputs)
        backward_outputs = jnp.flip(backward_outputs, axis=1)
        
        # Merge outputs
        if self.merge_fn == 'concat':
            return jnp.concatenate([forward_outputs, backward_outputs], axis=-1)
        elif self.merge_fn == 'sum':
            return forward_outputs + backward_outputs
        elif self.merge_fn == 'avg':
            return (forward_outputs + backward_outputs) / 2
        else:
            raise ValueError(f"Unknown merge_fn: {self.merge_fn}")
