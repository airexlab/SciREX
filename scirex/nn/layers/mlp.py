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
    Module: mlp.py

    This module implements Multi-Layer Perceptron (MLP) neural network architecture using Flax.NNX.

    Key Classes:
        MLP: Multi-Layer Perceptron

    Key Features:
        - Built on top of Flax.NNX getting all its functionalities
        - Efficient neural networks implementation using flax.nnx modules

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 02/01/2025: Initial version
        - 01/02/2026: Migrated from Equinox to Flax.NNX

"""
import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Sequence


class MLP(nnx.Module):
    """
    Multi-Layer Perceptron
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        hidden_size: int = 0,
        depth: int = 0,
        activation: Callable = nnx.relu,
        final_activation: Callable = lambda x: x,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        """
        Constructor for Multi-Layer Perceptron

        Args:
            in_size: Input size
            out_size: Output size
            hidden_size: Hidden size
            depth: Depth of the network
            activation: Activation function
            final_activation: Final activation function
            rngs: Random number generators
        """
        self.layers = []
        if depth == 0:
            self.layers.append(nnx.Linear(in_size, out_size, rngs=rngs))
        else:
            self.layers.append(nnx.Linear(in_size, hidden_size, rngs=rngs))
            self.layers.append(activation)
            for _ in range(depth - 1):
                self.layers.append(nnx.Linear(hidden_size, hidden_size, rngs=rngs))
                self.layers.append(activation)
            self.layers.append(nnx.Linear(hidden_size, out_size, rngs=rngs))

        self.layers.append(final_activation)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return x
