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
Module: sequential.py

This module implements sequential layers for Neural Networks using Flax.NNX.

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

Version Info:
    - 06/01/2025: Initial version
    - 01/02/2026: Migrated from Equinox to Flax.NNX

"""

from typing import Callable

import jax
from flax import nnx


class Sequential(nnx.Module):
    """
    Implements a Sequential layer, which is a stack of layers.
    Optimized for efficient forward passes.
    """

    def __init__(self, layers: list[nnx.Module]):
        """
        Initialize Sequential with a list of layers.

        Args:
            layers: List of nnx.Module layers to apply sequentially
        """
        self.layers = layers

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Forward pass through all layers sequentially.
        Optimized to minimize overhead.

        Args:
            x: Input tensor

        Returns:
            Output tensor after passing through all layers
        """
        for layer in self.layers:
            x = layer(x)
        return x


class Lambda(nnx.Module):
    """
    Implements a Lambda layer (which wraps a callable for use with Sequential).
    Useful for adding custom transformations in a Sequential pipeline.
    """

    def __init__(self, fn: Callable):
        """
        Initialize Lambda layer with a callable.

        Args:
            fn: Callable function to apply to inputs
        """
        self.fn = fn

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the wrapped function to the input."""
        return self.fn(x)


class StatefulLayer(nnx.Module):
    """
    Base class for stateful layers using Flax.NNX.
    All NNX modules are stateful by default.
    """

    pass
