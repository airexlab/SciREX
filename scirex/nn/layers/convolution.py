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
Module: convolution.py

This module implements convolutional layers for Neural Networks using Flax.NNX.

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

Version Info:
    - 06/01/2025: Initial version
    - 01/02/2026: Migrated from Equinox to Flax.NNX

"""

from typing import Union

from flax import nnx


class Conv(nnx.Conv):
    """
    Performs a convolution operation
    """


class Conv1d(nnx.Conv):
    """
    Performs a 1D convolution operation
    """


class Conv2d(nnx.Conv):
    """
    Performs a 2D convolution operation
    """


class Conv3d(nnx.Conv):
    """
    3D Convolution layer
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Union[int, tuple[int, int, int]],
        strides: Union[int, tuple[int, int, int]] = 1,
        padding: str = "SAME",
        use_bias: bool = True,
        rngs: nnx.Rngs = None,
        **kwargs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides, strides)

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            rngs=rngs,
            **kwargs,
        )


# Transposed Convolution Layers (for upsampling)


class ConvTranspose(nnx.ConvTranspose):
    """
    Transposed Convolution layer (Deconvolution) for upsampling.

    Args:
        in_features: Number of input channels
        out_features: Number of output channels
        kernel_size: Size of the convolutional kernel
        strides: Stride of the convolution (default: 1)
        padding: Padding mode ('SAME' or 'VALID', default: 'SAME')
        use_bias: Whether to use bias (default: True)
        rngs: Random number generators
    """

    pass


class ConvTranspose1d(nnx.ConvTranspose):
    """1D Transposed Convolution for upsampling 1D sequences."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Union[int, tuple[int]],
        strides: Union[int, tuple[int]] = 1,
        padding: str = "SAME",
        use_bias: bool = True,
        rngs: nnx.Rngs = None,
        **kwargs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,)
        if isinstance(strides, int):
            strides = (strides,)

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            rngs=rngs,
            **kwargs,
        )


class ConvTranspose2d(nnx.ConvTranspose):
    """2D Transposed Convolution for upsampling images."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Union[int, tuple[int, int]],
        strides: Union[int, tuple[int, int]] = 1,
        padding: str = "SAME",
        use_bias: bool = True,
        rngs: nnx.Rngs = None,
        **kwargs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides)

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            rngs=rngs,
            **kwargs,
        )


class ConvTranspose3d(nnx.ConvTranspose):
    """3D Transposed Convolution for upsampling 3D data."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: Union[int, tuple[int, int, int]],
        strides: Union[int, tuple[int, int, int]] = 1,
        padding: str = "SAME",
        use_bias: bool = True,
        rngs: nnx.Rngs = None,
        **kwargs,
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(strides, int):
            strides = (strides, strides, strides)

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            rngs=rngs,
            **kwargs,
        )
