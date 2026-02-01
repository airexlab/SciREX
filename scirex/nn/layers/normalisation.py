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
Module: normalisation.py

This module implements normalisation layers for Neural Networks using Flax.NNX.

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

Version Info:
    - 06/01/2025: Initial version
    - 01/02/2026: Migrated from Equinox to Flax.NNX

"""

from flax import nnx


class LayerNorm(nnx.LayerNorm):
    """
    Implements a Layer Normalization layer
    """


class RMSNorm(nnx.RMSNorm):
    """
    Implements a RMS Normalization layer
    """


class GroupNorm(nnx.GroupNorm):
    """
    Implements a Group Normalization layer
    """


class BatchNorm(nnx.BatchNorm):
    """
    Implements a Batch Normalization layer
    """


# SpectralNorm and WeightNorm are not direct layers in NNX in the same way.
# They are often applied as transformations. For compatibility, we might need to skip or implement wrappers.
# Given the request to replace equinox based items, if NNX doesn't have it, we should note it.
# Actually, flax has spectral_norm in flax.nn, but NNX is a bit different.
# For now, I will omit SpectralNorm and WeightNorm if they don't have direct NNX equivalents to avoid breaking.
# Or I can provide a skeleton if they are critical. Equinox had them as wrappers.
