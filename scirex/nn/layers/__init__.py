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
Module: layers/__init__.py

Exports all neural network layers.

Authors:
    - Lokesh Mohanty (lokeshm@iisc.ac.in)

Version Info:
    - 06/01/2025: Initial version
    - 01/02/2026: Migrated to Flax.NNX, added normalization subdirectory

"""

# Core layers
# Attention
from scirex.nn.layers.attention import MultiHeadAttention, RotaryPositionalEmbedding

# Convolution and pooling
from scirex.nn.layers.convolution import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)

# Regularization
from scirex.nn.layers.dropout import Dropout

# Embeddings
from scirex.nn.layers.embeddings import Embedding
from scirex.nn.layers.fcnn import FCNN

# Graph layers
from scirex.nn.layers.gcn import GCN, GCNModel
from scirex.nn.layers.linear import Einsum, Linear, LinearGeneral

# LoRA (Low-Rank Adaptation)
from scirex.nn.layers.lora import LoRA, LoRALinear, LoRAParam
from scirex.nn.layers.mlp import MLP

# Normalization layers (from subdirectory)
from scirex.nn.layers.normalization import (
    BatchNorm,
    GroupNorm,
    InstanceNorm,
    LayerNorm,
    RMSNorm,
    SpectralNorm,
    WeightNorm,
)
from scirex.nn.layers.pooling import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
)

# Recurrent layers
from scirex.nn.layers.recurrent import (
    RNN,
    Bidirectional,
    GRUCell,
    LSTMCell,
    OptimizedLSTMCell,
    SimpleCell,
)
from scirex.nn.layers.sequential import Lambda, Sequential, StatefulLayer

__all__ = [
    "FCNN",
    "GCN",
    "MLP",
    "RNN",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "BatchNorm",
    "Bidirectional",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "Dropout",
    "Einsum",
    "Embedding",
    "GCNModel",
    "GRUCell",
    "GroupNorm",
    "InstanceNorm",
    "LSTMCell",
    "Lambda",
    "LayerNorm",
    "Linear",
    "LinearGeneral",
    "LoRA",
    "LoRALinear",
    "LoRAParam",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MultiHeadAttention",
    "OptimizedLSTMCell",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "Sequential",
    "SimpleCell",
    "SpectralNorm",
    "StatefulLayer",
    "WeightNorm",
]
