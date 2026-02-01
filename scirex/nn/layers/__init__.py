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
from scirex.nn.layers.linear import Linear, LinearGeneral, Einsum
from scirex.nn.layers.mlp import MLP
from scirex.nn.layers.sequential import Sequential, Lambda, StatefulLayer
from scirex.nn.layers.fcnn import FCNN

# Normalization layers (from subdirectory)
from scirex.nn.layers.normalization import (
    BatchNorm,
    LayerNorm,
    RMSNorm,
    GroupNorm,
    InstanceNorm,
    SpectralNorm,
    WeightNorm,
)

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
from scirex.nn.layers.pooling import (
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    AvgPool1d,
    AvgPool2d,
    AvgPool3d,
    AdaptiveMaxPool1d,
    AdaptiveMaxPool2d,
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
)

# Regularization
from scirex.nn.layers.dropout import Dropout

# Recurrent layers
from scirex.nn.layers.recurrent import (
    GRUCell,
    LSTMCell,
    OptimizedLSTMCell,
    SimpleCell,
    RNN,
    Bidirectional,
)

# LoRA (Low-Rank Adaptation)
from scirex.nn.layers.lora import LoRA, LoRALinear, LoRAParam

# Embeddings
from scirex.nn.layers.embeddings import Embedding

# Attention
from scirex.nn.layers.attention import MultiHeadAttention, RotaryPositionalEmbedding

# Graph layers
from scirex.nn.layers.gcn import GCN, GCNModel

__all__ = [
    # Core
    "Linear",
    "LinearGeneral",
    "Einsum",
    "MLP",
    "Sequential",
    "Lambda",
    "StatefulLayer",
    "FCNN",
    # Normalization
    "BatchNorm",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "InstanceNorm",
    "SpectralNorm",
    "WeightNorm",
    # Convolution
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    # Pooling
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    # Regularization
    "Dropout",
    # Embeddings
    "Embedding",
    # Attention
    "MultiHeadAttention",
    "RotaryPositionalEmbedding",
    # Recurrent
    "GRUCell",
    "LSTMCell",
    # Graph
    "GCN",
    "GCNModel",
]
