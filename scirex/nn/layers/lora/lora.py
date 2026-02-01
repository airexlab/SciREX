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
    Module: lora/lora.py

    LoRA (Low-Rank Adaptation) implementations using Flax.NNX.
    
    LoRA is a parameter-efficient fine-tuning technique that reduces
    computational cost by approximating weight changes using low-rank matrices.

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 01/02/2026: Initial version

"""

from flax import nnx


class LoRAParam(nnx.LoRAParam):
    """
    LoRA parameter wrapper.
    
    Wraps a parameter to enable LoRA adaptation.
    
    Args:
        lora_rank: Rank of the low-rank adaptation matrices
        
    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>> 
        >>> # Create a LoRA parameter
        >>> param = LoRAParam(lora_rank=4)
    """
    pass


class LoRA(nnx.LoRA):
    """
    LoRA (Low-Rank Adaptation) layer.
    
    Implements parameter-efficient fine-tuning by learning low-rank
    updates to weight matrices: W' = W + BA, where B and A are low-rank.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        lora_rank: Rank of the adaptation matrices (default: 4)
        base_module: Optional base module to wrap
        rngs: Random number generators
        
    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> lora = LoRA(
        ...     in_features=64,
        ...     out_features=32,
        ...     lora_rank=4,
        ...     rngs=rngs
        ... )
        >>> x = jnp.ones((16, 64))
        >>> output = lora(x)  # Shape: (16, 32)
        
    Note:
        LoRA is particularly useful for fine-tuning large models where
        full parameter updates are computationally expensive. By using
        low-rank matrices, you can achieve good performance with far
        fewer trainable parameters.
    """
    pass


class LoRALinear(nnx.LoRALinear):
    """
    Linear layer with LoRA adaptation.
    
    Wraps a standard linear layer with LoRA, allowing efficient fine-tuning
    while maintaining compatibility with the original layer structure.
    
    Args:
        in_features: Number of input features
        out_features: Number of output features
        lora_rank: Rank of the adaptation matrices (default: 4)
        use_bias: Whether to use bias (default: True)
        rngs: Random number generators
        
    Example:
        >>> from flax import nnx
        >>> import jax.numpy as jnp
        >>> 
        >>> rngs = nnx.Rngs(0)
        >>> # Create a LoRA-enabled linear layer
        >>> lora_linear = LoRALinear(
        ...     in_features=64,
        ...     out_features=32,
        ...     lora_rank=4,
        ...     rngs=rngs
        ... )
        >>> x = jnp.ones((16, 64))
        >>> output = lora_linear(x)  # Shape: (16, 32)
        
    Typical Usage:
        1. Pre-train a model with standard Linear layers
        2. Replace Linear with LoRALinear for fine-tuning
        3. Freeze base weights, train only LoRA parameters
        4. Achieve good performance with ~1% of parameters
    """
    pass
