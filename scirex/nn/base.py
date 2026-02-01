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
    Module: base.py

    This module implements the base class for Neural Networks

    Key Classes:
        Network: Base class for a Neural Network Architecture
        Model: Base class for a Neural Network Model

    Key Features:
        - Built on top of Jax and Flax.NNX for efficient hardware-aware computation 
        - Optimized training using autograd and jit compilation from jax and flax.nnx
        - Efficient neural networks implementation using flax.nnx modules
        - Modular and extensible design for easy customization

    Authors:
        - Lokesh Mohanty (lokeshm@iisc.ac.in)

    Version Info:
        - 02/01/2025: Initial version
        - 01/02/2026: Migrated from Equinox to Flax.NNX

"""
import time
from tqdm import tqdm
from typing import Callable, Any

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from matplotlib import pyplot as plt


class Network(nnx.Module):
    """
    Base neural network class that inherits from Flax.NNX Module.
    Create your network by inheriting from this.

    This class serves as a template for implementing neural network architectures.
    Subclasses should implement the __call__ and predict methods according to
    their specific architecture requirements.
    """

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass of the neural network.

        Args:
            x (jnp.ndarray): Input tensor to the network.

        Returns:
            jnp.ndarray: Output tensor from the network.
        """
        raise NotImplementedError

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Make predictions using the network.

        Args:
            x (jnp.ndarray): Input tensor to make predictions on.

        Returns:
            jnp.ndarray: Predicted target from the network.
        """
        return self.__call__(x)


class Model:
    """
    High-level model class that handles training, evaluation, and prediction.

    This class implements the training loop, batch processing, and evaluation
    metrics for neural network training using Flax.NNX.

    Attributes:
        history (list): List storing training metrics for each epoch.
    """

    history = []

    def __init__(
        self,
        net: Network,
        optimizer: optax.GradientTransformation,
        loss_fn: Callable,
        metrics: list[Callable] = [],
    ):
        """
        Initialize the model with network architecture and training parameters.

        Args:
            net (Network): Neural network architecture to train.
            optimizer (optax.GradientTransformation): JAX optimizer for training.
            loss_fn (Callable): Loss function for training.
            metrics (list[Callable]): List of metric functions for evaluation.
        """
        self.net = net
        self.loss_fn = loss_fn
        self.optimizer = nnx.Optimizer(net, optimizer)
        self.metrics = metrics
        
        # Create JIT-compiled versions of critical functions for efficiency
        self._jit_update_step = self._create_jit_update_step()
        self._jit_evaluate = self._create_jit_evaluate()

    def evaluate(self, x: jnp.ndarray, y: jnp.ndarray):
        """
        Evaluate the model on given data.

        Args:
            x (jnp.ndarray): Input features for evaluation.
            y (jnp.ndarray): Target values for evaluation.

        Returns:
            tuple: Loss value and list of metric values.
        """
        return self._evaluate(self.net, x, y)

    def fit(
        self,
        features: jnp.ndarray,
        target: jnp.ndarray,
        num_epochs: int = 1,
        batch_size: int = 64,
    ):
        """
        Train the model on the provided data.

        Args:
            features (jnp.ndarray): Training features.
            target (jnp.ndarray): Training targets.
            num_epochs (int, optional): Number of training epochs. Defaults to 1.
            batch_size (int, optional): Size of training batches. Defaults to 64.

        Returns:
            list: Training history containing metrics for each epoch.
        """
        self.history = []
        print("Creating batches...")
        (x_train, y_train), (x_val, y_val) = self._create_batches(
            features, target, batch_size
        )

        print("Training...")
        for epoch in tqdm(range(num_epochs), desc="Epochs", total=num_epochs):
            loss, epoch_time = self._epoch_step(x_train, y_train)
            val_loss, val_metrics = self.evaluate(x_val, y_val)
            print(
                f"Epoch {epoch+1} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s"
            )
            self.history.append(
                {
                    "loss": loss,
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "epoch_time": epoch_time,
                }
            )

        return self.history

    def predict(self, x: jnp.ndarray):
        """
        Generate predictions for given input data.

        Args:
            x (jnp.ndarray): Input features to predict on.

        Returns:
            jnp.ndarray: Model predictions.
        """
        # NNX modules are stateful but we want to use them in a functional way for vmap
        # We can use nnx.split and nnx.merge or just call direct if it doesn't have side effects
        # For simple prediction, we might not need split/merge if no state is updated.
        return jax.vmap(self.net.predict)(jnp.array(x))

    def save_net(self, filename: str):
        """
        Save the network to a file.

        Args:
            filename (str): File name to save the network.
        """
        # Simplified saving using nnx.state
        state = nnx.state(self.net)
        # In a real scenario, we might want to use flax.checkpoints
        # For now, let's use jnp.savez or similar if it's just the state
        # But eqx.tree_serialise_leaves was likely used for weight saving.
        # We'll need a way to serialize this state.
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(state, f)

    def load_net(self, filename: str):
        """
        Load the network from a file.

        Args:
            filename (str): File name to load the network from.
        """
        import pickle
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        nnx.update(self.net, state)

    def update_net(self, state: Any):
        """
        Update the content of the network.

        Args:
            state: nnx.State or compatible dictionary to update the network
        """
        nnx.update(self.net, state)

    def plot_history(self, file_name: str, figsize=(12, 6)):
        """
        Plot training history metrics.

        Args:
            file_name (str): File name to save the plot.

        Raises:
            ValueError: If history is empty
        """
        if not self.history:
            print("No training history available. Train the model first.")
            return

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        epochs = range(1, len(self.history) + 1)

        ax[0].plot(
            epochs,
            [epoch_data["loss"] for epoch_data in self.history],
            label="loss",
            marker="o",
        )
        ax[0].plot(
            epochs,
            [epoch_data["val_loss"] for epoch_data in self.history],
            label="val_loss",
            marker="x",
        )
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].set_title("Loss History")
        ax[0].legend()
        ax[0].grid(True)

        if len(self.metrics) > 0:
            ax[1].plot(
                epochs, [epoch_data["val_metrics"][0] for epoch_data in self.history]
            )
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel(self.metrics[0].__name__)
            ax[1].set_title("Validation Metrics")
            ax[1].grid(True)

        fig.tight_layout()
        plt.savefig(file_name)

    def _create_jit_update_step(self):
        """
        Create a JIT-compiled update step function for efficient training.
        Uses nnx.jit for stateful module compilation.
        
        Returns:
            Callable: JIT-compiled update function
        """
        @nnx.jit
        def update_fn(net, optimizer, features, labels):
            def loss_fn(model):
                predictions = model(features)
                return self.loss_fn(predictions, labels).mean()
            
            loss, grads = nnx.value_and_grad(loss_fn)(net)
            optimizer.update(grads)
            return loss
        
        return update_fn
    
    def _create_jit_evaluate(self):
        """
        Create a JIT-compiled evaluation function for efficient inference.
        
        Returns:
            Callable: JIT-compiled evaluation function
        """
        @nnx.jit
        def eval_fn(net, x, y):
            output = net(x)
            loss = self.loss_fn(output, y).mean()
            return loss, output
        
        return eval_fn
    
    def _evaluate(self, net: Network, x: jnp.ndarray, y: jnp.ndarray):
        """
        Internal method for model evaluation.
        Uses JIT-compiled function for efficiency.

        Args:
            net (Network): Neural network to evaluate.
            x (jnp.ndarray): Input features.
            y (jnp.ndarray): Target values.

        Returns:
            tuple: Loss value and list of metric values.
        """
        loss, output = self._jit_evaluate(net, x, y)
        metric_values = [f(output, y) for f in self.metrics]
        return loss, metric_values

    def _create_batches(self, features, targets, batch_size):
        """
        Create training batches from features and targets.
        Optimized for memory efficiency using JAX arrays.

        Args:
            features (jnp.ndarray): Input features to batch.
            targets (jnp.ndarray): Target values to batch.
            batch_size (int): Size of each batch.

        Returns:
            tuple: Tuple containing:
                - Tuple of batched features and targets
                - Tuple of validation features and targets

        Raises:
            ValueError: If batch_size is greater than number of samples.
        """
        num_samples = len(features)
        num_complete_batches = num_samples // batch_size
        
        if num_complete_batches == 0:
            raise ValueError("Batch size must be smaller than number of samples")
        
        # Reserve last batch for validation if we have more than 1 batch
        num_train_batches = num_complete_batches - (0 if num_complete_batches == 1 else 1)
        num_features_batched = num_train_batches * batch_size
        
        # Efficient reshaping using JAX arrays
        batched_features = jnp.array(features[:num_features_batched]).reshape(
            (num_train_batches, batch_size, *features.shape[1:])
        )
        batched_targets = jnp.array(targets[:num_features_batched]).reshape(
            (num_train_batches, batch_size, *targets.shape[1:])
        )
        
        # Validation data from remaining samples
        validation_data = (
            jnp.array(features[num_features_batched:]),
            jnp.array(targets[num_features_batched:]),
        )
        
        return (batched_features, batched_targets), validation_data

    def _epoch_step(self, features, labels):
        """
        Perform single epoch with efficient batch processing.

        Args:
            features (jnp.ndarray): Batch of input features.
            labels (jnp.ndarray): Batch of target values.

        Returns:
            tuple: Tuple containing:
                - Average loss for the epoch
                - Time taken for the epoch
        """
        start_time = time.time()
        
        # Accumulate losses efficiently
        losses = []
        for batch_x, batch_y in zip(features, labels):
            loss = self._update_step(batch_x, batch_y)
            losses.append(loss)
        
        # Compute mean loss efficiently
        avg_loss = jnp.mean(jnp.array(losses))
        epoch_time = time.time() - start_time
        
        return float(avg_loss), epoch_time

    def _update_step(self, features, labels):
        """
        Perform single training step using JIT-compiled function.
        Optimized for maximum performance.

        Args:
            features (jnp.ndarray): Input features.
            labels (jnp.ndarray): Target values.

        Returns:
            float: Average loss for the batch
        """
        # Use JIT-compiled update function for efficiency
        loss = self._jit_update_step(self.net, self.optimizer, features, labels)
        return float(loss)
