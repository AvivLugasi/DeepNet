from typing import Iterable, Union, Tuple
from Functions.Losses.Loss import Loss
from Functions.Metrics.Metric import Metric
from Optimizers.Optimizer import Optimizer
from Structures.Layers.Input import Input
from Structures.Layers.Layer import Layer
import numpy as np
import cupy as cp


class Model:
    def __init__(self, input_layer: Input = None, hidden_layers: Iterable[Layer] = None):
        """
        Initialize the Model.

        :param hidden_layers: A list of layers (optional). If not provided, layers should be defined in a subclass.
        """
        if hidden_layers is not None:
            self.hidden_layers = hidden_layers
        if input_layer is not None:
            self.input_layer = input_layer
        self.loss = None
        self.optimizer = None
        self.metrics = None

    def add_layer(self, layer: Layer):
        """
        Add a layer to the model. Useful when layers are defined externally.

        :param layer: The layer to be added to the model.
        """
        if isinstance(layer, Input):
            self.input_layer = layer
        elif isinstance(layer, Layer):
            self.hidden_layers.append(layer)
        else:
            raise ValueError(f"expected layer to be instance of Layer, instead got {layer.__class__}")

    def forward(self, inputs):
        """
        Perform forward propagation through the layers.

        :param inputs: The input data for the model.
        :return: The output after passing through all layers.
        """
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer.forward(x)
        return x

    def backward(self, loss_grad):
        """
        Perform backward propagation through the layers.

        :param loss_grad: The gradient of the loss function with respect to the output.
        """
        grad = loss_grad
        for layer in reversed(self.hidden_layers):
            grad = layer.backward(grad)

    def compile(self,
                optimizer: Optimizer,
                loss: Loss,
                metrics: Iterable[Metric]):
        """
        Compile the model with an optimizer and loss function.

        :param metrics: list/tuple/dict of metrics to compute each contain list of metrics names(str) or metric instances.
        :param optimizer: The optimizer for training.
        :param loss: The loss function.
        """
        if isinstance(loss, Loss):
            self.loss = loss
        else:
            raise ValueError(f"expected loss to be instance of Loss, instead got {loss.__class__}")
        if isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise ValueError(f"expected optimizer to be instance of Optimizer, instead got {optimizer.__class__}")
        for metric in metrics:
            if not isinstance(metric, Metric):
                raise ValueError(
                    f"expected metrics to be iterable instances of Metric classes, instead got {metric.__class__}")

        self.metrics = metrics

    def fit(self,
            x_train: Union[np.ndarray, cp.ndarray],
            y_train: Union[np.ndarray, cp.ndarray],
            epochs=1,
            batch_size=32,
            validation_split: float = 0.0,
            validation_data: Tuple[Union[np.ndarray, cp.ndarray]] = None,
            shuffle: bool = True,
            ):
        """
        Train the model using the provided data.

        :param x_train:
        :param y_train: Training targets.
        :param epochs: Number of epochs to train for.
        :param batch_size: Size of each training batch.
        :param validation_split:
        :param validation_data:
        :param shuffle:
        """
        # Training loop here
        for epoch in range(epochs):
            # Implement batch training, forward, loss calculation, backward, optimizer step
            pass

    def evaluate(self, x_test, y_test):
        """
        Evaluate the model's performance on test data.

        :param x_test: Test inputs.
        :param y_test: Test targets.
        :return: Performance metrics (e.g., loss, accuracy).
        """
        pass

    def predict(self, x):
        """
        Make inference on input data

        :param x: Test inputs.
        :return: predictions vector
        """
        pass
