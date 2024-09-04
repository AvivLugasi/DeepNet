from typing import Iterable, Union, Tuple, Literal

from Functions.Losses.Consts import LOSS_FUNCTIONS_VALID_VALUES
from Functions.Losses.Loss import Loss
from Functions.Losses.MSE import MSE
from Functions.Losses.Utils import return_loss_func_from_str
from Functions.Metrics.Metric import Metric
from Optimizers.Consts import OPTIMIZERS_VALID_FUNCTIONS
from Optimizers.Optimizer import Optimizer
from Optimizers.SGD import SGD
from Optimizers.Utils import return_optimizer_from_str
from Structures.Layers.Dropout import _DropoutBase
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
        else:
            self.hidden_layers = []
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
        x = inputs
        for layer in self.hidden_layers:
            x = layer.forward_pass(x)
        return x

    def backward(self, loss_grad):
        """
        Perform backward propagation through the layers.

        :param loss_grad: The gradient of the loss function with respect to the output.
        """
        grads = loss_grad
        for layer in reversed(self.hidden_layers):
            grads = layer.backward_pass(grads=grads, optimizer=self.optimizer)

    def compile(self,
                optimizer: Union[Optimizer, Literal[OPTIMIZERS_VALID_FUNCTIONS]] = SGD(),
                loss: Union[Loss, Literal[LOSS_FUNCTIONS_VALID_VALUES]] = MSE(),
                metrics: Iterable[Metric] = None):
        """
        Compile the model with an optimizer, loss function and metrics.

        :param metrics: list/tuple/dict of metrics to compute each contain list of metrics names(str) or metric instances.
        :param optimizer: The optimizer for training.
        :param loss: The loss function.
        """
        if isinstance(loss, str):
            loss = return_loss_func_from_str(loss)
        if isinstance(loss, Loss):
            self.loss = loss
        else:
            raise ValueError(f"expected loss to be instance of Loss, instead got {loss.__class__}")
        if isinstance(optimizer, str):
            optimizer = return_optimizer_from_str(optimizer)
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
            y_train: Union[np.ndarray, cp.ndarray] = None,
            x_train: Union[np.ndarray, cp.ndarray] = None,
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
        self.set_mode(is_training=True)
        if x_train is not None and y_train is not None:
            self.input_layer.set_data(data_x=x_train, data_y=y_train)

        if shuffle:
            self.input_layer.shuffle_in_every_epoch = shuffle
        self.input_layer.batch_size = batch_size
        self.input_layer.init_queue()
        num_of_batches = len(self.input_layer.batches_queue)

        if validation_data is not None:
            validation_x = validation_data[0]
            validation_y = validation_data[1]
            if validation_x.shape[0] != self.input_layer.data_x.shape[0]:
                validation_x = validation_x.T
            if validation_y.shape[0] != self.input_layer.data_y.shape[0]:
                validation_y = validation_y.T

        # Training loop here
        for epoch in range(epochs):
            print(f"epoch number: {epoch}")
            total_loss = 0
            # Implement batch training, forward, loss calculation, backward, optimizer step
            for batch_i in range(0, num_of_batches):
                # forward pass
                print(f"batch number: {batch_i} of epoch:{epoch}")
                batch = self.input_layer.forward_pass()
                batch_x, batch_y = batch[0], batch[1]
                output = self.forward(inputs=batch_x)
                batch_loss = self.loss.loss(ground_truth=batch_y,
                                            predictions=output)
                total_loss += batch_loss
                print(f"batch {self.loss.config()} loss is:{batch_loss:.8f} for batch number: {batch_i} of epoch:{epoch}")
                # backward pass
                loss_grad = self.loss.loss_derivative(ground_truth=batch_y,
                                                      predictions=output)
                self.backward(loss_grad=loss_grad)

            avg_loss = total_loss / len(self.input_layer.batches_queue)
            print(f"training losses: {self.loss.config()}: avg loss is:{avg_loss:.8f} total loss is:{total_loss:.8f} for epoch:{epoch}")
            val_output = self.forward(inputs=validation_x)
            val_loss = self.loss.loss(ground_truth=validation_y,
                                      predictions=val_output)
            print(f"validation loss: {val_loss:.8f} for epoch:{epoch}")
            if self.input_layer.shuffle_in_every_epoch:
                self.input_layer.batches_queue = None

    def evaluate(self,
                 x_test: Union[np.ndarray, cp.ndarray],
                 y_test: Union[np.ndarray, cp.ndarray],
                 samples_as_cols: bool = False):
        """
        Evaluate the model's performance on test data.

        :param samples_as_cols: whether the samples represented as the array cols
        :param x_test: Test inputs.
        :param y_test: Test targets.
        :return: Performance metrics (e.g., loss, accuracy).
        """
        self.set_mode(is_training=False)
        if not samples_as_cols:
            y_test = y_test.T
            x_test = x_test.T
        predictions = self.forward(inputs=x_test)
        loss = self.loss.loss(ground_truth=y_test,
                              predictions=predictions)
        print(f"{self.loss.config()} loss is:{loss:.3f}")
        for metric in self.metrics:
            score = metric.score(ground_truth=y_test,
                                 predictions=predictions)
            print(f"{metric.config()} score is:{score:.3f}")

    def predict(self, x):
        """
        Make inference on input data

        :param x: Test inputs.
        :return: predictions vector
        """
        self.set_mode(is_training=False)

    def set_mode(self, is_training: bool = True):
        for layer in self.hidden_layers:
            if isinstance(layer, _DropoutBase):
                layer.set_mod(is_training=is_training)
