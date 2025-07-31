import logging
import numpy as np
import torch
from torch import Tensor
from typing import Callable, Dict, List, Optional, Tuple, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLoss:
    """
    Custom loss functions for the enhanced computer vision project.
    This class provides an implementation of custom loss functions used in training
    machine learning models for XR eye tracking.
    """

    def __init__(self, config: Dict[str, float] = None):
        """
        Initialize the CustomLoss class with optional configuration settings.

        :param config: Dictionary of configuration settings for the loss functions.
        """
        self.config = config or {}
        self._check_config()

    def _check_config(self):
        """
        Validate and set default values for the configuration settings.
        Raises an error if any required settings are missing.
        """
        # Example configuration settings and their defaults
        self.velocity_threshold = self.config.get('velocity_threshold', 0.5)
        self.flow_theory_constant = self.config.get('flow_theory_constant', 0.8)

    def custom_loss_function1(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Example custom loss function 1.

        :param input: Input tensor of shape (batch_size, ...).
        :param target: Target tensor of the same shape as input.
        :return: Scalar tensor representing the loss.
        """
        # Implement custom loss function logic here
        # ...

        # Return the computed loss
        return loss_value

    def custom_loss_function2(self, predicted: Tensor, true: Tensor) -> Tensor:
        """
        Example custom loss function 2.

        :param predicted: Predicted tensor of shape (batch_size, ...).
        :param true: True tensor of the same shape as predicted.
        :return: Scalar tensor representing the loss.
        """
        # Perform input validation
        if not isinstance(predicted, Tensor) or not isinstance(true, Tensor):
            raise TypeError("Input 'predicted' and 'true' must be torch Tensors.")
        if predicted.shape != true.shape:
            raise ValueError("Input tensors 'predicted' and 'true' must have the same shape.")

        # Initialize the loss variable
        loss = 0

        # Iterate over each element in the input and target tensors
        for i in range(predicted.shape[0]):
            pred = predicted[i]
            t = true[i]

            # Example loss calculation (replace with your custom loss function logic)
            element_loss = self._element_loss(pred, t)
            loss += element_loss

        # Return the mean loss across all elements
        return loss / predicted.shape[0]

    def _element_loss(self, pred: Tensor, t: Tensor) -> Tensor:
        """
        Helper function to compute the loss for a single element.

        :param pred: Predicted tensor for a single element.
        :param t: True tensor for the corresponding element.
        :return: Scalar tensor representing the loss for the element.
        """
        # Implement custom loss function logic for a single element
        # ...

        return loss_value

    def custom_loss_with_metrics(self, output: Tensor, target: Tensor, metric_names: List[str]) -> \
            Tuple[Tensor, Dict[str, float]]:
        """
        Custom loss function with additional metrics.

        :param output: Model output tensor.
        :param target: Target tensor.
        :param metric_names: List of names of the metrics to compute.
        :return: Tuple containing the scalar loss tensor and a dictionary of computed metrics.
        """
        # Perform input validation
        if not isinstance(output, Tensor) or not isinstance(target, Tensor):
            raise TypeError("Input 'output' and 'target' must be torch Tensors.")
        if output.shape != target.shape:
            raise ValueError("Input tensors 'output' and 'target' must have the same shape.")
        if not isinstance(metric_names, list):
            raise TypeError("Input 'metric_names' must be a list of strings.")
        for name in metric_names:
            if not isinstance(name, str):
                raise TypeError("All elements in 'metric_names' must be strings.")

        # Compute the loss
        loss = self.custom_loss_function2(output, target)

        # Initialize a dictionary to store the metrics
        metrics = {name: 0.0 for name in metric_names}

        # Compute and store the requested metrics
        for name in metric_names:
            metric_func = getattr(self, name, None)
            if callable(metric_func):
                metrics[name] = metric_func(output, target)

        # Return the loss and the dictionary of metrics
        return loss, metrics

    def mean_absolute_error(self, output: Tensor, target: Tensor) -> float:
        """
        Mean Absolute Error (MAE) metric.

        :param output: Model output tensor.
        :param target: Target tensor.
        :return: MAE value.
        """
        # Perform input validation
        if not isinstance(output, Tensor) or not isinstance(target, Tensor):
            raise TypeError("Input 'output' and 'target' must be torch Tensors.")
        if output.shape != target.shape:
            raise ValueError("Input tensors 'output' and 'target' must have the same shape.")

        # Compute and return the MAE
        return torch.mean(torch.abs(output - target))

    def balanced_accuracy(self, output: Tensor, target: Tensor) -> float:
        """
        Balanced Accuracy (BACC) metric.

        :param output: Model output tensor.
        :param target: Target tensor.
        :return: BACC value.
        """
        # Perform input validation
        if not isinstance(output, Tensor) or not isinstance(target, Tensor):
            raise TypeError("Input 'output' and 'target' must be torch Tensors.")
        if output.ndim != 2 or target.ndim != 1:
            raise ValueError("Input tensors must have dimensions: output (batch_size, num_classes) and target (batch_size,)")

        # Convert output probabilities to class labels
        predicted_labels = torch.argmax(output, dim=1)

        # Compute the number of classes
        num_classes = output.shape[1]

        # Compute the confusion matrix
        confusion_matrix = self._compute_confusion_matrix(predicted_labels, target, num_classes)

        # Compute the per-class accuracy
        per_class_accuracy = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)

        # Handle division by zero
        per_class_accuracy = np.nan_to_num(per_class_accuracy)

        # Compute and return the BACC
        return np.mean(per_class_accuracy)

    def _compute_confusion_matrix(self, predicted: Tensor, true: Tensor, num_classes: int) -> np.ndarray:
        """
        Helper function to compute the confusion matrix.

        :param predicted: Predicted labels tensor.
        :param true: True labels tensor.
        :param num_classes: Number of classes in the classification problem.
        :return: Confusion matrix of shape (num_classes, num_classes).
        """
        # Perform input validation
        if not isinstance(predicted, Tensor) or not isinstance(true, Tensor):
            raise TypeError("Input 'predicted' and 'true' must be torch Tensors.")
        if predicted.shape != true.shape:
            raise ValueError("Input tensors 'predicted' and 'true' must have the same shape.")
        if not isinstance(num_classes, int) or num_classes <= 1:
            raise ValueError("Input 'num_classes' must be an integer greater than 1.")

        # Initialize the confusion matrix
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

        # Convert tensors to numpy arrays
        predicted_np = predicted.cpu().numpy()
        true_np = true.cpu().numpy()

        # Populate the confusion matrix
        for i in range(predicted_np.shape[0]):
            confusion_matrix[true_np[i], predicted_np[i]] += 1

        return confusion_matrix

# Example usage
if __name__ == "__main__":
    # Create an instance of CustomLoss
    cl = CustomLoss()

    # Example inputs and targets (replace with your actual data)
    input_tensor = torch.rand(100, 5)
    target_tensor = torch.rand(100, 5)

    # Compute the loss using custom loss function 1
    loss1 = cl.custom_loss_function1(input_tensor, target_tensor)
    logger.info(f"Loss 1: {loss1}")

    # Compute the loss using custom loss function 2
    loss2 = cl.custom_loss_function2(input_tensor, target_tensor)
    logger.info(f"Loss 2: {loss2}")

    # Compute the loss and additional metrics
    metric_names = ["mean_absolute_error", "balanced_accuracy"]
    loss3, metrics = cl.custom_loss_with_metrics(input_tensor, target_tensor, metric_names)
    logger.info(f"Loss 3: {loss3}")
    logger.info(f"Metrics: {metrics}")