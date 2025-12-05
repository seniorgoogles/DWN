import torch
import torch.nn as nn
from .binarization import DistributiveThermometer, Thermometer
import numpy as np


class StraightThroughEstimator(torch.autograd.Function):
    """
    Simple STE: Forward pass uses hard thresholding, backward pass uses identity.
    """
    @staticmethod
    def forward(ctx, x, thresholds):
        output = (x > thresholds).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class EncoderLayer(nn.Module):
    """
    Encoder layer for continuous to binary conversion using learned thresholds.
    Thermometer encoding with learnable thresholds initialized from data distribution.
    """

    def __init__(self, inputs, output_size, input_dataset=None, estimator_type='ste', **kwargs):
        """
        Args:
            inputs: Number of input features
            output_size: Number of thresholds per feature
            input_dataset: Dataset to initialize thresholds (required for 'FiniteDifference')
            estimator_type: 'STE'
        """
        super().__init__()

        if self.estimator_type == 'ste':
            print("Initializing EncoderLayer with Straight-Through Estimator.")
            if x is not None:
                # Initialize thermometer encoder based on type
                if kwargs["thermometer_encoder"] == "uniform":
                    thermometer = Thermometer(num_bits=output_size, feature_wise=True)
                elif kwargs["thermometer_encoder"] == "gaussian":
                    thermometer = GaussianThermometer(num_bits=output_size, feature_wise=True)
                elif kwargs["thermometer_encoder"] == "distributive":
                    thermometer = DistributiveThermometer(num_bits=output_size, feature_wise=True)
                else:
                    raise ValueError(f"Unsupported thermometer_encoder: {kwargs['thermometer_encoder']}")
                
                thermometer.fit(x)
                thresholds = thermometer.thresholds
                if not torch.is_tensor(thresholds):
                    thresholds = torch.tensor(thresholds, dtype=torch.float32)
            else:
                # Initialize with uniform linspace if no dataset provided
                thresholds = torch.linspace(-0.9, 0.9, output_size).unsqueeze(0).repeat(inputs, 1)

            self.thresholds = nn.Parameter(thresholds, requires_grad=True)

        else:
            raise ValueError(f"Unsupported estimator_type: {estimator_type}")

    def forward(self, x):
        """
        Encode input using the selected estimator (finite difference or GLT).

        Args:
            x: Input tensor of shape (batch_size, features)

        Returns:
            Encoded tensor of shape (batch_size, features, thresholds)
        """
        # Convert to tensor if needed
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32, device=self.thresholds.device)
        else:
            x = x.to(self.thresholds.device).float()

        if self.estimator_type == 'ste':
            # Straight-Through Estimator
            if self.training:
                with torch.no_grad():
                    self.thresholds.clamp_(-1, 1)

            sorted_thresholds = torch.sort(self.thresholds, dim=1)[0]
            x_expanded = x.unsqueeze(-1)
            encoded = StraightThroughEstimator.apply(x_expanded, sorted_thresholds)
        else:
            raise ValueError(f"Unsupported estimator_type: {self.estimator_type}")

        return encoded
