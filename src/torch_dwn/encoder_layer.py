import torch
import torch.nn as nn
from .binarization import DistributiveThermometer, Thermometer
import numpy as np


def compute_bit_importance_from_lut(lut_layer, method='count', normalize=True):
    """
    Extract importance of each thermometer bit from LUTLayer mapping and truth tables.

    Args:
        lut_layer: A trained LUTLayer instance
        method: How to compute importance:
            - 'count': Count how many times each bit appears in the mapping
            - 'weighted': Weight by mean absolute LUT values
            - 'gradient': Weight by LUT gradient magnitudes (requires .grad)
        normalize: If True, normalize importance to [0, 1] range

    Returns:
        importance: Tensor of shape [input_size] with importance weight for each bit

    Example:
        >>> importance = compute_bit_importance_from_lut(lut_layer, method='weighted')
        >>> print(f"Most important bits: {importance.topk(10).indices}")
        >>> # Reshape to [num_features, num_thresholds] for EncoderLayer
        >>> importance_per_feature = importance.view(num_features, num_thresholds)
    """
    from .lut_layer import LearnableMapping

    # Get the mapping tensor
    if isinstance(lut_layer.mapping, LearnableMapping):
        # For learnable mapping, use the dummy mapping (sequential indices after soft routing)
        # LearnableMapping reorders inputs, then uses sequential connections
        if hasattr(lut_layer, '_LUTLayer__dummy_mapping'):
            mapping = lut_layer._LUTLayer__dummy_mapping
        else:
            raise ValueError("LearnableMapping detected but cannot access discrete mapping. "
                           "Use fixed mapping ('random' or 'arange') for importance analysis.")
    else:
        mapping = lut_layer.mapping

    # mapping shape: [output_size, n] where n is inputs per LUT
    input_size = lut_layer.input_size

    if method == 'count':
        # Count how many times each input bit appears in the mapping
        importance = torch.zeros(input_size, dtype=torch.float32, device=mapping.device)
        unique, counts = torch.unique(mapping, return_counts=True)
        importance[unique] = counts.float()

    elif method == 'weighted':
        # Weight by mean absolute value of LUT truth tables that use each bit
        importance = torch.zeros(input_size, dtype=torch.float32, device=mapping.device)
        lut_weights = lut_layer.luts.abs().mean(dim=1)  # [output_size]

        for bit_idx in range(input_size):
            # Find which LUTs use this bit
            lut_mask = (mapping == bit_idx).any(dim=1)  # [output_size]
            if lut_mask.any():
                importance[bit_idx] = lut_weights[lut_mask].mean()

    elif method == 'gradient':
        # Weight by mean absolute gradient magnitude
        if lut_layer.luts.grad is None:
            raise ValueError("method='gradient' requires LUT gradients. Run backward pass first.")

        importance = torch.zeros(input_size, dtype=torch.float32, device=mapping.device)
        lut_grad_mag = lut_layer.luts.grad.abs().mean(dim=1)  # [output_size]

        for bit_idx in range(input_size):
            lut_mask = (mapping == bit_idx).any(dim=1)
            if lut_mask.any():
                importance[bit_idx] = lut_grad_mag[lut_mask].mean()

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'count', 'weighted', or 'gradient'.")

    if normalize and importance.max() > 0:
        importance = importance / importance.max()

    return importance


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


class FiniteDifferenceEstimator(torch.autograd.Function):
    """
    Finite Difference for Threshold Training:
    - Forward: Hard thresholding (x > threshold)
    - Backward: Finite difference approximation of gradient w.r.t. thresholds
    """
    @staticmethod
    def forward(ctx, x, thresholds, delta=0.00334375):
        ctx.save_for_backward(x, thresholds)
        ctx.delta = delta
        output = (x > thresholds).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        x, thresholds = ctx.saved_tensors
        delta = ctx.delta

        # Finite difference approximation for threshold gradients
        output_plus = (x > (thresholds + delta)).float()
        output_minus = (x > (thresholds - delta)).float()

        # Gradient w.r.t. thresholds using central difference
        grad_thresholds = (output_plus - output_minus) / (2 * delta)
        grad_thresholds = grad_thresholds * grad_output

        # Average over batch dimension
        grad_thresholds = grad_thresholds.mean(dim=0)

        # For input: use straight-through
        grad_input = grad_output
        
        return grad_input, grad_thresholds, None


class ImportanceWeightedFiniteDifferenceEstimator(torch.autograd.Function):
    """
    Finite Difference Estimator with Importance Weighting:
    - Forward: Hard thresholding (x > threshold)
    - Backward: Finite difference with gradients weighted by bit importance

    This allows focusing gradient updates on thresholds that produce bits
    actually used by downstream LUTs (based on mapping analysis).

    Uses SOFT weighting: unused bits get reduced gradients (not zero),
    so all thresholds can still learn.
    """
    @staticmethod
    def forward(ctx, x, thresholds, importance_weights, min_weight=0.1, delta=0.00334375):
        """
        Args:
            x: Input tensor [batch, features, 1]
            thresholds: Threshold values [features, num_thresholds]
            importance_weights: Importance of each threshold [features, num_thresholds]
            min_weight: Minimum gradient weight for unused bits (default: 0.1)
            delta: Step size for finite difference
        """
        ctx.save_for_backward(x, thresholds, importance_weights)
        ctx.min_weight = min_weight
        ctx.delta = delta
        output = (x > thresholds).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, thresholds, importance_weights = ctx.saved_tensors
        min_weight = ctx.min_weight
        delta = ctx.delta

        # Finite difference approximation for threshold gradients
        output_plus = (x > (thresholds + delta)).float()
        output_minus = (x > (thresholds - delta)).float()

        # Gradient w.r.t. thresholds using central difference
        grad_thresholds = (output_plus - output_minus) / (2 * delta)
        grad_thresholds = grad_thresholds * grad_output

        # Average over batch dimension
        grad_thresholds = grad_thresholds.mean(dim=0)

        # SOFT weighting: scale importance to [min_weight, 1.0]
        # This way unused bits still get min_weight * gradient, not zero!
        importance_soft = importance_weights * (1.0 - min_weight) + min_weight

        # Weight by soft importance
        grad_thresholds = grad_thresholds * importance_soft

        # For input: use straight-through
        grad_input = grad_output

        return grad_input, grad_thresholds, None, None, None


class GLTEstimator(torch.autograd.Function):
    """
    GLT estimator as proposed in the paper.
    Forward: thermometer encoding using cumulative learned thresholds.
    Backward: rectified STE for threshold gradients, straight-through for input.
    """
    @staticmethod
    def forward(ctx, x, latent_thresholds, m=5, p=2):
        """
        x: [B,C,H,W] or [B,F] or [B,F,1] input
        latent_thresholds: [M] learnable latent parameters (>0)
        m: parameter for rectified STE
        p: parameter for rectified STE
        """
        # Compute cumulative normalized thresholds
        positive = torch.relu(latent_thresholds) + 1e-5
        normalized = positive / positive.sum()
        thresholds = torch.cumsum(normalized, dim=0)  # monotonic increasing, <1
        thresholds = thresholds * 2 - 1  # scale to [-1, 1]

        ctx.save_for_backward(x, thresholds)
        ctx.m = m
        ctx.p = p

        # Expand thresholds for broadcasting
        thresholds_exp = thresholds.view(*([1]*x.ndim), -1)  # [1,1,1,1,M]
        x_exp = x.unsqueeze(-1)                               # [B,C,H,W,1]

        # Thermometer encoding
        out = (x_exp >= thresholds_exp).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, thresholds = ctx.saved_tensors
        m = ctx.m
        p = ctx.p

        # Expand thresholds
        thresholds_exp = thresholds.view(*([1]*x.ndim), -1)
        x_exp = x.unsqueeze(-1)

        # Rectified STE gradient for thresholds
        diff = torch.abs(x_exp - thresholds_exp)
        grad_heaviside = (1.0 / m) * torch.min(torch.ones_like(diff), diff ** (p - 1))
        grad_thresholds = grad_heaviside * grad_output

        # Average over all dimensions except threshold dimension
        dims_to_avg = tuple(range(grad_thresholds.ndim - 1))
        grad_thresholds = grad_thresholds.mean(dim=dims_to_avg)

        # Straight-through for input
        grad_input = grad_output.sum(dim=-1)  # sum over bit planes

        return grad_input, grad_thresholds, None, None  # match forward inputs


class ImportanceWeightedGLTEstimator(torch.autograd.Function):
    """
    GLT estimator with importance weighting.
    Forward: thermometer encoding using cumulative learned thresholds.
    Backward: rectified STE weighted by bit importance.

    Uses SOFT weighting: unused bits get reduced gradients (not zero).
    """
    @staticmethod
    def forward(ctx, x, latent_thresholds, importance_weights, min_weight=0.1, m=5, p=2):
        """
        x: [B,C,H,W] or [B,F] or [B,F,1] input
        latent_thresholds: [M] learnable latent parameters (>0)
        importance_weights: [M] importance of each threshold
        min_weight: Minimum gradient weight for unused bits (default: 0.1)
        m: parameter for rectified STE
        p: parameter for rectified STE
        """
        # Compute cumulative normalized thresholds
        positive = torch.relu(latent_thresholds) + 1e-5
        normalized = positive / positive.sum()
        thresholds = torch.cumsum(normalized, dim=0)  # monotonic increasing, <1
        thresholds = thresholds * 2 - 1  # scale to [-1, 1]

        ctx.save_for_backward(x, thresholds, importance_weights)
        ctx.min_weight = min_weight
        ctx.m = m
        ctx.p = p

        # Expand thresholds for broadcasting
        thresholds_exp = thresholds.view(*([1]*x.ndim), -1)
        x_exp = x.unsqueeze(-1)

        # Thermometer encoding
        out = (x_exp >= thresholds_exp).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, thresholds, importance_weights = ctx.saved_tensors
        min_weight = ctx.min_weight
        m = ctx.m
        p = ctx.p

        # Expand thresholds
        thresholds_exp = thresholds.view(*([1]*x.ndim), -1)
        x_exp = x.unsqueeze(-1)

        # Rectified STE gradient for thresholds
        diff = torch.abs(x_exp - thresholds_exp)
        grad_heaviside = (1.0 / m) * torch.min(torch.ones_like(diff), diff ** (p - 1))
        grad_thresholds = grad_heaviside * grad_output

        # Average over all dimensions except threshold dimension
        dims_to_avg = tuple(range(grad_thresholds.ndim - 1))
        grad_thresholds = grad_thresholds.mean(dim=dims_to_avg)

        # SOFT weighting: scale importance to [min_weight, 1.0]
        importance_soft = importance_weights * (1.0 - min_weight) + min_weight

        # Weight by soft importance
        grad_thresholds = grad_thresholds * importance_soft

        # Straight-through for input
        grad_input = grad_output.sum(dim=-1)

        return grad_input, grad_thresholds, None, None, None, None


class EncoderLayer(nn.Module):
    """
    Encoder layer for continuous to binary conversion using learned thresholds.
    Thermometer encoding with learnable thresholds initialized from data distribution.
    """

    def __init__(self, inputs, output_size, input_dataset=None, estimator_type='glt',
                 glt_m=5, glt_p=2, **kwargs):
        """
        Args:
            inputs: Number of input features
            output_size: Number of thresholds per feature
            input_dataset: Dataset to initialize thresholds (required for 'FiniteDifference')
            estimator_type: 'GLT', 'glt', 'FiniteDifference', or 'finite_difference' (default: 'glt')
            glt_m: Parameter m for GLT rectified STE (default: 5)
            glt_p: Parameter p for GLT rectified STE (default: 2)
        """
        super().__init__()

        # Normalize estimator_type to handle multiple formats
        # Accept: 'GLT', 'glt', 'FiniteDifference', 'finite_difference'
        estimator_map = {
            'glt': 'glt',
            'finitedifference': 'finite_difference',
            'finite_difference': 'finite_difference'
        }
        normalized_type = estimator_type.lower().replace('_', '').replace('-', '')
        if normalized_type not in estimator_map:
            raise ValueError(
                f"Unknown estimator_type: '{estimator_type}'. "
                f"Choose 'GLT', 'glt', 'FiniteDifference', or 'finite_difference'."
            )
        self.estimator_type = estimator_map[normalized_type]
        self.glt_m = glt_m
        self.glt_p = glt_p
        self.inputs = inputs

        # Convert input_dataset to tensor (needed for both estimator types)
        if input_dataset is not None:
            if torch.is_tensor(input_dataset):
                x = input_dataset.detach().clone().float()
            else:
                x = torch.tensor(input_dataset, dtype=torch.float32)
        else:
            x = None

        if self.estimator_type == 'finite_difference':

            print("Initializing EncoderLayer with Finite Difference Estimator.")
            if x is None:
                raise ValueError("EncoderLayer with 'FiniteDifference' requires 'input_dataset' to initialize thresholds.")

            # Initialize with distributive thresholds
            thermometer = DistributiveThermometer(num_bits=output_size, feature_wise=True)
            thermometer.fit(x)
            thresholds = thermometer.thresholds

            if not torch.is_tensor(thresholds):
                thresholds = torch.tensor(thresholds, dtype=torch.float32)

            self.thresholds = nn.Parameter(thresholds, requires_grad=True)

        elif self.estimator_type == 'glt':
            print("Initializing EncoderLayer with GLT Estimator.")
            # GLT uses latent thresholds per feature
            if x is not None:
                # Initialize with Thermometer from data distribution
                thermometer = Thermometer(num_bits=output_size, feature_wise=True)
                thermometer.fit(x)
                thresholds = thermometer.thresholds
            else:
                # Initialize with uniform linspace if no dataset provided
                thresholds = torch.linspace(-0.9, 0.9, output_size).unsqueeze(0).repeat(inputs, 1)
            self.thresholds = nn.Parameter(thresholds, requires_grad=True)

        # Initialize importance weights as buffer (optional, for importance-weighted training)
        # Don't set it here - will be set via set_importance_from_lut() if needed
        self.importance_min_weight = 0.1  # Default: unused bits get 10% of gradient

    def set_importance_from_lut(self, lut_layer, method='weighted', min_weight=0.1):
        """
        Set importance weights from a LUTLayer's mapping and truth tables.

        Uses SOFT weighting: unused bits get min_weight * gradient (not zero),
        so all thresholds can still learn, just at different rates.

        Args:
            lut_layer: A LUTLayer instance (after training or with initialized mapping)
            method: 'count', 'weighted', or 'gradient' (see compute_bit_importance_from_lut)
            min_weight: Minimum gradient weight for unused bits (0.0-1.0, default: 0.1)
                       0.0 = hard weighting (unused bits get 0 gradient)
                       0.1 = soft weighting (unused bits get 10% gradient)
                       1.0 = no weighting (all bits equal)

        Example:
            >>> # After training model = EncoderLayer + LUTLayer
            >>> encoder.set_importance_from_lut(lut_layer, method='weighted', min_weight=0.1)
            >>> # Now encoder uses importance-weighted gradients (soft weighting)
        """
        self.importance_min_weight = min_weight
        importance = compute_bit_importance_from_lut(lut_layer, method=method, normalize=True)

        # Reshape from [total_bits] to [num_features, num_thresholds]
        num_features = self.thresholds.shape[0]
        num_thresholds = self.thresholds.shape[1]
        importance_per_feature = importance.view(num_features, num_thresholds).detach()

        # Store as buffer (moves with model to GPU but doesn't train)
        if hasattr(self, 'importance_weights'):
            # Update existing buffer
            self.importance_weights.data.copy_(importance_per_feature)
        else:
            # Register new buffer
            self.register_buffer('importance_weights', importance_per_feature)
        print(f"Set importance weights: shape={importance_per_feature.shape}, "
              f"range=[{importance_per_feature.min():.4f}, {importance_per_feature.max():.4f}]")

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

        if self.estimator_type == 'finite_difference':
            # Clamp and sort thresholds to keep them in valid range [-1, 1]
            if self.training:
                with torch.no_grad():
                    self.thresholds.clamp_(-1, 1)

            sorted_thresholds = torch.sort(self.thresholds, dim=1)[0]
            x_expanded = x.unsqueeze(-1)

            # Use importance-weighted estimator if importance weights are set
            if hasattr(self, 'importance_weights') and self.importance_weights is not None:
                # Also need to sort importance weights to match sorted thresholds
                sort_indices = torch.sort(self.thresholds, dim=1)[1]
                sorted_importance = torch.gather(self.importance_weights, 1, sort_indices)
                encoded = ImportanceWeightedFiniteDifferenceEstimator.apply(
                    x_expanded, sorted_thresholds, sorted_importance, self.importance_min_weight
                )
            else:
                encoded = FiniteDifferenceEstimator.apply(x_expanded, sorted_thresholds)

        elif self.estimator_type == 'glt':
            # GLT estimator with latent thresholds per feature
            # Note: GLT transforms thresholds via cumsum to keep them in [-1, 1] automatically
            # But we clamp to be safe during training
            if self.training:
                with torch.no_grad():
                    self.thresholds.clamp_(-1, 1)

            # Process each feature with its own set of latent thresholds
            encoded_list = []

            # Check if importance weighting is enabled
            use_importance = hasattr(self, 'importance_weights') and self.importance_weights is not None

            for i in range(self.inputs):
                x_feature = x[:, i]  # [batch]
                latent_thresh_feature = self.thresholds[i]  # [num_thresholds]

                if use_importance:
                    # Use importance-weighted GLT estimator
                    importance_feature = self.importance_weights[i]  # [num_thresholds]
                    encoded_feature = ImportanceWeightedGLTEstimator.apply(
                        x_feature, latent_thresh_feature, importance_feature,
                        self.importance_min_weight, self.glt_m, self.glt_p
                    )
                else:
                    # Use standard GLT estimator
                    encoded_feature = GLTEstimator.apply(x_feature, latent_thresh_feature, self.glt_m, self.glt_p)

                encoded_list.append(encoded_feature)
            encoded = torch.stack(encoded_list, dim=1)  # [batch, features, thresholds]

        return encoded
