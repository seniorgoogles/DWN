import numpy as np
import torch
import torch.nn as nn

from .binarization import DistributiveThermometer, LbpThermometer, LbpDistributiveThermometer, LbpDoubleThermometer, Thermometer


def compute_lbp_batch(images):
    """
    Compute Local Binary Patterns for a batch of grayscale images.

    Args:
        images: Tensor of shape (batch, height, width) - grayscale images

    Returns:
        lbp_codes: Tensor of shape (batch, height, width) with LBP codes (0-255)
    """
    batch_size, h, w = images.shape

    # Zero-pad images: (B, H, W) → (B, H+2, W+2)
    padded = torch.nn.functional.pad(images, (1, 1, 1, 1), mode='constant', value=0)

    # Define 8 neighbor offsets (clockwise from top-left)
    # (dy, dx) pairs
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),  # top row
        (0, 1),                       # right
        (1, 1), (1, 0), (1, -1),     # bottom row
        (0, -1)                       # left
    ]

    # Initialize LBP codes
    lbp_codes = torch.zeros((batch_size, h, w), dtype=torch.float32, device=images.device)

    # Compute LBP for each pixel
    for i in range(h):
        for j in range(w):
            center = padded[:, i+1, j+1]  # (batch,)
            code = torch.zeros(batch_size, dtype=torch.float32, device=images.device)

            for bit_pos, (dy, dx) in enumerate(neighbors):
                neighbor_val = padded[:, i+1+dy, j+1+dx]  # (batch,)
                # Set bit if neighbor >= center
                bit_value = (neighbor_val >= center).float() * (2 ** bit_pos)
                code = code + bit_value

            lbp_codes[:, i, j] = code

    return lbp_codes


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
        thresholds_exp = thresholds.view(*([1] * x.ndim), -1)  # [1,1,1,1,M]
        x_exp = x.unsqueeze(-1)  # [B,C,H,W,1]

        # Thermometer encoding
        out = (x_exp >= thresholds_exp).float()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, thresholds = ctx.saved_tensors
        m = ctx.m
        p = ctx.p

        # Expand thresholds
        thresholds_exp = thresholds.view(*([1] * x.ndim), -1)
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


class EncoderLayer(nn.Module):
    """
    Encoder layer for continuous to binary conversion using learned thresholds.
    Supports thermometer encoding, LBP encoding, or combination of both.
    """

    def __init__(
        self,
        inputs,
        output_size,
        input_dataset=None,
        estimator_type="glt",
        encoding_type="thermometer",
        image_shape=None,
        thermo_type="distributive",
        glt_m=5,
        glt_p=2,
        **kwargs,
    ):
        """
        Args:
            inputs: Number of input features
            output_size: Number of thresholds per feature (for thermometer modes)
            input_dataset: Dataset to initialize thresholds (required for 'FiniteDifference')
            estimator_type: 'STE', 'ste', 'GLT', 'glt', 'FiniteDifference', or 'finite_difference' (default: 'glt')
            encoding_type: 'thermometer', 'lbp', or 'lbp+thermometer' (default: 'thermometer')
            image_shape: (height, width) tuple for LBP modes - required if encoding_type uses 'lbp'
            thermo_type: Type of thermometer encoding - 'uniform', 'gaussian', or 'distributive' (default: 'distributive')
            glt_m: Parameter m for GLT rectified STE (default: 5)
            glt_p: Parameter p for GLT rectified STE (default: 2)
        """
        super().__init__()

        # Validate and normalize encoding_type
        valid_encoding_types = ['thermometer', 'lbp', 'lbp+thermometer', 'lbp_distributive', 'lbp_double_thermo']
        if encoding_type not in valid_encoding_types:
            raise ValueError(
                f"Unknown encoding_type: '{encoding_type}'. "
                f"Choose from: {valid_encoding_types}"
            )
        self.encoding_type = encoding_type

        # Validate image_shape for LBP modes
        if 'lbp' in self.encoding_type or self.encoding_type in ['lbp_distributive', 'lbp_double_thermo']:
            if image_shape is None:
                raise ValueError(
                    f"encoding_type='{encoding_type}' requires image_shape=(height, width)"
                )
            self.image_shape = image_shape
        else:
            self.image_shape = None

        # Normalize estimator_type to handle multiple formats
        # Accept: 'STE', 'ste', 'GLT', 'glt', 'FiniteDifference', 'finite_difference'
        estimator_map = {
            "ste": "ste",
            "straightthrough": "ste",
            "glt": "glt",
            "finitedifference": "finite_difference",
            "finite_difference": "finite_difference",
        }
        normalized_type = estimator_type.lower().replace("_", "").replace("-", "")
        if normalized_type not in estimator_map:
            raise ValueError(
                f"Unknown estimator_type: '{estimator_type}'. "
                f"Choose 'STE', 'ste', 'GLT', 'glt', 'FiniteDifference', or 'finite_difference'."
            )
        self.estimator_type = estimator_map[normalized_type]
        self.glt_m = glt_m
        self.glt_p = glt_p
        self.inputs = inputs
        self.thermo_type = thermo_type
        self.output_size = output_size

        # Convert input_dataset to tensor (needed for thermometer initialization)
        if input_dataset is not None:
            if torch.is_tensor(input_dataset):
                x = input_dataset.detach().clone().float()
            else:
                x = torch.tensor(input_dataset, dtype=torch.float32)
        else:
            x = None

        # Initialize LBP+Thermometer encoder if needed
        if self.encoding_type == 'lbp+thermometer':
            if x is None:
                raise ValueError(
                    "EncoderLayer with 'lbp+thermometer' requires 'input_dataset' to initialize thresholds."
                )

            # Reshape data to image format for LbpThermometer
            # x is (batch, features), need to reshape to (batch, height, width)
            batch_size = x.shape[0]
            h, w = self.image_shape
            x_images = x.reshape(batch_size, h, w)

            # Initialize LbpThermometer
            self.lbp_thermometer = LbpThermometer(
                num_bits=output_size,
                feature_wise=False,
                thermo_type=self.thermo_type
            )
            self.lbp_thermometer.fit(x_images)

            # Store thresholds as parameter for compatibility
            self.thresholds = nn.Parameter(
                self.lbp_thermometer.thresholds.clone(),
                requires_grad=True
            )
            self.lbp_distributive = None
            self.lbp_double = None
            self.importance_min_weight = 0.1
            return

        # Initialize LBP -> DistributiveThermometer encoder
        if self.encoding_type == 'lbp_distributive':
            if x is None:
                raise ValueError(
                    "EncoderLayer with 'lbp_distributive' requires 'input_dataset' to initialize thresholds."
                )

            # Reshape data to image format
            batch_size = x.shape[0]
            h, w = self.image_shape
            x_images = x.reshape(batch_size, h, w)

            # Initialize LbpDistributiveThermometer - fits on LBP-encoded data
            self.lbp_distributive = LbpDistributiveThermometer(
                num_bits=output_size,
                feature_wise=False
            )
            self.lbp_distributive.fit(x_images)

            # Store thresholds as parameter
            self.thresholds = nn.Parameter(
                self.lbp_distributive.thresholds.clone(),
                requires_grad=True
            )
            self.lbp_thermometer = None
            self.lbp_double = None
            self.importance_min_weight = 0.1
            return

        # Initialize LBP -> Double Thermometer encoder
        if self.encoding_type == 'lbp_double_thermo':
            if x is None:
                raise ValueError(
                    "EncoderLayer with 'lbp_double_thermo' requires 'input_dataset' to initialize thresholds."
                )

            # Reshape data to image format
            batch_size = x.shape[0]
            h, w = self.image_shape
            x_images = x.reshape(batch_size, h, w)

            # Initialize LbpDoubleThermometer - two-stage encoding
            self.lbp_double = LbpDoubleThermometer(
                num_bits1=output_size,  # First stage bits
                num_bits2=output_size,  # Second stage bits (same as first)
                feature_wise=False
            )
            self.lbp_double.fit(x_images)

            # Store thresholds2 as parameter (the final stage thresholds)
            self.thresholds = nn.Parameter(
                self.lbp_double.thresholds2.clone(),
                requires_grad=True
            )
            self.lbp_thermometer = None
            self.lbp_distributive = None
            self.importance_min_weight = 0.1
            return

        # Initialize thresholds only for thermometer-based encodings
        if self.encoding_type == 'thermometer':
            self.lbp_thermometer = None
            self.lbp_distributive = None
            self.lbp_double = None
        else:
            self.thresholds = None
            self.importance_min_weight = 0.1
            self.lbp_thermometer = None
            self.lbp_distributive = None
            self.lbp_double = None
            return

        if self.estimator_type == "ste":
            if x is not None:
                # Initialize with distributive thresholds from data
                thermometer = DistributiveThermometer(
                    num_bits=output_size, feature_wise=True
                )
                thermometer.fit(x)
                thresholds = thermometer.thresholds
                if not torch.is_tensor(thresholds):
                    thresholds = torch.tensor(thresholds, dtype=torch.float32)
            else:
                # Initialize with uniform linspace if no dataset provided
                thresholds = (
                    torch.linspace(-0.9, 0.9, output_size)
                    .unsqueeze(0)
                    .repeat(inputs, 1)
                )

            self.thresholds = nn.Parameter(thresholds, requires_grad=True)

        elif self.estimator_type == "finite_difference":
            if x is None:
                raise ValueError(
                    "EncoderLayer with 'FiniteDifference' requires 'input_dataset' to initialize thresholds."
                )

            # Initialize with distributive thresholds
            thermometer = DistributiveThermometer(
                num_bits=output_size, feature_wise=True
            )
            thermometer.fit(x)
            thresholds = thermometer.thresholds

            if not torch.is_tensor(thresholds):
                thresholds = torch.tensor(thresholds, dtype=torch.float32)

            self.thresholds = nn.Parameter(thresholds, requires_grad=True)

        elif self.estimator_type == "glt":
            # GLT uses latent thresholds per feature
            if x is not None:
                # Initialize with Thermometer from data distribution
                thermometer = Thermometer(num_bits=output_size, feature_wise=True)
                thermometer.fit(x)
                thresholds = thermometer.thresholds
            else:
                # Initialize with uniform linspace if no dataset provided
                thresholds = (
                    torch.linspace(-0.9, 0.9, output_size)
                    .unsqueeze(0)
                    .repeat(inputs, 1)
                )
            self.thresholds = nn.Parameter(thresholds, requires_grad=True)

        # Initialize importance weights as buffer (optional, for importance-weighted training)
        # Don't set it here - will be set via set_importance_from_lut() if needed
        self.importance_min_weight = 0.1  # Default: unused bits get 10% of gradient

    def forward(self, x):
        """
        Encode input using the selected encoding type and estimator.

        Args:
            x: Input tensor
               - For LBP modes: (batch_size, features) where features = height * width
               - For thermometer: (batch_size, features)

        Returns:
            Encoded tensor
            - encoding_type='lbp': (batch_size, features) - LBP codes
            - encoding_type='thermometer': (batch_size, features, thresholds)
            - encoding_type='lbp+thermometer': (batch_size, features, thresholds)
        """
        # Convert to tensor if needed
        if not torch.is_tensor(x):
            device = self.thresholds.device if self.thresholds is not None else 'cpu'
            x = torch.tensor(x, dtype=torch.float32, device=device)
        else:
            device = self.thresholds.device if self.thresholds is not None else x.device
            x = x.to(device).float()

        # Handle LBP encoding modes
        if 'lbp' in self.encoding_type or self.encoding_type in ['lbp_distributive', 'lbp_double_thermo']:
            # Reshape flattened features back to images: (B, H*W) → (B, H, W)
            batch_size = x.shape[0]
            h, w = self.image_shape
            images = x.view(batch_size, h, w)

            if self.encoding_type == 'lbp':
                # LBP only: compute and return
                lbp_codes = compute_lbp_batch(images)  # (batch, height, width)
                return lbp_codes.view(batch_size, -1)  # (batch, height*width)

            elif self.encoding_type == 'lbp+thermometer':
                # Use LbpThermometer class for combined encoding
                # Update thresholds in the encoder to use learned values
                self.lbp_thermometer.thresholds = self.thresholds.data.clone()

                # Apply LBP + Thermometer encoding
                encoded = self.lbp_thermometer.binarize(images)  # (batch, height*width, 8+num_bits)

                # Return in format (batch, pixels, features) to match thermometer encoding
                # The nn.Flatten layer will flatten this to (batch, pixels*features)
                return encoded  # (batch, height*width, 8+num_bits)

            elif self.encoding_type == 'lbp_distributive':
                # LBP then distributive thermometer on LBP values
                # Update thresholds in the encoder to use learned values
                self.lbp_distributive.thresholds = self.thresholds.data.clone()

                # Apply LBP + Distributive Thermometer encoding
                encoded = self.lbp_distributive.binarize(images)  # (batch, height*width, num_bits)

                return encoded  # (batch, height*width, num_bits)

            elif self.encoding_type == 'lbp_double_thermo':
                # LBP → Thermo1 → sum → Thermo2
                # Update thresholds2 in the encoder to use learned values
                self.lbp_double.thresholds2 = self.thresholds.data.clone()

                # Apply two-stage encoding
                encoded = self.lbp_double.binarize(images)  # (batch, height*width, num_bits2)

                return encoded  # (batch, height*width, num_bits2)

        # Thermometer encoding (for 'thermometer' or 'lbp+thermometer')
        if self.estimator_type == "ste":
            # Straight-Through Estimator
            if self.training:
                with torch.no_grad():
                    self.thresholds.clamp_(-1, 1)

            sorted_thresholds = torch.sort(self.thresholds, dim=1)[0]
            x_expanded = x.unsqueeze(-1)
            encoded = StraightThroughEstimator.apply(x_expanded, sorted_thresholds)

        elif self.estimator_type == "finite_difference":
            # Clamp and sort thresholds to keep them in valid range [-1, 1]
            if self.training:
                with torch.no_grad():
                    self.thresholds.clamp_(-1, 1)

            sorted_thresholds = torch.sort(self.thresholds, dim=1)[0]
            x_expanded = x.unsqueeze(-1)
            encoded = FiniteDifferenceEstimator.apply(x_expanded, sorted_thresholds)

        elif self.estimator_type == "glt":
            # GLT estimator with latent thresholds per feature
            # Note: GLT transforms thresholds via cumsum to keep them in [-1, 1] automatically
            # But we clamp to be safe during training
            if self.training:
                with torch.no_grad():
                    self.thresholds.clamp_(-1, 1)

            # Process each feature with its own set of latent thresholds
            encoded_list = []

            for i in range(self.inputs):
                x_feature = x[:, i]  # [batch]
                latent_thresh_feature = self.thresholds[i]  # [num_thresholds]
                encoded_feature = GLTEstimator.apply(
                    x_feature, latent_thresh_feature, self.glt_m, self.glt_p
                )
                encoded_list.append(encoded_feature)
            encoded = torch.stack(encoded_list, dim=1)  # [batch, features, thresholds]

        return encoded
