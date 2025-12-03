from abc import ABC

import torch


class ThermometerBase(ABC):
    def __init__(self, num_bits=1, feature_wise=True):
        pass

    def get_thresholds(self, x):
        pass

    def fit(self, x):
        pass

    def binarize(self, x):
        pass


class Thermometer(ThermometerBase):
    def __init__(self, num_bits=1, feature_wise=True):
        assert num_bits > 0
        assert type(feature_wise) is bool

        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.thresholds = None

    def get_thresholds(self, x):
        min_value = x.min(dim=0)[0] if self.feature_wise else x.min()
        max_value = x.max(dim=0)[0] if self.feature_wise else x.max()
        return min_value.unsqueeze(-1) + torch.arange(1, self.num_bits + 1).unsqueeze(
            0
        ) * ((max_value - min_value) / (self.num_bits + 1)).unsqueeze(-1)

    def fit(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        self.thresholds = self.get_thresholds(x)
        return self

    def binarize(self, x):
        if self.thresholds is None:
            raise "need to fit before calling apply"
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        x = x.unsqueeze(-1)
        return (x > self.thresholds).float()


class GaussianThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        std_skews = torch.distributions.Normal(0, 1).icdf(
            torch.arange(1, self.num_bits + 1) / (self.num_bits + 1)
        )
        mean = x.mean(dim=0) if self.feature_wise else x.mean()
        std = x.std(dim=0) if self.feature_wise else x.std()
        thresholds = torch.stack(
            [std_skew * std + mean for std_skew in std_skews], dim=-1
        )
        return thresholds


class DistributiveThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        data = (
            torch.sort(x.flatten())[0]
            if not self.feature_wise
            else torch.sort(x, dim=0)[0]
        )
        indicies = torch.tensor(
            [
                int(data.shape[0] * i / (self.num_bits + 1))
                for i in range(1, self.num_bits + 1)
            ]
        )
        thresholds = data[indicies]
        return torch.permute(thresholds, (*list(range(1, thresholds.ndim)), 0))


class LbpDistributiveThermometer(Thermometer):
    """
    LBP encoding followed by DistributiveThermometer on the LBP values.

    This fits the thermometer thresholds on LBP-encoded data (values 0-255),
    not on the original pixel values.
    """
    def __init__(self, num_bits=8, feature_wise=False):
        """
        Args:
            num_bits: number of thermometer thresholds for LBP values
            feature_wise: whether to compute thresholds per pixel position
        """
        super().__init__(num_bits, feature_wise)
        self.neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    def _compute_lbp_codes(self, x):
        """
        Compute LBP codes (0-255) for spatial data.

        Args:
            x: (batch, height, width) grayscale images
        Returns:
            lbp_codes: (batch, height, width) with values 0-255
        """
        batch_size, h, w = x.shape
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        center = padded[:, 1:h+1, 1:w+1]

        lbp_codes = torch.zeros((batch_size, h, w), dtype=torch.float32, device=x.device)

        for bit_pos, (dy, dx) in enumerate(self.neighbors):
            neighbor = padded[:, 1+dy:h+1+dy, 1+dx:w+1+dx]
            lbp_codes = lbp_codes + (neighbor >= center).float() * (2 ** bit_pos)

        return lbp_codes

    def fit(self, x):
        """
        Fit thresholds on LBP-encoded data.

        Args:
            x: (batch, height, width) grayscale images or (batch, features) flattened
        """
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)

        # If input is flattened, we need image_shape to be set externally
        if x.ndim == 2:
            raise ValueError("LbpDistributiveThermometer.fit() requires image data with shape (batch, height, width)")

        # Compute LBP codes on the input data
        lbp_codes = self._compute_lbp_codes(x)  # (batch, height, width)

        # Fit distributive thresholds on LBP values
        if self.feature_wise:
            # Per-pixel thresholds
            lbp_flat = lbp_codes.reshape(-1, lbp_codes.shape[1] * lbp_codes.shape[2])  # (batch, h*w)
            data = torch.sort(lbp_flat, dim=0)[0]
            indices = torch.tensor([
                int(data.shape[0] * i / (self.num_bits + 1))
                for i in range(1, self.num_bits + 1)
            ])
            self.thresholds = data[indices].T  # (h*w, num_bits)
        else:
            # Global thresholds across all LBP values
            data = torch.sort(lbp_codes.flatten())[0]
            indices = torch.tensor([
                int(data.shape[0] * i / (self.num_bits + 1))
                for i in range(1, self.num_bits + 1)
            ])
            self.thresholds = data[indices]  # (num_bits,)

        return self

    def binarize(self, x):
        """
        Apply LBP then thermometer encoding on LBP values.

        Args:
            x: (batch, height, width) grayscale images
        Returns:
            encoded: (batch, height*width, num_bits) thermometer-encoded LBP values
        """
        if self.thresholds is None:
            raise ValueError("Need to fit before calling binarize")

        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)

        batch_size, h, w = x.shape

        # Compute LBP codes
        lbp_codes = self._compute_lbp_codes(x)  # (batch, h, w)
        lbp_flat = lbp_codes.reshape(batch_size, -1)  # (batch, h*w)

        # Apply thermometer encoding on LBP values
        thresholds_device = self.thresholds.to(x.device)

        if self.feature_wise:
            # Per-pixel thresholds: (h*w, num_bits)
            thermo_codes = (lbp_flat.unsqueeze(-1) >= thresholds_device.unsqueeze(0)).float()
        else:
            # Global thresholds: (num_bits,)
            thermo_codes = (lbp_flat.unsqueeze(-1) >= thresholds_device.view(1, 1, -1)).float()

        return thermo_codes  # (batch, h*w, num_bits)


class LbpDoubleThermometer(Thermometer):
    """
    LBP → Thermo1 (on LBP values) → convert to float → Thermo2 (on floats).

    Two-stage encoding:
    1. Compute LBP codes, apply first distributive thermometer
    2. Sum the binary bits to get float values
    3. Apply second distributive thermometer on those floats
    """
    def __init__(self, num_bits1=8, num_bits2=8, feature_wise=False):
        """
        Args:
            num_bits1: number of thermometer thresholds for first stage (on LBP values)
            num_bits2: number of thermometer thresholds for second stage (on summed values)
            feature_wise: whether to compute thresholds per pixel position
        """
        super().__init__(num_bits1, feature_wise)
        self.num_bits1 = num_bits1
        self.num_bits2 = num_bits2
        self.neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        self.thresholds1 = None  # First stage thresholds (on LBP)
        self.thresholds2 = None  # Second stage thresholds (on summed values)

    def _compute_lbp_codes(self, x):
        """Compute LBP codes (0-255) for spatial data."""
        batch_size, h, w = x.shape
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)
        center = padded[:, 1:h+1, 1:w+1]

        lbp_codes = torch.zeros((batch_size, h, w), dtype=torch.float32, device=x.device)

        for bit_pos, (dy, dx) in enumerate(self.neighbors):
            neighbor = padded[:, 1+dy:h+1+dy, 1+dx:w+1+dx]
            lbp_codes = lbp_codes + (neighbor >= center).float() * (2 ** bit_pos)

        return lbp_codes

    def fit(self, x):
        """
        Fit both stages of thresholds.

        Args:
            x: (batch, height, width) grayscale images
        """
        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 2:
            raise ValueError("LbpDoubleThermometer.fit() requires image data with shape (batch, height, width)")

        # Stage 1: Compute LBP codes and fit first thresholds
        lbp_codes = self._compute_lbp_codes(x)  # (batch, h, w)

        # Fit thresholds1 on LBP values (distributive)
        data1 = torch.sort(lbp_codes.flatten())[0]
        indices1 = torch.tensor([
            int(data1.shape[0] * i / (self.num_bits1 + 1))
            for i in range(1, self.num_bits1 + 1)
        ])
        self.thresholds1 = data1[indices1]  # (num_bits1,)

        # Stage 2: Apply first thermometer and sum to get floats
        lbp_flat = lbp_codes.reshape(-1)  # flatten all
        thermo1_output = (lbp_flat.unsqueeze(-1) >= self.thresholds1.view(1, -1)).float()  # (N, num_bits1)
        summed_values = thermo1_output.sum(dim=-1)  # (N,) - values in range [0, num_bits1]

        # Fit thresholds2 on summed values (distributive)
        data2 = torch.sort(summed_values)[0]
        indices2 = torch.tensor([
            int(data2.shape[0] * i / (self.num_bits2 + 1))
            for i in range(1, self.num_bits2 + 1)
        ])
        self.thresholds2 = data2[indices2]  # (num_bits2,)

        # Store combined thresholds for compatibility
        self.thresholds = self.thresholds2

        return self

    def binarize(self, x):
        """
        Apply two-stage encoding.

        Args:
            x: (batch, height, width) grayscale images
        Returns:
            encoded: (batch, height*width, num_bits2) final thermometer encoding
        """
        if self.thresholds1 is None or self.thresholds2 is None:
            raise ValueError("Need to fit before calling binarize")

        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)

        batch_size, h, w = x.shape
        num_pixels = h * w

        # Stage 1: LBP → Thermo1
        lbp_codes = self._compute_lbp_codes(x)  # (batch, h, w)
        lbp_flat = lbp_codes.reshape(batch_size, -1)  # (batch, h*w)

        thresholds1_device = self.thresholds1.to(x.device)
        thermo1_output = (lbp_flat.unsqueeze(-1) >= thresholds1_device.view(1, 1, -1)).float()  # (batch, h*w, num_bits1)

        # Convert to float: sum the binary bits
        summed_values = thermo1_output.sum(dim=-1)  # (batch, h*w)

        # Stage 2: Apply second thermometer on summed values
        thresholds2_device = self.thresholds2.to(x.device)
        thermo2_output = (summed_values.unsqueeze(-1) >= thresholds2_device.view(1, 1, -1)).float()  # (batch, h*w, num_bits2)

        return thermo2_output  # (batch, h*w, num_bits2)


class LbpThermometer(Thermometer):
    """
    Combined LBP + Thermometer encoding for image data.

    Expects input with spatial structure (batch, height, width) or (batch, channels, height, width).
    Returns concatenated LBP codes and thermometer encoding.
    """
    def __init__(self, num_bits=1, feature_wise=False, stride=1, padding=0, thermo_type='distributive'):
        """
        Args:
            num_bits: number of thermometer thresholds
            feature_wise: whether to compute thresholds per feature
            stride: stride for spatial operations (not yet implemented)
            padding: padding for spatial operations (not yet implemented)
            thermo_type: type of thermometer encoding - 'uniform', 'gaussian', or 'distributive'
        """
        super().__init__(num_bits, feature_wise)
        self.stride = stride
        self.padding = padding
        self.thermo_type = thermo_type
        self.neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    def get_thresholds(self, x):
        """Get thresholds based on selected thermometer type."""
        if self.thermo_type == 'uniform':
            # Uniform/evenly spaced thresholds (basic Thermometer)
            min_value = x.min()
            max_value = x.max()
            thresholds = min_value + torch.arange(1, self.num_bits + 1) * ((max_value - min_value) / (self.num_bits + 1))
            return thresholds

        elif self.thermo_type == 'gaussian':
            # Gaussian-based thresholds
            std_skews = torch.distributions.Normal(0, 1).icdf(
                torch.arange(1, self.num_bits + 1) / (self.num_bits + 1)
            )
            mean = x.mean()
            std = x.std()
            thresholds = torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)
            return thresholds

        elif self.thermo_type == 'distributive':
            # Distributive/quantile-based thresholds
            data = torch.sort(x.flatten())[0]
            indices = torch.tensor([
                int(data.shape[0] * i / (self.num_bits + 1))
                for i in range(1, self.num_bits + 1)
            ])
            thresholds = data[indices]
            return thresholds

        else:
            raise ValueError(f"Unknown thermo_type: {self.thermo_type}. Use 'uniform', 'gaussian', or 'distributive'")

    def _compute_lbp(self, x):
        """
        Compute LBP codes for spatial data using vectorized operations.

        Args:
            x: (batch, height, width) or (batch, channels, height, width)
        Returns:
            lbp_binary: (batch, channels, height, width, 8) or (batch, height, width, 8)
        """
        if x.ndim == 3:
            # (batch, height, width)
            batch_size, h, w = x.shape
            channels = 1
            x = x.unsqueeze(1)  # (batch, 1, height, width)
        else:
            # (batch, channels, height, width)
            batch_size, channels, h, w = x.shape

        # Pad the input once for all channels
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='constant', value=0)

        # Extract center pixels (no padding offset needed after padding)
        center = padded[:, :, 1:h+1, 1:w+1]  # (batch, channels, h, w)

        # Initialize output for 8 binary bits per pixel
        lbp_binary = torch.zeros((batch_size, channels, h, w, 8), dtype=torch.float32, device=x.device)

        # Vectorized LBP computation: compare all 8 neighbors at once
        for bit_pos, (dy, dx) in enumerate(self.neighbors):
            # Extract neighbor values for all pixels at once
            neighbor = padded[:, :, 1+dy:h+1+dy, 1+dx:w+1+dx]  # (batch, channels, h, w)

            # Compare neighbor >= center and store as bit
            lbp_binary[:, :, :, :, bit_pos] = (neighbor >= center).float()

        if channels == 1:
            lbp_binary = lbp_binary.squeeze(1)  # (batch, height, width, 8)

        return lbp_binary  # (batch, channels, height, width, 8) or (batch, height, width, 8)

    def binarize(self, x):
        """
        Apply LBP + Thermometer encoding.

        Args:
            x: (batch, height, width) or (batch, channels, height, width) image data
        Returns:
            combined: Concatenated LBP bits (8 per channel) and thermometer features
                     For RGB: (batch, height*width, 3*8 + num_bits) = (batch, height*width, 24 + num_bits)
                     For grayscale: (batch, height*width, 8 + num_bits)
        """
        if self.thresholds is None:
            raise ValueError("Need to fit before calling binarize")

        if type(x) is not torch.Tensor:
            x = torch.tensor(x, dtype=torch.float32)

        batch_size = x.shape[0]

        # Determine if multi-channel or single channel
        if x.ndim == 3:
            # (batch, height, width) - single channel
            h, w = x.shape[1], x.shape[2]
            num_channels = 1
        else:
            # (batch, channels, height, width) - multi-channel
            num_channels, h, w = x.shape[1], x.shape[2], x.shape[3]

        num_pixels = h * w

        # Compute LBP codes for each channel (now returns 8 binary bits per pixel)
        lbp_binary = self._compute_lbp(x)  # (batch, channels, height, width, 8) or (batch, height, width, 8)

        if lbp_binary.ndim == 4:
            # Single channel: (batch, height, width, 8) -> (batch, height*width, 8)
            lbp_flat = lbp_binary.reshape(batch_size, num_pixels, 8)
        else:
            # Multi-channel: (batch, channels, height, width, 8) -> (batch, height*width, channels*8)
            # Rearrange to (batch, height, width, channels, 8) then flatten
            lbp_binary = lbp_binary.permute(0, 2, 3, 1, 4)  # (batch, height, width, channels, 8)
            lbp_flat = lbp_binary.reshape(batch_size, num_pixels, num_channels * 8)

        # Compute thermometer encoding on original pixel values
        # For RGB, we compute one thermometer code per pixel (using all channels)
        if x.ndim == 4:
            # Multi-channel: average channels or use intensity
            # Use luminosity-based grayscale conversion for RGB
            if num_channels == 3:
                # Assume RGB order
                grayscale = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            else:
                # Average channels
                grayscale = x.mean(dim=1)
            x_for_thermo = grayscale.reshape(batch_size, -1)  # (batch, height*width)
        else:
            # Single channel
            x_for_thermo = x.reshape(batch_size, -1)  # (batch, height*width)

        # Apply thresholds: (batch, height*width, num_bits)
        # Ensure thresholds are on the same device as input
        thresholds_device = self.thresholds.to(x.device)
        thermo_codes = (x_for_thermo.unsqueeze(-1) >= thresholds_device.view(1, 1, -1)).float()

        # Concatenate: (batch, height*width, num_channels*8 + num_bits)
        combined = torch.cat([lbp_flat, thermo_codes], dim=-1)

        return combined
