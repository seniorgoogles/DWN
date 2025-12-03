"""
LBP Encoding Example
Demonstrates different encoding strategies on image datasets.
Shows images before and after encoding with multiple encoders.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

# ============================================================================
# Helper Functions for Image Processing
# ============================================================================


def rgb_to_grayscale(images):
    """
    Convert RGB images to grayscale using luminosity method.

    Args:
        images: (N, H, W, 3) numpy array
    Returns:
        (N, H, W) grayscale images
    """
    return (
        0.299 * images[:, :, :, 0]
        + 0.587 * images[:, :, :, 1]
        + 0.114 * images[:, :, :, 2]
    )


def clip_and_normalize(images, sigma=3.5):
    """
    Clip to ±sigma standard deviations and normalize to [-1, 1).

    Args:
        images: (N, ...) array or tensor
        sigma: Number of standard deviations for clipping
    Returns:
        Normalized images in [-1, 1)
    """
    # Convert to tensor if needed
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)

    # Clip to ±sigma standard deviations
    mean = images.mean()
    std = images.std()
    clip_min = mean - sigma * std
    clip_max = mean + sigma * std
    images = torch.clamp(images, clip_min, clip_max)

    # Normalize to [-1, 1)
    data_min = images.min()
    data_max = images.max()
    data_range = data_max - data_min

    # Avoid division by zero
    if data_range == 0:
        return torch.zeros_like(images)

    images = 2 * (images - data_min) / data_range - 1

    return images


# ============================================================================
# Encoder Implementations
# ============================================================================


class BaseEncoder:
    """Base class for all encoders (follows binarization.py interface)"""

    def __init__(self, name):
        self.name = name

    def fit(self, x):
        """Fit encoder to data (optional, can be no-op)"""
        return self

    def encode(self, x):
        """Encode input - must be implemented by subclasses"""
        raise NotImplementedError


class LBPEncoder(BaseEncoder):
    """Local Binary Patterns encoder"""

    def __init__(self, image_shape, normalize_output=False):
        super().__init__("LBP")
        self.image_shape = image_shape
        self.normalize_output = normalize_output

    def encode(self, x):
        """
        Apply LBP transformation.

        Args:
            x: (batch, height, width) grayscale images
        Returns:
            (batch, height, width) LBP codes [0-255] or normalized
        """
        batch_size, h, w = x.shape

        # Zero-pad images
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=0)

        # 8 neighbors (clockwise from top-left)
        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]

        lbp_codes = torch.zeros((batch_size, h, w), dtype=torch.float32)

        # Compute LBP for each pixel
        for i in range(h):
            for j in range(w):
                center = padded[:, i + 1, j + 1]
                code = torch.zeros(batch_size, dtype=torch.float32)

                for bit_pos, (dy, dx) in enumerate(neighbors):
                    neighbor_val = padded[:, i + 1 + dy, j + 1 + dx]
                    bit_value = (neighbor_val >= center).float() * (2**bit_pos)
                    code = code + bit_value

                lbp_codes[:, i, j] = code

        # Normalize if requested
        if self.normalize_output:
            lbp_codes = (lbp_codes / 127.5) - 1.0

        return lbp_codes


class ThermometerEncoder(BaseEncoder):
    """Simplified thermometer encoder for visualization"""

    def __init__(self, num_thresholds=8):
        super().__init__(f"Thermometer({num_thresholds})")
        self.num_thresholds = num_thresholds
        self.thresholds = None

    def fit(self, x):
        """Initialize thresholds from data distribution"""
        # Use quantiles for threshold initialization
        flattened = x.flatten()
        quantiles = torch.linspace(0, 1, self.num_thresholds + 2)[1:-1]
        self.thresholds = torch.quantile(flattened, quantiles)
        return self

    def encode(self, x):
        """
        Apply thermometer encoding - visualize as averaged thermometer.

        Args:
            x: (batch, height, width) images
        Returns:
            (batch, height, width) averaged thermometer encoding for visualization
        """
        original_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1, 1)  # (batch, pixels, 1)

        # Apply thresholds: (batch, pixels, thresholds)
        encoded = (x_flat >= self.thresholds.view(1, 1, -1)).float()

        # Average over thresholds for visualization (convert back to "image-like")
        averaged = encoded.mean(dim=2)  # (batch, pixels)

        return averaged.reshape(original_shape)


class DistributiveThermometerEncoder(BaseEncoder):
    """Distributive thermometer encoder with evenly spaced thresholds"""

    def __init__(self, num_thresholds=8):
        super().__init__(f"DistThermo({num_thresholds})")
        self.num_thresholds = num_thresholds
        self.thresholds = None

    def fit(self, x):
        """Initialize evenly spaced thresholds from data range"""
        # Use evenly spaced thresholds across the data range
        data_min = x.min()
        data_max = x.max()
        self.thresholds = torch.linspace(data_min, data_max, self.num_thresholds + 2)[
            1:-1
        ]
        return self

    def encode(self, x):
        """
        Apply distributive thermometer encoding - visualize as averaged thermometer.

        Args:
            x: (batch, height, width) images
        Returns:
            (batch, height, width) averaged thermometer encoding for visualization
        """
        original_shape = x.shape
        x_flat = x.reshape(x.shape[0], -1, 1)  # (batch, pixels, 1)

        # Apply thresholds: (batch, pixels, thresholds)
        encoded = (x_flat >= self.thresholds.view(1, 1, -1)).float()

        # Average over thresholds for visualization (convert back to "image-like")
        averaged = encoded.mean(dim=2)  # (batch, pixels)

        return averaged.reshape(original_shape)


class LBPThermometerEncoder(BaseEncoder):
    """Hybrid LBP + Thermometer encoder (parallel combination)"""

    def __init__(self, image_shape, num_thresholds=8, use_quantile=True):
        super().__init__(f"LBP+Thermo({num_thresholds})")
        self.image_shape = image_shape
        self.num_thresholds = num_thresholds
        self.use_quantile = use_quantile
        self.thresholds = None

    def fit(self, x):
        """Initialize thresholds for thermometer encoding"""
        if self.use_quantile:
            # Quantile-based thresholds
            flattened = x.flatten()
            quantiles = torch.linspace(0, 1, self.num_thresholds + 2)[1:-1]
            self.thresholds = torch.quantile(flattened, quantiles)
        else:
            # Evenly spaced thresholds
            data_min = x.min()
            data_max = x.max()
            self.thresholds = torch.linspace(
                data_min, data_max, self.num_thresholds + 2
            )[1:-1]
        return self

    def encode(self, x):
        """
        Apply LBP + Thermometer encoding in parallel.

        Args:
            x: (batch, height, width) grayscale images
        Returns:
            (batch, height, width, num_thresholds + 1) combined encoding
            For visualization, we average over the feature dimension
        """
        batch_size, h, w = x.shape

        # ====================================================================
        # Part 1: Compute LBP codes
        # ====================================================================
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=0)

        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]

        lbp_codes = torch.zeros((batch_size, h, w), dtype=torch.float32)

        for i in range(h):
            for j in range(w):
                center = padded[:, i + 1, j + 1]
                code = torch.zeros(batch_size, dtype=torch.float32)

                for bit_pos, (dy, dx) in enumerate(neighbors):
                    neighbor_val = padded[:, i + 1 + dy, j + 1 + dx]
                    bit_value = (neighbor_val >= center).float() * (2**bit_pos)
                    code = code + bit_value

                lbp_codes[:, i, j] = code

        # Normalize LBP codes to [0, 1]
        lbp_codes = lbp_codes / 255.0

        # ====================================================================
        # Part 2: Compute Thermometer encoding for center pixels
        # ====================================================================
        # Reshape for thermometer encoding: (batch, h*w, 1)
        x_flat = x.reshape(batch_size, -1, 1)

        # Apply thresholds: (batch, h*w, num_thresholds)
        thermo_codes = (x_flat >= self.thresholds.view(1, 1, -1)).float()

        # Reshape back: (batch, h, w, num_thresholds)
        thermo_codes = thermo_codes.reshape(batch_size, h, w, self.num_thresholds)

        # ====================================================================
        # Part 3: Concatenate LBP and Thermometer codes
        # ====================================================================
        # Add LBP as an additional channel: (batch, h, w, 1)
        lbp_codes = lbp_codes.unsqueeze(-1)

        # Concatenate: (batch, h, w, num_thresholds + 1)
        combined = torch.cat([lbp_codes, thermo_codes], dim=-1)

        # For visualization, average over the feature dimension
        # This gives us a (batch, h, w) image-like output
        output = combined.mean(dim=-1)

        return output


class CompressedLBPThermometerEncoder(BaseEncoder):
    """Compressed LBP+Thermo encoder with stride for spatial downsampling"""

    def __init__(self, image_shape, num_thresholds=8, stride=2, pool_mode="mean"):
        super().__init__(f"Compressed_LBP+Thermo(s={stride})")
        self.image_shape = image_shape
        self.num_thresholds = num_thresholds
        self.stride = stride
        self.pool_mode = pool_mode  # 'mean', 'max', or 'center'
        self.thresholds = None

    def fit(self, x):
        """Initialize thresholds for thermometer encoding"""
        flattened = x.flatten()
        quantiles = torch.linspace(0, 1, self.num_thresholds + 2)[1:-1]
        self.thresholds = torch.quantile(flattened, quantiles)
        return self

    def encode(self, x):
        """
        Apply compressed LBP + Thermometer encoding with stride.

        Args:
            x: (batch, height, width) grayscale images
        Returns:
            (batch, height//stride, width//stride) compressed encoding
        """
        batch_size, h, w = x.shape

        # Calculate output dimensions
        out_h = h // self.stride
        out_w = w // self.stride

        # ====================================================================
        # Compute full LBP+Thermo first
        # ====================================================================
        # LBP computation
        padded = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="constant", value=0)

        neighbors = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]

        lbp_codes = torch.zeros((batch_size, h, w), dtype=torch.float32)

        for i in range(h):
            for j in range(w):
                center = padded[:, i + 1, j + 1]
                code = torch.zeros(batch_size, dtype=torch.float32)

                for bit_pos, (dy, dx) in enumerate(neighbors):
                    neighbor_val = padded[:, i + 1 + dy, j + 1 + dx]
                    bit_value = (neighbor_val >= center).float() * (2**bit_pos)
                    code = code + bit_value

                lbp_codes[:, i, j] = code

        # Normalize LBP codes to [0, 1]
        lbp_codes = lbp_codes / 255.0

        # Thermometer encoding
        x_flat = x.reshape(batch_size, -1, 1)
        thermo_codes = (x_flat >= self.thresholds.view(1, 1, -1)).float()
        thermo_codes = thermo_codes.reshape(batch_size, h, w, self.num_thresholds)

        # Combine LBP and Thermometer
        lbp_codes_expanded = lbp_codes.unsqueeze(-1)
        combined = torch.cat(
            [lbp_codes_expanded, thermo_codes], dim=-1
        )  # (batch, h, w, num_thresholds+1)

        # Average over feature dimension for each pixel
        full_encoding = combined.mean(dim=-1)  # (batch, h, w)

        # ====================================================================
        # Apply spatial compression with stride
        # ====================================================================
        compressed = torch.zeros((batch_size, out_h, out_w), dtype=torch.float32)

        for i in range(out_h):
            for j in range(out_w):
                # Extract patch
                i_start = i * self.stride
                j_start = j * self.stride
                i_end = min(i_start + self.stride, h)
                j_end = min(j_start + self.stride, w)

                patch = full_encoding[:, i_start:i_end, j_start:j_end]

                # Apply pooling
                if self.pool_mode == "mean":
                    compressed[:, i, j] = patch.mean(dim=(1, 2))
                elif self.pool_mode == "max":
                    compressed[:, i, j] = patch.reshape(batch_size, -1).max(dim=1)[0]
                elif self.pool_mode == "center":
                    # Take center pixel value
                    center_i = (i_end - i_start) // 2
                    center_j = (j_end - j_start) // 2
                    compressed[:, i, j] = patch[:, center_i, center_j]

        return compressed


class IdentityEncoder(BaseEncoder):
    """Identity encoder (no transformation) for comparison"""

    def __init__(self):
        super().__init__("Identity")

    def encode(self, x):
        return x


class BinaryThresholdEncoder(BaseEncoder):
    """Simple binary threshold at 0.0"""

    def __init__(self):
        super().__init__("Binary@0.0")

    def encode(self, x):
        return (x >= 0.0).float()


# ============================================================================
# Image Saving Utilities
# ============================================================================

def save_image_by_channels(image, filename, normalize=True):
    """
    Save image as PNG based on number of channels.

    Args:
        image: numpy array or tensor of shape (H, W) or (H, W, C)
        filename: output filename (without extension)
        normalize: whether to normalize to [0, 255] range
    """
    import numpy as np
    from PIL import Image

    # Convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    # Normalize to [0, 255] if requested
    if normalize:
        img_min = image.min()
        img_max = image.max()
        if img_max - img_min > 0:
            image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            image = np.zeros_like(image, dtype=np.uint8)
    else:
        image = image.astype(np.uint8)

    # Handle different channel counts
    if image.ndim == 2:
        # Grayscale (H, W)
        img = Image.fromarray(image, mode='L')
        img.save(f"{filename}.png")
        print(f"  Saved grayscale image: {filename}.png")

    elif image.ndim == 3:
        num_channels = image.shape[-1]

        if num_channels == 1:
            # Single channel (H, W, 1) -> treat as grayscale
            img = Image.fromarray(image.squeeze(-1), mode='L')
            img.save(f"{filename}.png")
            print(f"  Saved grayscale image: {filename}.png")

        elif num_channels == 3:
            # RGB (H, W, 3)
            img = Image.fromarray(image, mode='RGB')
            img.save(f"{filename}.png")
            print(f"  Saved RGB image: {filename}.png")

        else:
            # Multiple channels - save each separately
            for c in range(num_channels):
                img = Image.fromarray(image[:, :, c], mode='L')
                img.save(f"{filename}_channel_{c}.png")
            print(f"  Saved {num_channels} channel images: {filename}_channel_*.png")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


# ============================================================================
# Visualization
# ============================================================================


def plot_encoded_results(original_images, encoders, encoded_results, num_images=5):
    """
    Plot original images and their encoded versions.

    Args:
        original_images: (N, H, W) original grayscale images
        encoders: List of encoder objects
        encoded_results: List of (N, H, W) encoded images
        num_images: Number of images to display
    """
    num_encoders = len(encoders)
    fig, axes = plt.subplots(
        num_images, num_encoders + 1, figsize=(3 * (num_encoders + 1), 3 * num_images)
    )

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for img_idx in range(num_images):
        # Plot original image
        ax = axes[img_idx, 0]
        ax.imshow(original_images[img_idx], cmap="gray", vmin=-1, vmax=1)
        if img_idx == 0:
            ax.set_title("Original\n(normalized)", fontsize=10, fontweight="bold")
        ax.axis("off")

        # Plot encoded versions
        for enc_idx, (encoder, encoded) in enumerate(zip(encoders, encoded_results)):
            ax = axes[img_idx, enc_idx + 1]

            # Determine colormap and range based on encoder type
            if (
                "LBP" in encoder.name
                and hasattr(encoder, "normalize_output")
                and not encoder.normalize_output
            ):
                # LBP codes [0-255]
                ax.imshow(encoded[img_idx], cmap="viridis", vmin=0, vmax=255)
            else:
                # Normalized data
                ax.imshow(encoded[img_idx], cmap="gray", vmin=-1, vmax=1)

            if img_idx == 0:
                ax.set_title(f"{encoder.name}", fontsize=10, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("lbp_encoding_comparison.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved visualization to: lbp_encoding_comparison.png")
    plt.close()


def plot_statistics(original_images, encoders, encoded_results):
    """Plot histograms showing value distributions before/after encoding"""
    num_encoders = len(encoders)
    fig, axes = plt.subplots(1, num_encoders + 1, figsize=(4 * (num_encoders + 1), 4))

    # Original distribution
    axes[0].hist(
        original_images.flatten().numpy(),
        bins=50,
        color="blue",
        alpha=0.7,
        edgecolor="black",
    )
    axes[0].set_title("Original\n(normalized)", fontweight="bold")
    axes[0].set_xlabel("Value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True, alpha=0.3)

    # Encoded distributions
    for enc_idx, (encoder, encoded) in enumerate(zip(encoders, encoded_results)):
        axes[enc_idx + 1].hist(
            encoded.flatten().numpy(),
            bins=50,
            color="green",
            alpha=0.7,
            edgecolor="black",
        )
        axes[enc_idx + 1].set_title(f"{encoder.name}", fontweight="bold")
        axes[enc_idx + 1].set_xlabel("Value")
        axes[enc_idx + 1].set_ylabel("Frequency")
        axes[enc_idx + 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lbp_encoding_distributions.png", dpi=150, bbox_inches="tight")
    print(f"✓ Saved distributions to: lbp_encoding_distributions.png")
    plt.close()


# ============================================================================
# Main Script
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="LBP Encoding Example")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "mnist"],
        help="Dataset to use (default: cifar10)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=5,
        help="Number of images to visualize (default: 5)",
    )
    parser.add_argument(
        "--sigma", type=float, default=3.5, help="Clipping sigma (default: 3.5)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Data directory (default: ./data)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("LBP Encoding Example")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Number of images: {args.num_images}")
    print(f"  - Clipping sigma: {args.sigma}")

    # ========================================================================
    # Load Dataset
    # ========================================================================
    print(f"\nLoading {args.dataset.upper()} dataset...")

    if args.dataset == "cifar10":
        dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True)
        images = dataset.data[: args.num_images].astype(np.float32)  # (N, 32, 32, 3)

        # Convert to grayscale
        print("Converting RGB to grayscale...")
        images = rgb_to_grayscale(images)  # (N, 32, 32)
        image_shape = (32, 32)

    else:  # mnist
        dataset = datasets.MNIST(root=args.data_dir, train=True, download=True)
        images = (
            dataset.data[: args.num_images].numpy().astype(np.float32)
        )  # (N, 28, 28)
        image_shape = (28, 28)

    print(f"  - Loaded {args.num_images} images with shape: {images.shape}")

    # ========================================================================
    # Preprocessing
    # ========================================================================
    print(f"\nPreprocessing:")
    print(f"  - Clipping to ±{args.sigma} standard deviations...")
    print(f"  - Normalizing to [-1, 1)...")

    images = clip_and_normalize(images, sigma=args.sigma)

    print(f"  - Min value: {images.min().item():.4f}")
    print(f"  - Max value: {images.max().item():.4f}")
    print(f"  - Mean value: {images.mean().item():.4f}")
    print(f"  - Std value: {images.std().item():.4f}")

    # ========================================================================
    # Define Encoders to Test
    # ========================================================================
    print("\nDefining encoders...")

    encoders = [
        # LBPEncoder(image_shape=image_shape, normalize_output=True),
        # LBPEncoder(image_shape=image_shape, normalize_output=False),
        ThermometerEncoder(num_thresholds=32),
        DistributiveThermometerEncoder(num_thresholds=32),
        LBPThermometerEncoder(
            image_shape=image_shape, num_thresholds=32, use_quantile=True
        ),
        CompressedLBPThermometerEncoder(
            image_shape=image_shape, num_thresholds=32, stride=2, pool_mode="mean"
        ),
        CompressedLBPThermometerEncoder(
            image_shape=image_shape, num_thresholds=32, stride=3, pool_mode="mean"
        ),
        CompressedLBPThermometerEncoder(
            image_shape=image_shape, num_thresholds=32, stride=1, pool_mode="mean"
        ),
        # BinaryThresholdEncoder(),
    ]

    print(f"  - Testing {len(encoders)} encoders:")
    for encoder in encoders:
        print(f"    • {encoder.name}")

    # ========================================================================
    # Fit Encoders (if needed)
    # ========================================================================
    print("\nFitting encoders to data...")
    for encoder in encoders:
        encoder.fit(images)
        print(f"  ✓ {encoder.name}")

    # ========================================================================
    # Encode Images
    # ========================================================================
    print("\nEncoding images...")
    encoded_results = []

    for encoder in encoders:
        encoded = encoder.encode(images)
        encoded_results.append(encoded)
        print(
            f"  ✓ {encoder.name}: shape={encoded.shape}, "
            f"range=[{encoded.min().item():.2f}, {encoded.max().item():.2f}]"
        )

    # ========================================================================
    # Visualize Results
    # ========================================================================
    print("\nGenerating visualizations...")
    plot_encoded_results(images, encoders, encoded_results, num_images=args.num_images)
    plot_statistics(images, encoders, encoded_results)

    # ========================================================================
    # Summary Statistics
    # ========================================================================
    print("\n" + "=" * 70)
    print("ENCODING STATISTICS")
    print("=" * 70)

    for encoder, encoded in zip(encoders, encoded_results):
        print(f"\n{encoder.name}:")
        print(f"  - Output shape: {encoded.shape}")
        print(
            f"  - Value range: [{encoded.min().item():.4f}, {encoded.max().item():.4f}]"
        )
        print(f"  - Mean: {encoded.mean().item():.4f}")
        print(f"  - Std: {encoded.std().item():.4f}")
        print(f"  - Unique values: {len(torch.unique(encoded))}")

    print("\n" + "=" * 70)
    print("Example complete! Check the generated PNG files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
