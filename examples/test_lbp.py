import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms
from PIL import Image

def lbp(image):
    """
    Compute Local Binary Pattern for each channel.

    Args:
        image: (H, W) grayscale or (H, W, C) multi-channel numpy array or tensor
    Returns:
        lbp_image: (H, W) or (H, W, C) array with LBP codes [0-255] for each channel
    """
    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        image = torch.tensor(image, dtype=torch.float32)

    neighbours = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

    # Handle grayscale (H, W) vs multi-channel (H, W, C)
    if image.ndim == 2:
        h, w = image.shape
        c = 1
        image = image.unsqueeze(-1)  # Convert to (H, W, 1)
    else:
        h, w, c = image.shape

    # Initialize output
    lbp_image = torch.zeros((h, w, c), dtype=torch.float32)

    # Process each channel
    for ch in range(c):
        channel = image[:, :, ch]

        # Pad the channel to handle borders
        padded = torch.nn.functional.pad(channel.unsqueeze(0), (1, 1, 1, 1), mode='constant', value=0).squeeze(0)

        # Compute LBP for each pixel
        for row in range(h):
            for col in range(w):
                center_val = padded[row + 1, col + 1]
                code = 0

                # Check all 8 neighbors
                for bit_pos, (dy, dx) in enumerate(neighbours):
                    neighbor_val = padded[row + 1 + dy, col + 1 + dx]
                    # Add weighted bit if neighbor >= center
                    if neighbor_val >= center_val:
                        code += (2 ** bit_pos)

                lbp_image[row, col, ch] = code

    # Return (H, W) for grayscale input, (H, W, C) for multi-channel
    if c == 1:
        lbp_image = lbp_image.squeeze(-1)

    return lbp_image.numpy() if isinstance(image, torch.Tensor) else lbp_image

def thermometer_encode(image, num_thresholds=8, use_quantile=True):
    """
    Apply thermometer encoding to image.

    Args:
        image: (H, W) grayscale or (H, W, C) multi-channel numpy array or tensor
        num_thresholds: number of thresholds to use
        use_quantile: if True, use quantile-based thresholds, else evenly spaced
    Returns:
        encoded: (H, W, num_thresholds) or (H, W, C, num_thresholds) binary encoding
        For visualization, average over threshold dimension
    """
    # Convert to tensor if needed
    if isinstance(image, np.ndarray):
        image = torch.tensor(image, dtype=torch.float32)

    # Handle grayscale (H, W) vs multi-channel (H, W, C)
    if image.ndim == 2:
        h, w = image.shape
        c = 1
        image = image.unsqueeze(-1)  # Convert to (H, W, 1)
    else:
        h, w, c = image.shape

    # Compute thresholds
    if use_quantile:
        # Quantile-based thresholds
        flattened = image.flatten()
        quantiles = torch.linspace(0, 1, num_thresholds + 2)[1:-1]
        thresholds = torch.quantile(flattened, quantiles)
    else:
        # Evenly spaced thresholds
        data_min = image.min()
        data_max = image.max()
        thresholds = torch.linspace(data_min, data_max, num_thresholds + 2)[1:-1]

    # Apply thermometer encoding
    # Reshape: (H, W, C) -> (H*W*C, 1)
    image_flat = image.reshape(-1, 1)

    # Compare against thresholds: (H*W*C, num_thresholds)
    encoded_flat = (image_flat >= thresholds.view(1, -1)).float()

    # Reshape back: (H, W, C, num_thresholds)
    encoded = encoded_flat.reshape(h, w, c, num_thresholds)

    # Average over thresholds for visualization: (H, W, C)
    encoded_vis = encoded.mean(dim=-1)

    # Return (H, W) for grayscale input, (H, W, C) for multi-channel
    if c == 1:
        encoded_vis = encoded_vis.squeeze(-1)

    return encoded_vis.numpy() if isinstance(image, torch.Tensor) else encoded_vis


def rgb_to_grayscale(images):
    """
    Convert RGB images to grayscale using luminosity method.

    Args:
        images: (H, W, 3) single image or (N, H, W, 3) batch of images
    Returns:
        (H, W) or (N, H, W) grayscale images
    """
    # Handle single image (H, W, 3)
    if isinstance(images, torch.Tensor):
        if images.ndim == 3:
            return 0.299 * images[:, :, 0] + 0.587 * images[:, :, 1] + 0.114 * images[:, :, 2]
        else:  # Batch (N, H, W, 3)
            return 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
    else:  # numpy array
        if images.ndim == 3:
            return 0.299 * images[:, :, 0] + 0.587 * images[:, :, 1] + 0.114 * images[:, :, 2]
        else:  # Batch (N, H, W, 3)
            return 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]


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


def save_image_by_channels(image, filename, normalize=True, scale_factor=8, interpolation='nearest'):
    """
    Save image as PNG based on number of channels.

    Args:
        image: numpy array or tensor of shape (H, W) or (H, W, C)
        filename: output filename (without extension)
        normalize: whether to normalize to [0, 255] range
        scale_factor: integer factor to upscale image (default: 8 for 32x32->256x256)
        interpolation: 'nearest' (pixelated) or 'bilinear' (smooth)
    """
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

    # Select interpolation method
    interp_method = Image.NEAREST if interpolation == 'nearest' else Image.BILINEAR

    # Handle different channel counts
    if image.ndim == 2:
        # Grayscale (H, W)
        img = Image.fromarray(image, mode='L')
        if scale_factor > 1:
            new_size = (img.width * scale_factor, img.height * scale_factor)
            img = img.resize(new_size, interp_method)
        img.save(f"{filename}.png")
        print(f"  Saved grayscale image: {filename}.png ({img.width}x{img.height})")

    elif image.ndim == 3:
        num_channels = image.shape[-1]

        if num_channels == 1:
            # Single channel (H, W, 1) -> treat as grayscale
            img = Image.fromarray(image.squeeze(-1), mode='L')
            if scale_factor > 1:
                new_size = (img.width * scale_factor, img.height * scale_factor)
                img = img.resize(new_size, interp_method)
            img.save(f"{filename}.png")
            print(f"  Saved grayscale image: {filename}.png ({img.width}x{img.height})")

        elif num_channels == 3:
            # RGB (H, W, 3)
            img = Image.fromarray(image, mode='RGB')
            if scale_factor > 1:
                new_size = (img.width * scale_factor, img.height * scale_factor)
                img = img.resize(new_size, interp_method)
            img.save(f"{filename}.png")
            print(f"  Saved RGB image: {filename}.png ({img.width}x{img.height})")

        else:
            # Multiple channels - save each separately
            for c in range(num_channels):
                img = Image.fromarray(image[:, :, c], mode='L')
                if scale_factor > 1:
                    new_size = (img.width * scale_factor, img.height * scale_factor)
                    img = img.resize(new_size, interp_method)
                img.save(f"{filename}_channel_{c}.png")
            print(f"  Saved {num_channels} channel images: {filename}_channel_*.png ({img.width}x{img.height})")
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


DATASET_FOLDER = "./data"
IMAGE_NUM = 5
SIGMA = 3.5

dataset = datasets.CIFAR10(root=DATASET_FOLDER, train=True, download=True)
images = dataset.data[: IMAGE_NUM - 1].astype(np.float32)  # (N, 32, 32, 3)
images = clip_and_normalize(images, sigma=SIGMA)

print(images[0].shape)

image = images[1]
#image = rgb_to_grayscale(image)

# Apply encodings
lbp_encoded = lbp(image)
thermo_encoded = thermometer_encode(image, num_thresholds=32, use_quantile=True)

print(f"Original image range: [{image.min():.2f}, {image.max():.2f}]")
print(f"LBP encoded range: [{lbp_encoded.min():.2f}, {lbp_encoded.max():.2f}]")
print(f"Thermometer encoded range: [{thermo_encoded.min():.2f}, {thermo_encoded.max():.2f}]")

# Save example images with different channel counts
print("\nSaving images...")

# Save original grayscale image
save_image_by_channels(image, "test_grayscale_original", normalize=True, scale_factor=8, interpolation='nearest')

# Save LBP encoded image (LBP codes 0-255)
save_image_by_channels(lbp_encoded, "test_lbp_encoded", normalize=False, scale_factor=8, interpolation='nearest')

# Save Thermometer encoded image (averaged binary codes 0-1)
save_image_by_channels(thermo_encoded, "test_thermo_encoded", normalize=True, scale_factor=8, interpolation='nearest')

# Save grayscale image (single channel)
# Uncomment the code below to test grayscale saving
"""
images_gray = rgb_to_grayscale(images)
save_image_by_channels(images_gray[0], "test_grayscale_image", normalize=True)
"""

"""
# Convert to grayscale
print("Converting RGB to grayscale...")
images = rgb_to_grayscale(images)  # (N, 32, 32)
image_shape = (32, 32)


toy_example = torch.Tensor([[1, 9, 9, 1], [2, 5, 7, 4], [5, 3, 1, 5], [3,4,5,3]])

print(toy_example)

center = (int(toy_example.shape[0] / 3), int(toy_example.shape[1] / 3))

neighbours = [(-1, -1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]

output_value = ""

for row in range(toy_example.shape[0]):
    for column in range(toy_example.shape[1]):
        #print(toy_example[row, column], end=" ")

        # Without padding
        wo_pad = (row > 0 and column > 0 and row < toy_example.shape[0] - 1 and column < toy_example.shape[0] - 1)

        if wo_pad:
            #print(row, column)
            for neighbor in neighbours:
                center_val = toy_example[row, column]
                coord_x = row + neighbor[0]
                coord_y = column + neighbor[1]

                print(toy_example[coord_x, coord_y])

                output_value += "1" if toy_example[coord_x, coord_y] >= center_val else "0"


            print(f"Out {row}/{column}: {output_value}")
            output_value = ""
"""
