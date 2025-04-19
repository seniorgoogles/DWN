import torch

def to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)

class Thermometer:
    """
    Base thermometer encoder. Splits [min, max] into num_bits + 1 equal intervals
    and produces thresholds at those boundaries.
    """
    def __init__(self, num_bits=1, feature_wise=True):
        assert num_bits > 0, "num_bits must be > 0"
        assert isinstance(feature_wise, bool), "feature_wise must be a bool"
        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.thresholds = None

    def get_thresholds(self, x):
        x = to_tensor(x)
        # compute min and max per-feature or global
        min_val = x.min(dim=0)[0] if self.feature_wise else x.min()
        max_val = x.max(dim=0)[0] if self.feature_wise else x.max()
        # compute evenly spaced cut points
        step = (max_val - min_val) / (self.num_bits + 1)
        cuts = torch.arange(1, self.num_bits + 1, dtype=torch.float32)
        # shape alignment
        min_val = min_val.unsqueeze(-1)
        step = step.unsqueeze(-1)
        # thresholds: min + cuts * step
        return min_val + cuts.unsqueeze(0) * step

    def fit(self, x):
        """Compute and store thresholds from data x."""
        x = to_tensor(x)
        self.thresholds = self.get_thresholds(x)
        return self

    def binarize(self, x):
        """Encode x into thermometer code of shape (*x.shape, num_bits)."""
        if self.thresholds is None:
            raise ValueError("Call fit() before binarize().")
        x = to_tensor(x)
        # add last dim for comparison
        x_exp = x.unsqueeze(-1)
        return (x_exp > self.thresholds).float()


class GaussianThermometer(Thermometer):
    """
    Thermometer encoder with thresholds at Gaussian quantiles of the data
    (assuming roughly normal distribution).
    """
    def get_thresholds(self, x):
        x = to_tensor(x)
        # standard normal quantiles for uniform probabilities
        probs = torch.arange(1, self.num_bits + 1, dtype=torch.float32) / (self.num_bits + 1)
        std_skews = torch.distributions.Normal(0, 1).icdf(probs)
        mean = x.mean(dim=0) if self.feature_wise else x.mean()
        std = x.std(dim=0, unbiased=False) if self.feature_wise else x.std(unbiased=False)
        # thresholds at mean + skew * std
        # shape: (features, num_bits) or (1, num_bits)
        thresholds = torch.stack([(mean + s * std) for s in std_skews], dim=-1)
        if not self.feature_wise:
            thresholds = thresholds.unsqueeze(0)
        return thresholds


class DistributiveThermometer(Thermometer):
    """
    Thermometer encoder with thresholds at empirical quantiles
    so that each bin has (approximately) equal data counts.
    """
    def get_thresholds(self, x):
        x = to_tensor(x)
        if self.feature_wise:
            # sort each feature column independently
            sorted_vals = torch.sort(x, dim=0)[0]  # shape (n_samples, n_features)
            n = sorted_vals.shape[0]
            idx = [int(n * i / (self.num_bits + 1)) for i in range(1, self.num_bits + 1)]
            # select rows at quantile positions: shape (num_bits, n_features)
            raw = sorted_vals[idx, :]
            # transpose to (n_features, num_bits)
            thresholds = raw.transpose(0, 1)
        else:
            # global quantiles
            flattened = torch.sort(x.flatten())[0]
            n = flattened.shape[0]
            idx = [int(n * i / (self.num_bits + 1)) for i in range(1, self.num_bits + 1)]
            raw = flattened[idx]  # shape (num_bits,)
            thresholds = raw.unsqueeze(0)  # shape (1, num_bits)
        return thresholds


if __name__ == "__main__":
    # example usage and shape test
    
    x = torch.randn(100, 5)
    
    '''
    # Print x values also min and max
    print(f"x:\n{x[:3]}")
    print(f"min: {x.min()}, max: {x.max()}")
    print(f"shape: {x.shape}")
    
    for cls in (Thermometer, GaussianThermometer, DistributiveThermometer):
        print(f"\nTesting {cls.__name__}")
        tm = cls(num_bits=4, feature_wise=True).fit(x)
        out = tm.binarize(x[:3])
        print(f"input shape: {x[:3].shape}, encoded shape: {out.shape}\n{out}")

    '''
    num_bits = 4
    # Test with min and max value
    x = torch.tensor([[-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0],[-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0, 6.0, 8.0]])
    x = torch.randn(100, 5)

    print(f"\nx:\n{x}")
    print(f"min: {x.min()}, max: {x.max()}")
    print(f"shape: {x.shape}")
    
    for cls in (Thermometer, GaussianThermometer, DistributiveThermometer):
        print(f"\nTesting {cls.__name__}")
        tm = cls(num_bits=num_bits, feature_wise=True).fit(x)
        out = tm.binarize(x[:3])
        print(f"input shape: {x[:3].shape}, encoded shape: {out.shape}\n{out}")