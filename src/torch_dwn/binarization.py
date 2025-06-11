import torch
from brevitas.nn import QuantIdentity
import numpy as np

class Thermometer:
    def __init__(self, num_bits=1, feature_wise=True):
        
        assert num_bits > 0
        assert type(feature_wise) is bool

        self.num_bits = int(num_bits)
        self.feature_wise = feature_wise
        self.thresholds = None

    def get_thresholds(self, x):
        min_value = x.min(dim=0)[0] if self.feature_wise else x.min()
        max_value = x.max(dim=0)[0] if self.feature_wise else x.max()
        return min_value.unsqueeze(-1) + torch.arange(1, self.num_bits+1).unsqueeze(0) * ((max_value - min_value) / (self.num_bits + 1)).unsqueeze(-1)

    def fit(self, x):
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        self.thresholds = self.get_thresholds(x)
        return self
    
    def binarize(self, x, bit_width=-1):
        quant_identity = QuantIdentity(return_quant_tensor=True, bit_width=8)

        if self.thresholds is None:
            raise 'need to fit before calling apply'
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        
        if bit_width > 0:
            x = quant_identity(x).tensor

        x = x.unsqueeze(-1)
        return (x > self.thresholds).float()

class GaussianThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        std_skews = torch.distributions.Normal(0, 1).icdf(torch.arange(1, self.num_bits+1)/(self.num_bits+1))
        mean = x.mean(dim=0) if self.feature_wise else x.mean()
        std = x.std(dim=0) if self.feature_wise else x.std() 
        thresholds = torch.stack([std_skew * std + mean for std_skew in std_skews], dim=-1)
        return thresholds
    
class DistributiveThermometer(Thermometer):
    def __init__(self, num_bits=1, feature_wise=True):
        super().__init__(num_bits, feature_wise)

    def get_thresholds(self, x):
        data = torch.sort(x.flatten())[0] if not self.feature_wise else torch.sort(x, dim=0)[0]
        indicies = torch.tensor([int(data.shape[0]*i/(self.num_bits+1)) for i in range(1, self.num_bits+1)])
        thresholds = data[indicies]
        return torch.permute(thresholds, (*list(range(1, thresholds.ndim)), 0))

class AdaptiveThermometer:
    def __init__(self, num_bits=1, method='density', feature_wise=True):
        self.num_bits = int(num_bits)
        self.method = method
        self.feature_wise = feature_wise
        self.thresholds = None

    def fit(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if self.feature_wise:
            # Stelle sicher, dass wir für jede Feature exakt num_bits Thresholds haben
            thresholds_list = []
            for i in range(x.shape[1]):
                feature_data = x[:, i]
                if self.method == 'density':
                    thresholds = self._density_based_thresholds(feature_data)
                elif self.method == 'variance':
                    thresholds = self._variance_based_thresholds(feature_data)
                else:
                    raise ValueError(f"Unknown method: {self.method}")
                
                # Stelle sicher, dass wir exakt num_bits Thresholds haben
                if len(thresholds) < self.num_bits:
                    # Fülle mit uniform verteilten Thresholds auf
                    additional_needed = self.num_bits - len(thresholds)
                    min_val, max_val = feature_data.min(), feature_data.max()
                    additional_thresholds = torch.linspace(min_val, max_val, additional_needed + 2)[1:-1]
                    thresholds = torch.cat([thresholds, additional_thresholds])
                elif len(thresholds) > self.num_bits:
                    # Reduziere auf num_bits
                    indices = torch.linspace(0, len(thresholds)-1, self.num_bits).long()
                    thresholds = thresholds[indices]
                
                thresholds_list.append(thresholds)
            
            # Shape: (num_features, num_bits)
            self.thresholds = torch.stack(thresholds_list)
        else:
            if self.method == 'density':
                self.thresholds = self._density_based_thresholds(x.flatten())
            elif self.method == 'variance':
                self.thresholds = self._variance_based_thresholds(x.flatten())
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Stelle sicher, dass wir exakt num_bits Thresholds haben für non-feature-wise
            if len(self.thresholds) < self.num_bits:
                additional_needed = self.num_bits - len(self.thresholds)
                min_val, max_val = x.min(), x.max()
                additional_thresholds = torch.linspace(min_val, max_val, additional_needed + 2)[1:-1]
                self.thresholds = torch.cat([self.thresholds, additional_thresholds])
            elif len(self.thresholds) > self.num_bits:
                indices = torch.linspace(0, len(self.thresholds)-1, self.num_bits).long()
                self.thresholds = self.thresholds[indices]

        return self

    def _density_based_thresholds(self, data, n_bins=50):
        data_np = data.numpy()
        hist, bin_edges = np.histogram(data_np, bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        density = hist / (hist.sum() + 1e-10)
        density = density + 1e-10
        density /= density.sum()
        
        # Verwende weniger Thresholds als num_bits, um Raum für das Auffüllen zu lassen
        num_to_select = min(self.num_bits, len(bin_centers))
        threshold_indices = np.random.choice(len(bin_centers), size=num_to_select,
                                             p=density, replace=False)
        thresholds = torch.tensor(np.sort(bin_centers[threshold_indices]), dtype=torch.float32)
        thresholds = torch.clamp(thresholds, data.min(), data.max())
        return torch.unique(thresholds)

    def _variance_based_thresholds(self, data, window_size=None):
        data_sorted = torch.sort(data)[0]
        if window_size is None:
            window_size = max(10, len(data_sorted) // 20)

        variances = []
        positions = []
        step = max(1, window_size // 4)
        for i in range(0, len(data_sorted) - window_size, step):
            window = data_sorted[i:i + window_size]
            var = torch.var(window).item()
            variances.append(var)
            positions.append(i + window_size // 2)

        if not variances or sum(variances) == 0:
            return self._uniform_fallback(data_sorted)

        var_probs = np.array(variances) / sum(variances)
        num_to_select = min(self.num_bits, len(positions))
        threshold_positions = np.random.choice(positions, size=num_to_select,
                                               p=var_probs, replace=False)
        thresholds = data_sorted[np.sort(threshold_positions)]
        return torch.unique(thresholds)

    def _uniform_fallback(self, data):
        return torch.linspace(data.min(), data.max(), self.num_bits + 2)[1:-1]

    def binarize(self, x):
        if self.thresholds is None:
            raise ValueError('need to fit before calling binarize')
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if self.feature_wise:
            # x hat Shape: (batch_size, num_features)
            # thresholds hat Shape: (num_features, num_bits)
            # Wir wollen Output Shape: (batch_size, num_features, num_bits)
            
            x_expanded = x.unsqueeze(-1)  # Shape: (batch_size, num_features, 1)
            thresholds_expanded = self.thresholds.unsqueeze(0)  # Shape: (1, num_features, num_bits)
            
            # Broadcasting: (batch_size, num_features, 1) > (1, num_features, num_bits)
            # Ergebnis: (batch_size, num_features, num_bits)
            return (x_expanded > thresholds_expanded).float()
        else:
            # x hat Shape: (batch_size, num_features)
            # thresholds hat Shape: (num_bits,)
            # Wir wollen Output Shape: (batch_size, num_features, num_bits)
            
            x_expanded = x.unsqueeze(-1)  # Shape: (batch_size, num_features, 1)
            # Broadcasting: (batch_size, num_features, 1) > (num_bits,)
            # Ergebnis: (batch_size, num_features, num_bits)
            return (x_expanded > self.thresholds).float()