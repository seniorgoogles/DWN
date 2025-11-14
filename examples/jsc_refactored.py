"""
JSC Classification with LUT-based Neural Networks
Refactored following SOLID principles and Python best practices

Pipeline:
1. Train dense model
2. Prune to 6 inputs per neuron
3. Fine-tune pruned model
4. Train LUT-based model with fixed connections
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum
import pickle

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_dwn as dwn
import openml
import numpy as np
from sklearn.model_selection import train_test_split


# =============================================================================
# Constants
# =============================================================================

DATASET_ID = 42468
RANDOM_SEED = 42
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Default hyperparameters
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 128
DEFAULT_TARGET_N_WEIGHTS = 6


# =============================================================================
# Configuration Classes (Single Responsibility + Type Safety)
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    epochs: int = 20
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LR
    device: str = DEFAULT_DEVICE
    save_best: bool = True
    phase_name: str = 'training'


@dataclass
class PruningConfig:
    """Configuration for iterative pruning."""
    target_n_weights: int = DEFAULT_TARGET_N_WEIGHTS
    pruning_steps: int = 25
    epochs_per_step: int = 6
    base_lr: float = DEFAULT_LR
    batch_size: int = DEFAULT_BATCH_SIZE

    # Backtracking
    use_backtracking: bool = True
    backtrack_threshold: float = 0.05
    max_backtracks: int = 10

    # Schedule control
    pruning_steepness: float = 2.0
    extra_epochs_final: int = 12
    early_stop_threshold: float = 0.15

    # Adaptive learning rate
    adaptive_lr: bool = True
    device: str = DEFAULT_DEVICE


@dataclass
class ModelArchitecture:
    """Model architecture specification."""
    input_size: int
    hidden_size: int
    output_size: int


@dataclass
class NeuronConnection:
    """Connection information for a single neuron."""
    neuron_idx: int
    active_input_indices: List[int]
    num_connections: int
    weights: Optional[List[float]] = None


@dataclass
class LayerConnection:
    """Connection information for a layer."""
    layer_idx: int
    in_features: int
    out_features: int
    connections_per_neuron: List[NeuronConnection] = field(default_factory=list)


# =============================================================================
# Data Management (Single Responsibility Principle)
# =============================================================================

class DatasetLoader:
    """Handles dataset loading and preprocessing."""

    def __init__(self, dataset_id: int = DATASET_ID, random_seed: int = RANDOM_SEED):
        self.dataset_id = dataset_id
        self.random_seed = random_seed

    def load_jsc_dataset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and preprocess JSC dataset.

        Returns:
            Tuple of (x_train, y_train, x_test, y_test) as tensors
        """
        print(f"Loading JSC dataset (ID: {self.dataset_id})...")

        dataset = openml.datasets.get_dataset(self.dataset_id)
        df_features, df_labels, _, _ = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute
        )

        # Convert to numpy
        features = df_features.values.astype(np.float32)
        label_names = list(df_labels.unique())
        labels = np.array(df_labels.map(lambda x: label_names.index(x)).values)

        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels,
            train_size=0.8,
            random_state=self.random_seed
        )

        # Convert to tensors
        x_train = torch.tensor(x_train, dtype=torch.float32)
        x_test = torch.tensor(x_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.int64)
        y_test = torch.tensor(y_test, dtype=torch.int64)

        print(f"  Train samples: {len(x_train)}")
        print(f"  Test samples: {len(x_test)}")
        print(f"  Features: {x_train.shape[1]}")
        print(f"  Classes: {len(label_names)}")

        return x_train, y_train, x_test, y_test

    def binarize_features(
        self,
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        n_bins: int = 200
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Binarize features using distributive thermometer encoding.

        Args:
            x_train: Training features
            x_test: Test features
            n_bins: Number of bins for thermometer encoding

        Returns:
            Tuple of (x_train_bin, x_test_bin)
        """
        print(f"Binarizing features (n_bins={n_bins})...")

        thermometer = dwn.DistributiveThermometer(n_bins).fit(x_train)
        x_train_bin = thermometer.binarize(x_train).flatten(start_dim=1)
        x_test_bin = thermometer.binarize(x_test).flatten(start_dim=1)

        print(f"  Original shape: {x_train.shape}")
        print(f"  Binarized shape: {x_train_bin.shape}")

        return x_train_bin, x_test_bin


# =============================================================================
# Model Evaluation (Single Responsibility Principle)
# =============================================================================

class ModelEvaluator:
    """Handles model evaluation."""

    def __init__(self, device: str = DEFAULT_DEVICE):
        self.device = device

    def evaluate(self, model: nn.Module, x_test: torch.Tensor, y_test: torch.Tensor) -> float:
        """Evaluate model accuracy.

        Args:
            model: Model to evaluate
            x_test: Test features
            y_test: Test labels

        Returns:
            Accuracy (0.0 to 1.0)
        """
        model.eval()
        with torch.no_grad():
            x_batch = x_test.to(self.device)
            y_batch = y_test.to(self.device)
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_batch).float().mean().item()
        return accuracy

    def calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate model sparsity.

        Args:
            model: Model to analyze

        Returns:
            Sparsity ratio (0.0 to 1.0)
        """
        total_params = 0
        zero_params = 0

        for module in model.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'weight_mask'):
                    total_params += module.weight_mask.numel()
                    zero_params += (module.weight_mask == 0).sum().item()
                else:
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()

        return zero_params / total_params if total_params > 0 else 0.0


# =============================================================================
# Model Training (Single Responsibility Principle)
# =============================================================================

class ModelTrainer:
    """Handles model training."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.evaluator = ModelEvaluator(config.device)
        self.best_model_state = None
        self.best_accuracy = 0.0

    def train_epoch(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        x_train: torch.Tensor,
        y_train: torch.Tensor
    ) -> float:
        """Train for one epoch.

        Returns:
            Average loss for the epoch
        """
        model.train()
        total_loss = 0.0
        num_batches = 0

        indices = torch.randperm(len(x_train))

        for i in range(0, len(x_train), self.config.batch_size):
            batch_idx = indices[i:i + self.config.batch_size]
            x_batch = x_train[batch_idx].to(self.config.device)
            y_batch = y_train[batch_idx].to(self.config.device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = nn.CrossEntropyLoss()(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor
    ) -> float:
        """Train model for configured number of epochs.

        Returns:
            Best accuracy achieved
        """
        print(f"\n{'='*70}")
        print(f"Training: {self.config.phase_name} ({self.config.epochs} epochs)")
        print(f"{'='*70}")

        self.best_accuracy = 0.0
        self.best_model_state = None

        for epoch in range(self.config.epochs):
            # Train
            avg_loss = self.train_epoch(model, optimizer, x_train, y_train)

            # Evaluate
            accuracy = self.evaluator.evaluate(model, x_test, y_test)

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

            # Save best
            if self.config.save_best and accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Print progress
            if (epoch + 1) % max(1, self.config.epochs // 10) == 0:
                print(f"  Epoch {epoch+1}/{self.config.epochs}: loss={avg_loss:.4f}, acc={accuracy:.4f}")

        # Restore best weights
        if self.config.save_best and self.best_model_state is not None:
            model.load_state_dict({k: v.to(self.config.device) for k, v in self.best_model_state.items()})
            print(f"\n✅ Best accuracy: {self.best_accuracy:.4f}")

        return self.best_accuracy


# =============================================================================
# Model Pruning (Single Responsibility + Open/Closed Principle)
# =============================================================================

class BasePruner:
    """Abstract base pruner (Open/Closed Principle)."""

    def __init__(self, layer_indices: Optional[List[int]] = None):
        self.layer_indices = layer_indices

    def prune(self, model: nn.Module) -> nn.Module:
        """Prune the model. Must be implemented by subclasses."""
        raise NotImplementedError


class StructuredPruner(BasePruner):
    """Prunes to exactly N weights per neuron by magnitude."""

    def __init__(self, n_weights: int, layer_indices: Optional[List[int]] = None):
        super().__init__(layer_indices)
        self.n_weights = n_weights

    def prune(self, model: nn.Module) -> nn.Module:
        """Prune model to n_weights per neuron.

        Args:
            model: Model to prune

        Returns:
            Pruned model
        """
        print(f"\nStructured pruning to {self.n_weights} weights per neuron...")

        linear_layers = [(name, module) for name, module in model.named_modules()
                         if isinstance(module, nn.Linear)]

        for idx, (name, module) in enumerate(linear_layers):
            if self.layer_indices is not None and idx not in self.layer_indices:
                print(f"  Skipping layer {idx}: {name}")
                continue

            # Get weights (remove existing pruning if present)
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')

            weight = module.weight.data
            out_features, in_features = weight.shape

            if in_features <= self.n_weights:
                print(f"  Skipping layer {idx} ({name}): only {in_features} inputs")
                continue

            print(f"  Pruning layer {idx} ({name}): {weight.shape}")

            # Create mask: keep top-N by magnitude per neuron
            mask = torch.zeros_like(weight)
            for neuron_idx in range(out_features):
                neuron_weights_abs = weight[neuron_idx].abs()
                _, top_indices = torch.topk(neuron_weights_abs, k=self.n_weights, largest=True)
                mask[neuron_idx, top_indices] = 1.0

            # Apply pruning
            prune.custom_from_mask(module, name='weight', mask=mask)

            # Report
            sparsity = (mask == 0).sum().item() / mask.numel()
            print(f"    Sparsity: {sparsity*100:.2f}%")

        return model


# =============================================================================
# Iterative Pruning Manager (Facade Pattern)
# =============================================================================

class IterativePruningManager:
    """Manages iterative pruning with backtracking."""

    def __init__(
        self,
        config: PruningConfig,
        trainer_factory: Callable[[int, float], Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]]
    ):
        self.config = config
        self.trainer_factory = trainer_factory
        self.evaluator = ModelEvaluator(config.device)

    def _calculate_pruning_schedule(self, start_weights: int) -> List[int]:
        """Calculate exponential decay schedule for pruning.

        Args:
            start_weights: Starting number of weights per neuron

        Returns:
            List of weight counts for each step
        """
        t_values = np.linspace(0, 1, self.config.pruning_steps)
        schedule = []

        for t in t_values:
            w = self.config.target_n_weights + \
                (start_weights - self.config.target_n_weights) * \
                np.exp(-self.config.pruning_steepness * t)
            schedule.append(int(w))

        # Ensure monotonic decrease
        for i in range(1, len(schedule)):
            if schedule[i] >= schedule[i-1]:
                schedule[i] = max(self.config.target_n_weights, schedule[i-1] - 1)

        schedule[-1] = self.config.target_n_weights

        return schedule

    def prune_and_retrain(
        self,
        model: nn.Module,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_test: torch.Tensor,
        y_test: torch.Tensor
    ) -> nn.Module:
        """Iteratively prune and retrain model.

        Args:
            model: Initial model
            x_train, y_train: Training data
            x_test, y_test: Test data

        Returns:
            Pruned and retrained model
        """
        print("\n" + "="*70)
        print("ITERATIVE PRUNING AND RETRAINING")
        print(f"  Target: {self.config.target_n_weights} weights/neuron")
        print(f"  Steps: {self.config.pruning_steps}")
        print("="*70)

        # Get starting point
        first_layer = next(m for m in model.modules() if isinstance(m, nn.Linear))
        start_weights = first_layer.weight.shape[1] // 2

        # Calculate schedule
        if self.config.use_backtracking:
            current_n_weights = float(start_weights)
            step_size = (start_weights - self.config.target_n_weights) / self.config.pruning_steps
        else:
            weights_schedule = self._calculate_pruning_schedule(start_weights)
            print(f"  Schedule: {weights_schedule[:5]}...{weights_schedule[-1]}")

        # Training loop
        step = 0
        best_accuracies = []
        backtrack_count = 0
        target_reached = False

        while step < self.config.pruning_steps and not target_reached:
            print(f"\n{'='*70}")
            print(f"PRUNING STEP {step + 1}/{self.config.pruning_steps}")
            if self.config.use_backtracking:
                print(f"  Current weights/neuron: {current_n_weights:.0f}")
                print(f"  Step size: {step_size:.2f}")
            print(f"{'='*70}")

            # Save checkpoint
            checkpoint_state = None
            checkpoint_accuracy = None
            if self.config.use_backtracking:
                # Remove pruning to save clean checkpoint
                for module in model.modules():
                    if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
                        prune.remove(module, 'weight')

                import copy
                checkpoint_state = copy.deepcopy(model.state_dict())
                checkpoint_accuracy = self.evaluator.evaluate(model, x_test, y_test)

            # Prune
            if self.config.use_backtracking:
                target_n_weights = max(self.config.target_n_weights,
                                      int(current_n_weights - step_size))
            else:
                target_n_weights = weights_schedule[step]

            print(f"Target weights: {target_n_weights}")
            pruner = StructuredPruner(n_weights=target_n_weights, layer_indices=[0, 1])
            model = pruner.prune(model)

            # Evaluate after pruning
            acc_after_prune = self.evaluator.evaluate(model, x_test, y_test)
            print(f"Accuracy after pruning: {acc_after_prune:.4f}")

            # Retrain
            epochs = self.config.epochs_per_step
            is_final_phase = step >= int(self.config.pruning_steps * 0.8)
            if is_final_phase:
                epochs += self.config.extra_epochs_final

            lr = self._calculate_adaptive_lr(step, target_n_weights,
                                            current_n_weights if self.config.use_backtracking else weights_schedule[step-1] if step > 0 else start_weights)

            optimizer, scheduler = self.trainer_factory(epochs, lr)

            trainer = ModelTrainer(TrainingConfig(
                epochs=epochs,
                batch_size=self.config.batch_size,
                learning_rate=lr,
                device=self.config.device,
                save_best=True,
                phase_name=f'pruning_step_{step+1}'
            ))

            best_acc = trainer.train(model, optimizer, scheduler, x_train, y_train, x_test, y_test)

            # Check backtracking
            should_backtrack = False
            if self.config.use_backtracking and len(best_accuracies) > 0:
                best_so_far = max(best_accuracies)
                drop_ratio = (best_so_far - best_acc) / best_so_far if best_so_far > 0 else 0

                if drop_ratio > self.config.backtrack_threshold and backtrack_count < self.config.max_backtracks:
                    should_backtrack = True
                    print(f"\n🔄 BACKTRACKING: Accuracy dropped by {drop_ratio*100:.1f}%")

                    # Remove pruning
                    for module in model.modules():
                        if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
                            prune.remove(module, 'weight')

                    # Restore
                    model.load_state_dict(checkpoint_state)
                    step_size /= 2
                    backtrack_count += 1
                    print(f"  New step size: {step_size:.2f}")

            if not should_backtrack:
                best_accuracies.append(best_acc)

                if self.config.use_backtracking:
                    current_n_weights = target_n_weights
                    if current_n_weights <= self.config.target_n_weights:
                        target_reached = True
                        print(f"🎯 TARGET REACHED: {self.config.target_n_weights} weights per neuron!")

                step += 1

        return model

    def _calculate_adaptive_lr(self, step: int, current_n: int, prev_n: int) -> float:
        """Calculate adaptive learning rate based on pruning severity."""
        if not self.config.adaptive_lr:
            return self.config.base_lr

        if prev_n > 0:
            pruning_ratio = (prev_n - current_n) / prev_n
        else:
            pruning_ratio = 0.0

        if pruning_ratio > 0.1:
            return self.config.base_lr * 2.0
        elif pruning_ratio > 0.05:
            return self.config.base_lr * 1.5
        else:
            return self.config.base_lr * 0.5


# =============================================================================
# Connection Export (Single Responsibility)
# =============================================================================

class ConnectionExporter:
    """Exports pruned connections from model."""

    def export_connections(self, model: nn.Module) -> List[LayerConnection]:
        """Export connection information from pruned model.

        Args:
            model: Pruned model

        Returns:
            List of LayerConnection objects
        """
        layer_connections = []

        for layer_idx, module in enumerate(model.modules()):
            if not isinstance(module, nn.Linear):
                continue

            # Get weights and mask
            if hasattr(module, 'weight_mask'):
                mask = module.weight_mask.cpu().numpy()
                weights = module.weight_orig.detach().cpu().numpy()
            else:
                weights = module.weight.detach().cpu().numpy()
                mask = (weights != 0).astype(float)

            # Extract connections per neuron
            connections_per_neuron = []
            for neuron_idx in range(module.weight.shape[0]):
                active_indices = np.where(mask[neuron_idx] > 0)[0].tolist()
                active_weights = weights[neuron_idx][active_indices].tolist()

                connections_per_neuron.append(NeuronConnection(
                    neuron_idx=neuron_idx,
                    active_input_indices=active_indices,
                    num_connections=len(active_indices),
                    weights=active_weights
                ))

            layer_connections.append(LayerConnection(
                layer_idx=layer_idx,
                in_features=module.weight.shape[1],
                out_features=module.weight.shape[0],
                connections_per_neuron=connections_per_neuron
            ))

        return layer_connections

    def validate_connections(
        self,
        connections: List[LayerConnection],
        target_n: int = DEFAULT_TARGET_N_WEIGHTS
    ) -> bool:
        """Validate that all neurons have ≤target_n connections.

        Returns:
            True if valid, False if violations found
        """
        print(f"\n🔍 Validating connections (max {target_n} weights per neuron)...")

        has_violations = False
        for layer_conn in connections:
            violations = [
                (nc.neuron_idx, nc.num_connections)
                for nc in layer_conn.connections_per_neuron
                if nc.num_connections > target_n
            ]

            if violations:
                has_violations = True
                print(f"\n⚠️  Layer {layer_conn.layer_idx}: {len(violations)} neurons with >{target_n} connections")
                for neuron_idx, num_conns in violations[:3]:
                    print(f"    Neuron {neuron_idx}: {num_conns} connections")
            else:
                num_conns = [nc.num_connections for nc in layer_conn.connections_per_neuron]
                if all(n == target_n for n in num_conns):
                    print(f"  ✅ Layer {layer_conn.layer_idx}: All neurons have exactly {target_n} connections")
                else:
                    print(f"  ⚠️  Layer {layer_conn.layer_idx}: {min(num_conns)}-{max(num_conns)} connections")

        if has_violations:
            print("\n⚠️  WARNING: Some neurons exceed target connections!")
        else:
            print("\n✅ VALIDATION PASSED")

        return not has_violations


# =============================================================================
# LUT Model Builder (Single Responsibility)
# =============================================================================

class LUTModelBuilder:
    """Builds LUT-based model from pruned connections."""

    def __init__(self, target_n: int = DEFAULT_TARGET_N_WEIGHTS):
        self.target_n = target_n

    def create_mapping(self, layer_conn: LayerConnection) -> torch.Tensor:
        """Create fixed mapping tensor for LUTLayer.

        Args:
            layer_conn: Layer connection information

        Returns:
            Mapping tensor [out_features, target_n]
        """
        mapping = torch.zeros(layer_conn.out_features, self.target_n, dtype=torch.int32)

        warnings = {'too_few': [], 'too_many': []}

        for neuron_conn in layer_conn.connections_per_neuron:
            neuron_idx = neuron_conn.neuron_idx
            active_indices = neuron_conn.active_input_indices
            num_conns = neuron_conn.num_connections

            if num_conns == self.target_n:
                # Perfect
                for i, idx in enumerate(active_indices):
                    mapping[neuron_idx, i] = idx

            elif num_conns < self.target_n:
                # Too few - pad
                warnings['too_few'].append((neuron_idx, num_conns))
                for i in range(self.target_n):
                    if i < num_conns:
                        mapping[neuron_idx, i] = active_indices[i]
                    else:
                        mapping[neuron_idx, i] = active_indices[0]

            else:  # num_conns > self.target_n
                # Too many - trim by magnitude
                warnings['too_many'].append((neuron_idx, num_conns))

                weights_list = neuron_conn.weights
                if weights_list and len(weights_list) == num_conns:
                    # Sort by magnitude
                    weight_magnitude_pairs = [
                        (abs(w), idx) for w, idx in zip(weights_list, active_indices)
                    ]
                    weight_magnitude_pairs.sort(reverse=True, key=lambda x: x[0])
                    sorted_indices = [idx for _, idx in weight_magnitude_pairs[:self.target_n]]

                    for i in range(self.target_n):
                        mapping[neuron_idx, i] = sorted_indices[i]
                else:
                    # Fallback
                    for i in range(self.target_n):
                        mapping[neuron_idx, i] = active_indices[i]

        # Report warnings
        if warnings['too_few']:
            print(f"  ⚠️  {len(warnings['too_few'])} neurons padded to {self.target_n}")

        if warnings['too_many']:
            print(f"  ⚠️  {len(warnings['too_many'])} neurons trimmed to top {self.target_n} by magnitude")

        return mapping

    def build_model(self, connections: List[LayerConnection]) -> nn.Module:
        """Build LUT model from connections.

        Args:
            connections: List of layer connections

        Returns:
            LUT-based model
        """
        print("\n🔨 Building LUT model with fixed connections...")

        layers = []
        for layer_conn in connections:
            mapping = self.create_mapping(layer_conn)

            # Verify shape
            assert mapping.shape[1] == self.target_n, \
                f"Mapping should have {self.target_n} inputs, got {mapping.shape[1]}"

            lut_layer = dwn.LUTLayer(
                layer_conn.in_features,
                layer_conn.out_features,
                n=self.target_n,
                mapping=mapping
            )

            layers.append(lut_layer)
            print(f"  Layer {layer_conn.layer_idx}: {layer_conn.out_features} LUTs × {2**self.target_n} entries")

        print(f"  ✅ All neurons configured with exactly {self.target_n} inputs")

        return nn.Sequential(*layers)


# =============================================================================
# Pipeline Orchestrator (Facade Pattern)
# =============================================================================

class LUTPruningPipeline:
    """Main pipeline orchestrator."""

    def __init__(
        self,
        architecture: ModelArchitecture,
        pruning_config: PruningConfig,
        output_dir: Path = Path(".")
    ):
        self.architecture = architecture
        self.pruning_config = pruning_config
        self.output_dir = output_dir

        self.dataset_loader = DatasetLoader()
        self.connection_exporter = ConnectionExporter()
        self.lut_builder = LUTModelBuilder(pruning_config.target_n_weights)

    def create_dense_model(self) -> nn.Module:
        """Create initial dense model."""
        return nn.Sequential(
            nn.Linear(self.architecture.input_size, self.architecture.hidden_size),
            nn.ReLU(),
            nn.Linear(self.architecture.hidden_size, self.architecture.output_size)
        ).to(self.pruning_config.device)

    def run(self) -> Dict:
        """Run complete pipeline.

        Returns:
            Dictionary with results and models
        """
        print("="*70)
        print("LUT PRUNING PIPELINE")
        print("="*70)

        # 1. Load data
        x_train, y_train, x_test, y_test = self.dataset_loader.load_jsc_dataset()
        x_train, x_test = self.dataset_loader.binarize_features(x_train, x_test)

        # 2. Train dense model
        print("\n" + "="*70)
        print("STAGE 1: TRAINING DENSE MODEL")
        print("="*70)

        model = self.create_dense_model()

        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        trainer = ModelTrainer(TrainingConfig(
            epochs=50,
            phase_name='initial_training',
            device=self.pruning_config.device
        ))

        initial_acc = trainer.train(model, optimizer, scheduler,
                                   x_train, y_train, x_test, y_test)

        # 3. Prune model
        print("\n" + "="*70)
        print("STAGE 2: PRUNING TO 6 WEIGHTS")
        print("="*70)

        def trainer_factory(epochs: int, lr: float):
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            return opt, sched

        pruning_manager = IterativePruningManager(self.pruning_config, trainer_factory)
        model = pruning_manager.prune_and_retrain(model, x_train, y_train, x_test, y_test)

        evaluator = ModelEvaluator(self.pruning_config.device)
        pruned_acc = evaluator.evaluate(model, x_test, y_test)

        # 4. Export connections
        print("\n" + "="*70)
        print("STAGE 3: EXPORTING CONNECTIONS")
        print("="*70)

        connections = self.connection_exporter.export_connections(model)
        self.connection_exporter.validate_connections(connections)

        # 5. Build LUT model
        print("\n" + "="*70)
        print("STAGE 4: BUILDING LUT MODEL")
        print("="*70)

        lut_model = self.lut_builder.build_model(connections).to(self.pruning_config.device)

        # 6. Train LUT model
        print("\n" + "="*70)
        print("STAGE 5: TRAINING LUT MODEL")
        print("="*70)

        lut_optimizer = torch.optim.Adam(lut_model.parameters(), lr=1e-2)
        lut_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lut_optimizer, T_max=50)

        lut_trainer = ModelTrainer(TrainingConfig(
            epochs=50,
            learning_rate=1e-2,
            phase_name='lut_training',
            device=self.pruning_config.device
        ))

        lut_acc = lut_trainer.train(lut_model, lut_optimizer, lut_scheduler,
                                    x_train, y_train, x_test, y_test)

        # 7. Save results
        self._save_results(model, lut_model, connections, {
            'initial_acc': initial_acc,
            'pruned_acc': pruned_acc,
            'lut_acc': lut_acc
        })

        # Summary
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE")
        print("="*70)
        print(f"Initial accuracy:  {initial_acc:.4f}")
        print(f"Pruned accuracy:   {pruned_acc:.4f}")
        print(f"LUT accuracy:      {lut_acc:.4f}")
        print("="*70)

        return {
            'model': model,
            'lut_model': lut_model,
            'connections': connections,
            'accuracies': {
                'initial': initial_acc,
                'pruned': pruned_acc,
                'lut': lut_acc
            }
        }

    def _save_results(
        self,
        model: nn.Module,
        lut_model: nn.Module,
        connections: List[LayerConnection],
        accuracies: Dict[str, float]
    ):
        """Save models and results."""
        print("\n💾 Saving results...")

        # Save pruned model
        pruned_path = self.output_dir / "pruned_models"
        pruned_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), pruned_path / "jsc_pruned_final.pth")

        # Save LUT model
        torch.save({
            'model_state_dict': lut_model.state_dict(),
            'accuracy': accuracies['lut']
        }, self.output_dir / "dwn_lut_model_6inputs.pth")

        # Save connections
        with open(self.output_dir / "pruned_connections_6inputs.pkl", 'wb') as f:
            pickle.dump({
                'connections': connections,
                'architecture': {
                    'input_size': self.architecture.input_size,
                    'hidden_size': self.architecture.hidden_size,
                    'output_size': self.architecture.output_size,
                    'n_inputs_per_lut': self.pruning_config.target_n_weights
                },
                'accuracy': accuracies
            }, f)

        print("  ✓ pruned_models/jsc_pruned_final.pth")
        print("  ✓ dwn_lut_model_6inputs.pth")
        print("  ✓ pruned_connections_6inputs.pkl")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point."""
    # Configuration
    architecture = ModelArchitecture(
        input_size=3200,  # Will be set after binarization
        hidden_size=10,
        output_size=5
    )

    pruning_config = PruningConfig(
        target_n_weights=6,
        pruning_steps=25,
        epochs_per_step=6,
        pruning_steepness=2.0,
        use_backtracking=True,
        backtrack_threshold=0.05
    )

    # Run pipeline
    pipeline = LUTPruningPipeline(architecture, pruning_config)
    results = pipeline.run()


if __name__ == '__main__':
    main()
