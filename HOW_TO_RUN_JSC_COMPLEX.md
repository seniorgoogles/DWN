# How to Use jsc_complex.py - Complete Guide

## Overview

`jsc_complex.py` performs a complete pipeline:
1. **Phase 1**: Train initial dense model
2. **Phase 2**: Prune down to 6 inputs per neuron (structured pruning)
3. **Phase 3**: Fine-tune the pruned model
4. **Phase 4**: Train LUT-based model with fixed connections

---

## Quick Start

```bash
cd /home/fry/Documents/repositories/DWN/examples
python3 jsc_complex.py
```

That's it! The script runs the complete pipeline automatically.

---

## Configuration

### Key Parameters to Adjust

The main pruning configuration is at **lines 1011-1031**:

```python
model = iterative_pruning_and_retraining(
    model,
    optimizer_fn=make_optimizer,
    scheduler_fn=make_scheduler,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,

    # === KEY PARAMETERS TO ADJUST ===

    final_n_weights=6,          # Target: 6 inputs per neuron (for 6-input LUTs)

    pruning_steps=25,           # Number of pruning iterations
                                # More steps = slower, gentler pruning
                                # Fewer steps = faster, more aggressive

    epochs_per_step=6,          # Training epochs after each pruning step
                                # More epochs = better recovery, but slower

    pruning_steepness=2.0,      # Controls early pruning aggressiveness
                                # Lower (1.0-1.5) = gentler early pruning
                                # Higher (3.0-4.0) = aggressive early pruning

    use_backtracking=True,      # Enable adaptive backtracking
                                # Automatically halves step size if accuracy drops too much

    backtrack_threshold=0.05,   # Backtrack if accuracy drops >5%

    early_stop_threshold=0.15,  # Stop if accuracy drops >15% (catastrophic)

    extra_epochs_final=12,      # Extra epochs for final 20% of pruning steps

    adaptive_lr=True,           # Auto-adjust learning rate
    base_lr=1e-3,              # Base learning rate
                                # Auto-scaled: 2x aggressive, 0.5x fine-tuning

    batch_size=128,
)
```

### Recommended Configurations

#### 🎯 Balanced (Default)
```python
pruning_steps=25
epochs_per_step=6
pruning_steepness=2.0
backtrack_threshold=0.05
```
- **Time**: ~30-40 minutes
- **Quality**: Good balance

#### 🐢 Gentle (Maximum Accuracy)
```python
pruning_steps=40
epochs_per_step=10
pruning_steepness=1.0        # Very gentle early pruning
backtrack_threshold=0.03     # More sensitive backtracking
extra_epochs_final=20        # More training near target
```
- **Time**: ~60-90 minutes
- **Quality**: Best accuracy, slowest

#### 🚀 Fast (Quick Testing)
```python
pruning_steps=15
epochs_per_step=4
pruning_steepness=3.0        # Aggressive early pruning
backtrack_threshold=0.07     # Less sensitive backtracking
extra_epochs_final=5
```
- **Time**: ~15-20 minutes
- **Quality**: Lower accuracy, good for testing

---

## Pipeline Stages

### Stage 1: Dense Model Training
**Lines**: ~800-950
**What**: Trains initial unpruned model
**Duration**: ~5 minutes
**Output**: Baseline accuracy (~75%)

### Stage 2: Adaptive Pruning to 6 Inputs
**Lines**: 1011-1031
**What**: Gradually prunes to 6 inputs per neuron with backtracking
**Duration**: ~20-30 minutes (depends on `pruning_steps`)
**Output**: Pruned model with 6 connections per neuron

**What happens:**
- Starts at ~50% of inputs (e.g., 64 inputs for 128-dim input)
- Gradually reduces to 6 inputs
- If accuracy drops too much → backtracks and uses smaller steps
- Retrains after each pruning step

### Stage 3: Fine-Tuning (3 Phases)
**Lines**: 1036-1088
**What**: Intensive training to maximize accuracy at 6 inputs
**Duration**: ~10-15 minutes

1. **Phase 1** (30 epochs): LR=5e-4, gentle refinement
2. **Phase 2** (20 epochs): LR=1e-4, precise tuning
3. **Phase 3** (15 epochs): SGD momentum, escape local minima

**Expected improvement**: +2-5% accuracy

### Stage 4: Export Connections
**Lines**: 1091-1142
**What**: Extracts which 6 inputs each neuron uses
**Output**: `connections` list with mapping info

### Stage 5: Build & Train LUT Model
**Lines**: 1167-1277
**What**: Creates LUT model with fixed connections, trains truth tables
**Duration**: ~5-10 minutes (50 epochs)

**Key points:**
- Connections are FIXED (from pruned model)
- Only LUT truth table values are trained
- Each neuron → 64-entry LUT (2^6 entries)

### Stage 6: Save Results
**Lines**: 1279-1299
**Output files**:
- `dwn_lut_model_6inputs.pth` - Trained LUT model
- `pruned_connections_6inputs.pkl` - Connection mappings
- `pruned_models/jsc_pruned_final.pth` - Pruned linear model

---

## Expected Results

### Typical Accuracy Progression

```
Initial dense model:           ~75%
After pruning to 6 inputs:     ~68-70%
After fine-tuning:             ~71-73%
LUT model (truth tables):      ~70-72%
```

### Files Generated

```
📁 Current directory:
  ├── dwn_lut_model_6inputs.pth          # Final LUT model ✨
  ├── pruned_connections_6inputs.pkl      # Connection info
  └── best_model_*.pth                    # Checkpoints during training

📁 pruned_models/:
  └── jsc_pruned_final.pth               # Fine-tuned pruned model
```

---

## Troubleshooting

### Issue: Accuracy drops too much during pruning

**Solution 1**: Use gentler pruning
```python
pruning_steps=40              # More gradual
pruning_steepness=1.0         # Gentler early steps
backtrack_threshold=0.03      # More sensitive backtracking
```

**Solution 2**: More training per step
```python
epochs_per_step=10            # Train longer after each prune
extra_epochs_final=20         # More training near target
```

### Issue: Pruning takes too long

**Solution**: Use faster settings
```python
pruning_steps=15              # Fewer steps
epochs_per_step=4             # Less training per step
extra_epochs_final=5          # Less final training
```

### Issue: Sparsity shows 0%

**Fixed!** The recent updates (lines 544-562) now properly maintain sparsity tracking.

### Issue: Backtracking errors

**Fixed!** The checkpoint save/restore logic (lines 544-562, 628-635) now handles pruned model states correctly.

---

## Monitoring Progress

### During Pruning
Watch for:
```
PRUNING STEP 5/25
  Current weights/neuron: 32
  Target: 28

Accuracy after pruning: 0.7123
Accuracy after retraining: 0.7245
Recovery: +0.0122
```

**Good signs:**
- ✅ Gradual weight reduction
- ✅ Good recovery after retraining
- ✅ No backtracking (or only 1-2 backtracks)

**Warning signs:**
- ⚠️ Frequent backtracking (>5 times)
- ⚠️ Poor recovery (<0.005)
- ⚠️ Large accuracy drops (>10%)

### During Fine-Tuning
Watch for:
```
Fine-tuning phase 1: Lower LR (5e-4) for 30 epochs
Best accuracy this step: 0.7289
```

**Good signs:**
- ✅ Accuracy improving each phase
- ✅ +0.01 to +0.03 improvement total

### During LUT Training
Watch for:
```
Training DWN LUT truth tables...
Epoch 25/50: accuracy=0.7156
```

**Good signs:**
- ✅ Accuracy close to pruned model (within 1-2%)
- ✅ Steady improvement over epochs

---

## Advanced: Customizing for Different Targets

### For 4-input LUTs
```python
final_n_weights=4
pruning_steps=30              # More steps (harder target)
extra_epochs_final=20         # More training needed
```

### For 8-input LUTs
```python
final_n_weights=8
pruning_steps=20              # Fewer steps (easier target)
extra_epochs_final=8
```

---

## Quick Reference

| Parameter | Default | What it does |
|-----------|---------|--------------|
| `final_n_weights` | 6 | Target inputs per neuron |
| `pruning_steps` | 25 | Number of pruning iterations |
| `epochs_per_step` | 6 | Training after each prune |
| `pruning_steepness` | 2.0 | Early pruning aggressiveness (lower=gentler) |
| `backtrack_threshold` | 0.05 | Accuracy drop that triggers backtracking |
| `extra_epochs_final` | 12 | Extra training for final steps |
| `base_lr` | 1e-3 | Base learning rate |

---

## Summary

**To run with defaults:**
```bash
python3 jsc_complex.py
```

**To customize:**
Edit lines 1011-1031 in `jsc_complex.py`

**Total runtime:**
- Fast: ~15-20 min
- Default: ~30-40 min
- Gentle: ~60-90 min

**Key outputs:**
- `dwn_lut_model_6inputs.pth` - Your trained LUT model!
- `pruned_connections_6inputs.pkl` - Connection info
