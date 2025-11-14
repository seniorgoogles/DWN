# Quick Test Configuration for jsc_complex.py

## Goal
Fast testing: Train → Prune to 6 weights → Train LUT model

**Expected time: ~8-12 minutes** (instead of 30-40 minutes)

---

## Changes to Make in `jsc_complex.py`

### 1. Reduce Initial Training (Line ~830-850)

**Find:**
```python
epochs=50, batch_size=128
```

**Change to:**
```python
epochs=15, batch_size=128  # Quick: only 15 epochs
```

---

### 2. Fast Pruning Configuration (Lines 1011-1031)

**Find:**
```python
model = iterative_pruning_and_retraining(
    model,
    optimizer_fn=make_optimizer,
    scheduler_fn=make_scheduler,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    target_sparsity=0.9,
    pruning_steps=25,           # ← CHANGE THIS
    epochs_per_step=6,          # ← CHANGE THIS
    batch_size=128,
    method='structured',
    final_n_weights=6,
    adaptive_lr=True,
    base_lr=1e-3,
    early_stop_threshold=0.15,
    extra_epochs_final=12,      # ← CHANGE THIS
    use_backtracking=True,
    backtrack_threshold=0.05,
    pruning_steepness=2.0
)
```

**Change to:**
```python
model = iterative_pruning_and_retraining(
    model,
    optimizer_fn=make_optimizer,
    scheduler_fn=make_scheduler,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    target_sparsity=0.9,
    pruning_steps=8,            # ← Only 8 steps (fast!)
    epochs_per_step=3,          # ← Only 3 epochs per step
    batch_size=128,
    method='structured',
    final_n_weights=6,
    adaptive_lr=True,
    base_lr=1e-3,
    early_stop_threshold=0.20,  # More lenient
    extra_epochs_final=3,       # ← Only 3 extra
    use_backtracking=True,
    backtrack_threshold=0.08,   # Less sensitive
    pruning_steepness=2.5       # More aggressive early pruning
)
```

---

### 3. Skip or Reduce Fine-Tuning (Lines 1046-1076)

**Option A: Comment out all 3 phases (fastest)**

Add `"""` before line 1046 and after line 1088 to comment out the entire fine-tuning section:

```python
"""  # ← ADD THIS
# Fine-tuning phase 1: Lower LR (5e-4) for 30 epochs
optimizer_ft1 = torch.optim.Adam(model.parameters(), lr=5e-4)
...
print(f"  Total improvement:  {final_acc_after_finetune - final_acc_before_finetune:+.4f}")
print("="*70)
"""  # ← ADD THIS
```

**Option B: Keep only Phase 1 with fewer epochs (recommended)**

```python
# Fine-tune with lower learning rate for longer
print("\nFine-tuning phase 1: Lower LR (5e-4) for 10 epochs")  # ← Changed from 30
optimizer_ft1 = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler_ft1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ft1, T_max=10)  # ← Changed from 30

best_acc_ft1 = train_and_evaluate(model, optimizer_ft1, scheduler_ft1,
                                   x_train, y_train, x_test, y_test,
                                   epochs=10, batch_size=128, save_best=True,  # ← Changed from 30
                                   phase_name='finetune_phase1')

# Comment out Phase 2 and Phase 3 (lines 1057-1076)
```

---

### 4. Reduce LUT Training (Lines 1263-1267)

**Find:**
```python
print("\nTraining for 50 epochs...")
best_dwn_lut_acc = train_and_evaluate(dwn_lut_model, dwn_lut_optimizer, dwn_lut_scheduler,
                                       x_train, y_train, x_test, y_test,
                                       epochs=50, batch_size=128, save_best=True,
                                       phase_name='lut_training')
```

**Change to:**
```python
print("\nTraining for 20 epochs...")  # ← Changed from 50
best_dwn_lut_acc = train_and_evaluate(dwn_lut_model, dwn_lut_optimizer, dwn_lut_scheduler,
                                       x_train, y_train, x_test, y_test,
                                       epochs=20, batch_size=128, save_best=True,  # ← Changed from 50
                                       phase_name='lut_training')
```

And update T_max in the scheduler (line 1261):
```python
dwn_lut_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(dwn_lut_optimizer, T_max=20)  # Changed from 50
```

---

## Summary of Changes

| Stage | Original | Quick Test | Time Saved |
|-------|----------|------------|------------|
| Initial training | 50 epochs | 15 epochs | ~3 min |
| Pruning steps | 25 steps × 6 epochs | 8 steps × 3 epochs | ~15 min |
| Fine-tuning | 3 phases (65 epochs) | 1 phase (10 epochs) or skip | ~10 min |
| LUT training | 50 epochs | 20 epochs | ~3 min |
| **Total** | **~35-40 min** | **~8-12 min** | **~25-30 min** |

---

## Expected Results (Quick Test)

```
Initial dense model:       ~72-74%  (vs ~75% with full training)
After pruning to 6:        ~65-68%  (vs ~68-70%)
After quick fine-tune:     ~67-70%  (vs ~71-73%)
LUT model (20 epochs):     ~66-69%  (vs ~70-72%)
```

**Quality loss:** ~2-4% accuracy compared to full training
**Speed gain:** ~4x faster

---

## Even Faster: Minimal Testing

For the absolute minimum test, use these values:

```python
# Initial training
epochs=10  # Instead of 15

# Pruning
pruning_steps=5              # Only 5 steps!
epochs_per_step=2            # Only 2 epochs
extra_epochs_final=2
use_backtracking=False       # Disable backtracking

# Fine-tuning
# Skip entirely (comment out)

# LUT training
epochs=10  # Instead of 20
```

**Expected time: ~5 minutes**
**Quality: ~60-65% accuracy (good enough to verify pipeline works)**

---

## How to Run

1. Make the changes above in `jsc_complex.py`
2. Run:
   ```bash
   python3 jsc_complex.py
   ```
3. You'll get the same outputs:
   - `dwn_lut_model_6inputs.pth`
   - `pruned_connections_6inputs.pkl`

---

## Restore Full Training

To go back to full quality training, just change the values back to the originals shown in this guide.
