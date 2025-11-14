"""
Visualize how pruning_steepness affects the pruning schedule
"""
import numpy as np

def compute_schedule(pruning_steps, start_weights, final_n_weights, steepness):
    """Compute pruning schedule for given steepness"""
    t_values = np.linspace(0, 1, pruning_steps)
    weights_schedule = []

    for t in t_values:
        w = final_n_weights + (start_weights - final_n_weights) * np.exp(-steepness * t)
        weights_schedule.append(int(w))

    # Ensure monotonic decrease
    for i in range(1, len(weights_schedule)):
        if weights_schedule[i] >= weights_schedule[i-1]:
            weights_schedule[i] = max(final_n_weights, weights_schedule[i-1] - 1)

    weights_schedule[-1] = final_n_weights

    return weights_schedule


# Example: pruning from 128 inputs to 6 inputs in 10 steps
pruning_steps = 10
start_weights = 64  # 50% of 128 inputs
final_n_weights = 6

print("=" * 80)
print("PRUNING SCHEDULE COMPARISON")
print("=" * 80)
print(f"Starting weights: {start_weights}, Target: {final_n_weights}, Steps: {pruning_steps}")
print()

steepness_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

print(f"{'Step':<6}", end="")
for s in steepness_values:
    print(f"steep={s:<4.1f}", end="  ")
print()
print("-" * 80)

# Compute schedules for all steepness values
schedules = {s: compute_schedule(pruning_steps, start_weights, final_n_weights, s)
             for s in steepness_values}

# Print side by side
for step in range(pruning_steps):
    print(f"{step+1:<6}", end="")
    for s in steepness_values:
        print(f"{schedules[s][step]:<10}", end="  ")
    print()

print()
print("=" * 80)
print("STEP SIZE (weights removed per step)")
print("=" * 80)

print(f"{'Step':<6}", end="")
for s in steepness_values:
    print(f"steep={s:<4.1f}", end="  ")
print()
print("-" * 80)

for step in range(pruning_steps):
    print(f"{step+1:<6}", end="")
    for s in steepness_values:
        if step == 0:
            removed = start_weights - schedules[s][step]
        else:
            removed = schedules[s][step-1] - schedules[s][step]
        print(f"-{removed:<9}", end="  ")
    print()

print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)
print("Lower steepness (1.0-1.5):")
print("  ✓ Gentle early pruning (small steps initially)")
print("  ✓ More aggressive later (bigger steps near the end)")
print("  ✓ Good when early phases are less crucial")
print()
print("Higher steepness (3.0-4.0):")
print("  ✓ Aggressive early pruning (big steps initially)")
print("  ✓ Gentle later (small steps near the end)")
print("  ✓ Good for quickly removing obviously redundant connections")
print()
print(f"Default (2.0): Balanced approach")
