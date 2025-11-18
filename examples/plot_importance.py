"""
Visualize bit importance from trained model
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, 'src')

def plot_importance_analysis(encoder, lut_layer, save_path='importance_analysis.png'):
    """
    Create comprehensive visualization of bit importance

    Args:
        encoder: EncoderLayer instance (after set_importance_from_lut)
        lut_layer: LUTLayer instance
        save_path: Where to save the plot
    """
    from torch_dwn.encoder_layer import compute_bit_importance_from_lut

    # Get importance
    if not hasattr(encoder, 'importance_weights') or encoder.importance_weights is None:
        print("Setting importance from LUT layer...")
        encoder.set_importance_from_lut(lut_layer, method='weighted')

    importance = encoder.importance_weights.cpu().detach().numpy()
    num_features, num_thresholds = importance.shape

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Heatmap of importance per feature
    ax1 = fig.add_subplot(gs[0, :])
    im1 = ax1.imshow(importance, aspect='auto', cmap='hot', interpolation='nearest')
    ax1.set_xlabel('Threshold Index', fontsize=12)
    ax1.set_ylabel('Feature Index', fontsize=12)
    ax1.set_title('Bit Importance Heatmap (Brighter = More Important)', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Importance Weight')

    # 2. Histogram of importance values
    ax2 = fig.add_subplot(gs[1, 0])
    importance_flat = importance.flatten()
    ax2.hist(importance_flat, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label=f'Zero ({(importance_flat==0).sum()} bits)')
    ax2.set_xlabel('Importance Weight', fontsize=11)
    ax2.set_ylabel('Number of Bits', fontsize=11)
    ax2.set_title('Distribution of Importance Weights', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Average importance per feature (bar plot)
    ax3 = fig.add_subplot(gs[1, 1])
    feature_importance = importance.mean(axis=1)
    bars = ax3.bar(range(num_features), feature_importance, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Feature Index', fontsize=11)
    ax3.set_ylabel('Average Importance', fontsize=11)
    ax3.set_title('Average Importance per Feature', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # Color bars by importance
    colors = plt.cm.hot(feature_importance / feature_importance.max() if feature_importance.max() > 0 else feature_importance)
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 4. Coverage: % of thresholds used per feature
    ax4 = fig.add_subplot(gs[1, 2])
    coverage_per_feature = (importance > 0).mean(axis=1) * 100
    bars = ax4.barh(range(num_features), coverage_per_feature, edgecolor='black', alpha=0.7)
    ax4.set_ylabel('Feature Index', fontsize=11)
    ax4.set_xlabel('Coverage (%)', fontsize=11)
    ax4.set_title('% of Thresholds Used per Feature', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.invert_yaxis()

    # Color by coverage
    colors = plt.cm.RdYlGn(coverage_per_feature / 100)
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # 5. Top 10 most important bits
    ax5 = fig.add_subplot(gs[2, 0])
    importance_flat_indexed = [(i // num_thresholds, i % num_thresholds, val)
                                for i, val in enumerate(importance_flat)]
    importance_flat_indexed.sort(key=lambda x: x[2], reverse=True)

    top_10 = importance_flat_indexed[:10]
    labels = [f'F{f}T{t}' for f, t, _ in top_10]
    values = [v for _, _, v in top_10]

    ax5.barh(range(len(labels)), values, edgecolor='black', alpha=0.7, color='orange')
    ax5.set_yticks(range(len(labels)))
    ax5.set_yticklabels(labels, fontsize=9)
    ax5.set_xlabel('Importance Weight', fontsize=11)
    ax5.set_title('Top 10 Most Important Bits', fontsize=12, fontweight='bold')
    ax5.invert_yaxis()
    ax5.grid(True, alpha=0.3, axis='x')

    # 6. Statistics text box
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')

    stats_text = f"""
    IMPORTANCE STATISTICS

    Total bits: {importance.size}
    Zero importance bits: {(importance_flat == 0).sum()} ({100*(importance_flat == 0).mean():.1f}%)
    Used bits: {(importance_flat > 0).sum()} ({100*(importance_flat > 0).mean():.1f}%)

    Importance range: [{importance.min():.4f}, {importance.max():.4f}]
    Mean importance: {importance.mean():.4f}
    Std importance: {importance.std():.4f}
    Median importance: {np.median(importance_flat):.4f}

    Features with 0% usage: {(coverage_per_feature == 0).sum()}
    Features with 100% usage: {(coverage_per_feature == 100).sum()}

    Most important feature: {feature_importance.argmax()} (avg={feature_importance.max():.4f})
    Least important feature: {feature_importance.argmin()} (avg={feature_importance.min():.4f})
    """

    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Bit Importance Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved importance visualization to: {save_path}")
    plt.close()

    return importance


if __name__ == "__main__":
    print("Loading model to visualize importance...")
    print("This assumes you have a trained model with encoder and lut_layer")
    print("\nUsage in your training script:")
    print("="*70)
    print("""
from plot_importance import plot_importance_analysis

# After training:
encoder = model[0]
lut_layer = model[2]

# Visualize importance
plot_importance_analysis(encoder, lut_layer, save_path='my_importance.png')
    """)
    print("="*70)
