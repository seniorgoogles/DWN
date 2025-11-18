"""
Quick script to plot importance after training jsc_thermlayer.py
Just add this at the end of your training script!
"""
import sys
sys.path.insert(0, 'src')
from plot_importance import plot_importance_analysis

# Add this to your jsc_thermlayer.py after setting importance:
# (assumes you have encoder and lut_layer as model[0] and model[2])

def add_to_jsc_script():
    """
    Add these lines to jsc_thermlayer.py after line 301:

    # Line 301: model[0].set_importance_from_lut(model[2], method='weighted')

    # ADD THESE LINES:
    from plot_importance import plot_importance_analysis
    plot_importance_analysis(model[0], model[2], save_path='jsc_importance.png')
    """
    pass

if __name__ == "__main__":
    print(__doc__)
    print("\nInstructions:")
    print("="*70)
    print("Add these 2 lines to your jsc_thermlayer.py after setting importance:")
    print()
    print("    from plot_importance import plot_importance_analysis")
    print("    plot_importance_analysis(model[0], model[2], save_path='jsc_importance.png')")
    print()
    print("="*70)
    print("\nThis will generate a comprehensive visualization showing:")
    print("  • Heatmap of importance per feature")
    print("  • Distribution of importance values")
    print("  • Which features are most/least important")
    print("  • Coverage statistics")
    print("  • Top 10 most important bits")
