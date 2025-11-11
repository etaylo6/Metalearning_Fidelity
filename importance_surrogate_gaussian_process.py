"""
IMPORTANCE SURROGATE MODEL: Gaussian Process Regression for Parameter Importance
===============================================================================

This script implements a Gaussian Process Regression (GPR) model to predict local Sobol
sensitivity (parameter importance) for the beam model. Unlike accuracy surrogate models,
this model predicts how sensitive the beam displacement is to changes in each input
parameter, providing insights into which parameters have the most influence on the output.

Architecture Description:
- **Model Type**: Gaussian Process Regression (GPR)
- **Input Features**: Beam parameters (P, L, I, E, G, A, κ, T)
- **Output**: Local Sobol importance (sensitivity) for each parameter, calculated as
  (dY/dX) * (X/Y), where Y is the displacement and X is the parameter. This provides
  a normalized measure of how much a relative change in parameter X affects the output Y.
- **Kernel**: Configurable kernel (RBF, Matern, with optional WhiteKernel for noise).
  The kernel choice affects the smoothness and uncertainty quantification of the predictions.
- **Uncertainty Quantification**: GPR provides predictive uncertainty (standard deviation)
  for each prediction, which is valuable for understanding prediction confidence.
- **Multi-Output**: A separate GPR model is trained for each parameter's importance,
  allowing each parameter to have its own importance-parameter relationship.
- **Advantages**: Provides uncertainty quantification, can handle non-linear relationships,
  and is a Bayesian approach that naturally handles uncertainty. Good for small datasets.
- **Disadvantages**: Computationally expensive for large datasets, and the computational
  cost scales as O(n³) with the number of training samples. Kernel hyperparameters may
  need careful tuning.

Modularization:
- Uses `beam_physics.py` for beam displacement calculations.
- Uses `data_generator.py` for sample generation (but calculates importance separately).
- Uses `beam_config.py` for centralized configuration management.

To run this script:
1. Ensure `scikit-learn`, `numpy`, `matplotlib` are installed.
2. Ensure `beam_physics.py`, `data_generator.py`, `beam_config.py` are in the Python path.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel, Matern
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.optimize import approx_fprime

# Import shared modules
from beam_physics import beam_displacement_timoshenko
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, GP_IMPORTANCE_CONFIG
from data_generator import generate_samples

# Consolidate configuration
CONFIG = {**DATA_CONFIG, **GP_IMPORTANCE_CONFIG}

# ============================================================================
# IMPORTANCE CALCULATION
# ============================================================================

def local_sobol_importance(P, L, I, E, G, A, κ, T, eps=1e-6):
    """
    Calculate local Sobol importance (sensitivity) for beam displacement.
    
    Local Sobol importance measures how sensitive the output is to changes in each
    input parameter. It is calculated as (dY/dX) * (X/Y), which represents the
    relative change in output for a relative change in input.
    
    Args:
        P, L, I, E, G, A, κ, T: Beam parameters
        eps: Perturbation size for finite difference calculation
    
    Returns:
        Array of importance values for each parameter [P, L, I, E, G, A, κ, T]
    """
    # Calculate base displacement
    result = beam_displacement_timoshenko(P, L, I, E, G, A, κ, T)
    
    # Central difference for derivatives
    dP = (beam_displacement_timoshenko(P+eps, L, I, E, G, A, κ, T) - 
          beam_displacement_timoshenko(P-eps, L, I, E, G, A, κ, T)) / (2*eps)
    dL = (beam_displacement_timoshenko(P, L+eps, I, E, G, A, κ, T) - 
          beam_displacement_timoshenko(P, L-eps, I, E, G, A, κ, T)) / (2*eps)
    dI = (beam_displacement_timoshenko(P, L, I+eps, E, G, A, κ, T) - 
          beam_displacement_timoshenko(P, L, I-eps, E, G, A, κ, T)) / (2*eps)
    dE = (beam_displacement_timoshenko(P, L, I, E+eps, G, A, κ, T) - 
          beam_displacement_timoshenko(P, L, I, E-eps, G, A, κ, T)) / (2*eps)
    dG = (beam_displacement_timoshenko(P, L, I, E, G+eps, A, κ, T) - 
          beam_displacement_timoshenko(P, L, I, E, G-eps, A, κ, T)) / (2*eps)
    dA = (beam_displacement_timoshenko(P, L, I, E, G, A+eps, κ, T) - 
          beam_displacement_timoshenko(P, L, I, E, G, A-eps, κ, T)) / (2*eps)
    dκ = (beam_displacement_timoshenko(P, L, I, E, G, A, κ+eps, T) - 
          beam_displacement_timoshenko(P, L, I, E, G, A, κ-eps, T)) / (2*eps)
    dT = (beam_displacement_timoshenko(P, L, I, E, G, A, κ, T+eps) - 
          beam_displacement_timoshenko(P, L, I, E, G, A, κ, T-eps)) / (2*eps)
    
    # Relative sensitivity: (dY/dX) * (X/Y)
    importance = np.array([
        dP * P / result,  # P importance
        dL * L / result,  # L importance  
        dI * I / result,  # I importance
        dE * E / result,  # E importance
        dG * G / result,  # G importance
        dA * A / result,  # A importance
        dκ * κ / result,  # κ importance
        dT * T / result   # T importance
    ])
    
    return importance

# ============================================================================
# GAUSSIAN PROCESS MODEL
# ============================================================================

def create_gp_kernel():
    """Create Gaussian Process kernel based on configuration"""
    kernel_type = CONFIG['gp_kernel']
    
    if kernel_type == 'rbf_white':
        return (ConstantKernel(1.0) * 
                RBF(length_scale=CONFIG['gp_length_scale']) + 
                WhiteKernel(noise_level=CONFIG['gp_noise_level']))
    
    elif kernel_type == 'matern_white':
        return (ConstantKernel(1.0) * 
                Matern(length_scale=CONFIG['gp_length_scale'], nu=CONFIG['gp_nu']) + 
                WhiteKernel(noise_level=CONFIG['gp_noise_level']))
    
    elif kernel_type == 'rbf':
        return ConstantKernel(1.0) * RBF(length_scale=CONFIG['gp_length_scale'])
    
    elif kernel_type == 'matern':
        return ConstantKernel(1.0) * Matern(length_scale=CONFIG['gp_length_scale'], nu=CONFIG['gp_nu'])
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

def fit_gaussian_process(X, y):
    """
    Fit a Gaussian Process Regression model for importance prediction.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (importance scores)
    
    Returns:
        GPWrapper object with predict() method interface
    """
    # Normalize inputs for GP
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create kernel
    kernel = create_gp_kernel()
    
    # Fit GP
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=CONFIG['gp_alpha'],
        random_state=CONFIG['random_seed'],
        normalize_y=True
    )
    gp.fit(X_scaled, y)
    
    # Return GP with scaler for predictions
    class GPWrapper:
        def __init__(self, gp, scaler):
            self.gp = gp
            self.scaler = scaler
            
        def predict(self, X, return_std=False):
            X_scaled = self.scaler.transform(X)
            if return_std:
                return self.gp.predict(X_scaled, return_std=True)
            else:
                return self.gp.predict(X_scaled)
    
    return GPWrapper(gp, scaler)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_vs_extrapolation(gp_models, X_train, importance_train, X_test, importance_test, param_names):
    """Plot training vs extrapolation performance for all parameters with uncertainty"""
    n_params = len(param_names)
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(param_names)))
    
    for i, param in enumerate(param_names):
        ax = axes[i]
        gp_model = gp_models[param]
        
        # Get true and predicted values
        y_train_true = importance_train[:, i]
        y_test_true = importance_test[:, i]
        
        y_train_pred = gp_model.predict(X_train)
        y_test_pred, y_test_std = gp_model.predict(X_test, return_std=True)
        
        # Calculate R² scores
        r2_train = r2_score(y_train_true, y_train_pred)
        r2_test = r2_score(y_test_true, y_test_pred)
        
        # Create scatter plots
        ax.scatter(y_train_true, y_train_pred, alpha=0.6, s=30, 
                  color=colors[i], label=f'Training (R²={r2_train:.3f})', marker='o')
        ax.scatter(y_test_true, y_test_pred, alpha=0.8, s=40, 
                  color=colors[i], label=f'Extrapolation (R²={r2_test:.3f})', marker='^')
        
        # Add uncertainty bars for extrapolation points
        ax.errorbar(y_test_true, y_test_pred, yerr=2*y_test_std, 
                   fmt='none', alpha=0.3, color=colors[i], capsize=2)
        
        # Perfect prediction line
        min_val = min(np.min(y_train_true), np.min(y_test_true))
        max_val = max(np.max(y_train_true), np.max(y_test_true))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1)
        
        ax.set_xlabel(f'True {param} Importance')
        ax.set_ylabel(f'Predicted {param} Importance')
        ax.set_title(f'{param} Importance')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add R² degradation info
        degradation = r2_train - r2_test
        ax.text(0.05, 0.95, f'Degradation: {degradation:.3f}', 
                transform=ax.transAxes, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove empty subplots
    for i in range(n_params, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    plt.show()
    
    # Summary plot
    plt.figure(figsize=(12, 6))
    
    # Calculate all R² scores
    r2_train_scores = []
    r2_test_scores = []
    degradations = []
    
    for i, param in enumerate(param_names):
        gp_model = gp_models[param]
        y_train_true = importance_train[:, i]
        y_test_true = importance_test[:, i]
        y_train_pred = gp_model.predict(X_train)
        y_test_pred = gp_model.predict(X_test)
        
        r2_train = r2_score(y_train_true, y_train_pred)
        r2_test = r2_score(y_test_true, y_test_pred)
        degradation = r2_train - r2_test
        
        r2_train_scores.append(r2_train)
        r2_test_scores.append(r2_test)
        degradations.append(degradation)
    
    # Bar plot
    x = np.arange(len(param_names))
    width = 0.35
    
    plt.bar(x - width/2, r2_train_scores, width, label='Training R²', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, r2_test_scores, width, label='Extrapolation R²', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Parameters')
    plt.ylabel('R² Score')
    plt.title('Gaussian Process Surrogate Performance: Training vs Extrapolation')
    plt.xticks(x, param_names)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add degradation values on top of bars
    for i, (train_r2, test_r2, deg) in enumerate(zip(r2_train_scores, r2_test_scores, degradations)):
        plt.text(i - width/2, train_r2 + 0.01, f'{train_r2:.3f}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i + width/2, test_r2 + 0.01, f'{test_r2:.3f}', 
                ha='center', va='bottom', fontsize=8)
        plt.text(i, max(train_r2, test_r2) + 0.05, f'Δ={deg:.3f}', 
                ha='center', va='bottom', fontsize=8, color='red', fontweight='bold')
    
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.show()
    
    return r2_train_scores, r2_test_scores, degradations

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run Gaussian Process importance analysis"""
    print("="*60)
    print("GAUSSIAN PROCESS BEAM IMPORTANCE SURROGATE MODELING")
    print("="*60)
    
    # Generate samples using shared data generator
    print(f"Generating {CONFIG['n_samples']} samples using {CONFIG['sampling_method']}...")
    X_samples, param_names = generate_samples(
        TRAINING_BOUNDS, 
        CONFIG['n_samples'], 
        CONFIG['sampling_method'],
        CONFIG['boundary_bias'],
        CONFIG.get('extrapolation_fraction', 0.0),
        CONFIG['random_seed']
    )
    
    # Calculate importance for each sample
    print("Calculating local Sobol importance...")
    importance_data = []
    for sample in X_samples:
        P, L, I, E, G, A, κ, T = sample
        importance = local_sobol_importance(P, L, I, E, G, A, κ, T)
        importance_data.append(importance)
    
    importance_data = np.array(importance_data)
    
    # Fit Gaussian Process surrogates for each parameter
    print("Fitting Gaussian Process surrogates...")
    print(f"Using kernel: {CONFIG['gp_kernel']}")
    gp_models = {}
    
    for i, param in enumerate(param_names):
        print(f"  Fitting {param} importance model...")
        y = importance_data[:, i]
        gp_model = fit_gaussian_process(X_samples, y)
        
        # Calculate training performance
        y_pred = gp_model.predict(X_samples)
        r2 = r2_score(y, y_pred)
        print(f"    Training R² = {r2:.4f}")
        
        gp_models[param] = gp_model
    
    # Generate test samples for evaluation with wider range
    print("Evaluating on extrapolation points (wide range)...")
    X_test, _ = generate_samples(
        EXTRAPOLATION_BOUNDS,
        CONFIG.get('n_test_samples', 50),
        'halton',
        0.0,
        0.0,
        CONFIG['random_seed'] + 1  # Different seed for test data
    )
    
    # Calculate importance for test samples
    importance_test = []
    for sample in X_test:
        P, L, I, E, G, A, κ, T = sample
        importance = local_sobol_importance(P, L, I, E, G, A, κ, T)
        importance_test.append(importance)
    
    importance_test = np.array(importance_test)
    
    # Evaluate performance
    print("\nSURROGATE MODEL PERFORMANCE:")
    print("-" * 40)
    
    for param in param_names:
        gp_model = gp_models[param]
        param_idx = param_names.index(param)
        
        y_train = importance_data[:, param_idx]
        y_test = importance_test[:, param_idx]
        
        # Training performance
        y_train_pred = gp_model.predict(X_samples)
        r2_train = r2_score(y_train, y_train_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        
        # Testing performance
        y_test_pred = gp_model.predict(X_test)
        r2_test = r2_score(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        degradation = r2_train - r2_test
        
        print(f"{param}:")
        print(f"  Training R² = {r2_train:.4f}, MSE = {mse_train:.6f}")
        print(f"  Extrapolation R² = {r2_test:.4f}, MSE = {mse_test:.6f}")
        print(f"  Degradation = {degradation:.4f}")
        print()
    
    # Visualizations
    print("Generating visualizations...")
    
    # Training vs Extrapolation performance for all parameters
    r2_train_scores, r2_test_scores, degradations = plot_training_vs_extrapolation(
        gp_models, X_samples, importance_data, X_test, importance_test, param_names
    )
    
    # Print summary statistics
    print("\n" + "="*60)
    print("GAUSSIAN PROCESS PERFORMANCE SUMMARY")
    print("="*60)
    print(f"{'Parameter':<8} {'Train R²':<10} {'Test R²':<10} {'Degradation':<12}")
    print("-" * 45)
    for param, train_r2, test_r2, deg in zip(param_names, r2_train_scores, r2_test_scores, degradations):
        print(f"{param:<8} {train_r2:<10.4f} {test_r2:<10.4f} {deg:<12.4f}")
    
    avg_train = np.mean(r2_train_scores)
    avg_test = np.mean(r2_test_scores)
    avg_degradation = np.mean(degradations)
    print("-" * 45)
    print(f"{'Average':<8} {avg_train:<10.4f} {avg_test:<10.4f} {avg_degradation:<12.4f}")
    
    print("\nImportance analysis complete!")
    print(f"Kernel: {CONFIG['gp_kernel']}")
    print(f"Training samples: {len(X_samples)}")
    print(f"Test samples: {len(X_test)}")

if __name__ == "__main__":
    main()

