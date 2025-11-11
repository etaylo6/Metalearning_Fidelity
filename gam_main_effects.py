"""
ACCURACY SURROGATE MODEL: GAM WITH MAIN EFFECTS ONLY
=====================================================

This module implements a Generalized Additive Model (GAM) with main effects only
(no interactions) for predicting model accuracy in combinatorial fidelity assessment.

Architecture:
- Uses pygam library for GAM implementation
- Main effects: Smooth splines for continuous features, linear terms for binary features
- No interactions: Only individual feature effects (simpler than GAM²)
- Regularization tuning via grid search
- Handles mixed continuous and binary features appropriately

Key Features:
1. Main effects only: Each feature contributes independently (no interactions)
2. Smooth splines for continuous parameters
3. Linear terms for binary features
4. Regularization tuning to prevent overfitting
5. Bounded predictions in [0, 1] range for accuracy scores

Input Structure:
- Continuous parameters: [P, L, I, E, G, A, κ, T]
- Binary features: [temperature_enabled, shear_enabled]
- Total input dimension: 10 (8 continuous + 2 binary)

Output:
- Accuracy score in [0, 1] range
- 1.0 = perfect accuracy (no error)
- 0.0 = complete error

To adapt for different systems:
1. Update parameter names in logging functions
2. Adjust n_features if number of parameters changes
3. Modify continuous_features and binary_features indices if feature order changes
4. Update CONFIG dictionary with appropriate hyperparameters
"""

import numpy as np

# Import shared modules
from beam_physics import beam_displacement_with_features
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, MODEL_CONFIG
from data_generator import generate_samples, calculate_combinatorial_accuracy, prepare_accuracy_data
from model_evaluation import plot_combinatorial_accuracy_surrogate_performance, log_training_testing_data

# Import pygam for GAM
try:
    from pygam import LinearGAM, s, l
    PYGAM_AVAILABLE = True
    print("GAM: Using pygam for Generalized Additive Model with main effects only")
except ImportError:
    raise ImportError("pygam is required for GAM. Install with: pip install pygam")

# ============================================================================
# NUMPY-BASED UTILITIES (sklearn-free)
# ============================================================================

class StandardScaler:
    """Numpy-based StandardScaler replacement (sklearn-free)"""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X):
        """Fit the scaler to data"""
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        return self
    
    def transform(self, X):
        """Transform data using fitted parameters"""
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before transforming")
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

# ============================================================================
# GAM CONFIGURATION
# ============================================================================

CONFIG = {
    'random_seed': DATA_CONFIG['random_seed'],
    
    # GAM Configuration (main effects only)
    'gam_n_splines': 10,              # Number of splines per feature
    'gam_spline_order': 3,            # Spline order (cubic)
    'gam_fit_intercept': True,        # Fit intercept
    'gam_max_iter': 1000,             # Maximum iterations
    'gam_tol': 1e-4,                  # Convergence tolerance
    'gam_link': 'identity',           # Link function (identity for continuous output)
    'gam_tune_regularization': True,  # Use grid search for regularization tuning
    'gam_lam_range': np.logspace(-3, 1, 8),  # Regularization range for tuning
    'gam_cv_folds': 3,                # Cross-validation folds for tuning
}

# ============================================================================
# GAM MODEL FUNCTIONS
# ============================================================================

def fit_gam_model(X, y):
    """
    Fit a Generalized Additive Model (GAM) with main effects only (no interactions).
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (bounded [0,1])
    
    Returns:
        GAMWrapper object with predict() method interface
    """
    np.random.seed(CONFIG['random_seed'])
    
    # Normalize inputs for numerical stability (using numpy-based StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use pygam for GAM (Generalized Additive Model with main effects only)
    print(f"Fitting GAM model (main effects only) using pygam...")
    n_features = X_scaled.shape[1]
    n_splines = CONFIG['gam_n_splines']
    
    # Identify binary/categorical features (last 2 are temperature_enabled and shear_enabled)
    # Features: [P, L, I, E, G, A, κ, T, temp_enabled, shear_enabled]
    # For generic use: assume last N features are binary (default: 2)
    n_binary_features = 2  # Can be adjusted for different systems
    continuous_features = list(range(n_features - n_binary_features))
    binary_features = list(range(n_features - n_binary_features, n_features))
    
    # Build GAM formula with main effects only (no interactions)
    # For continuous features: use smooth splines
    # For binary features: use linear terms
    
    # Start with first continuous feature
    combined_terms = s(continuous_features[0], n_splines=n_splines, spline_order=CONFIG['gam_spline_order'])
    
    # Add remaining continuous features as smooth terms
    for i in continuous_features[1:]:
        combined_terms = combined_terms + s(i, n_splines=n_splines, spline_order=CONFIG['gam_spline_order'])
    
    # Add binary features as linear terms (more appropriate for binary variables)
    for i in binary_features:
        # Use linear term for binary features - they don't need smoothing
        combined_terms = combined_terms + l(i)
    
    print(f"  Model terms: {len(continuous_features)} smooth splines + {len(binary_features)} linear terms")
    print(f"  Total features: {n_features} (no interactions)")
    
    # Create base GAM model
    gam_base = LinearGAM(terms=combined_terms, max_iter=CONFIG['gam_max_iter'], tol=CONFIG['gam_tol'])
    gam_base.link = 'identity'
    gam_base.distribution = 'normal'
    
    # Regularization tuning using pygam's gridsearch method
    if CONFIG.get('gam_tune_regularization', True):
        print(f"  Tuning regularization using gridsearch...")
        try:
            lam_range = CONFIG.get('gam_lam_range', np.logspace(-3, 1, 8))
            
            # Use pygam's built-in gridsearch
            gam_base.gridsearch(X_scaled, y, lam=lam_range)
            gam = gam_base
            print(f"  Regularization tuning completed")
        except Exception as e:
            print(f"  Warning: Grid search failed ({e}), fitting without tuning")
            gam = gam_base
            gam.fit(X_scaled, y)
    else:
        # Fit without tuning
        gam = gam_base
        gam.fit(X_scaled, y)
    
    # Wrapper class for predict() interface
    class GAMWrapper:
        def __init__(self, gam, scaler):
            self.gam = gam
            self.scaler = scaler
            
        def predict(self, X, return_std=False):
            X_scaled = self.scaler.transform(X)
            predictions = self.gam.predict(X_scaled)
            
            # Ensure predictions are in [0,1] range
            predictions = np.clip(predictions, 0, 1)
            
            if return_std:
                # GAM doesn't provide uncertainty by default, return zeros
                return predictions, np.zeros(len(X))
            else:
                return predictions
    
    return GAMWrapper(gam, scaler)

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run accuracy surrogate analysis with GAM"""
    print("="*60)
    print("BEAM MODEL ACCURACY SURROGATE ANALYSIS - GAM")
    print("="*60)
    
    # Generate samples using shared data generator
    print(f"Generating {DATA_CONFIG['n_samples']} samples using {DATA_CONFIG['sampling_method']}...")
    X_samples, param_names = generate_samples(
        TRAINING_BOUNDS, 
        DATA_CONFIG['n_samples'], 
        DATA_CONFIG['sampling_method'],
        DATA_CONFIG['boundary_bias'],
        DATA_CONFIG.get('extrapolation_fraction', 0.0),
        DATA_CONFIG['random_seed']
    )
    
    # Generate test samples for evaluation with much wider range
    print("Evaluating on extrapolation points (wide range)...")
    print(f"Training range: ±10% around nominal values")
    print(f"Extrapolation range: ±25-100% around nominal values")
    
    X_test, _ = generate_samples(
        EXTRAPOLATION_BOUNDS,
        DATA_CONFIG['n_test_samples'],
        'halton',
        0.0,
        0.0,
        DATA_CONFIG['random_seed'] + 1  # Different seed for test data
    )
    
    # Calculate accuracy for all combinations using shared function
    print("Building combinatorial accuracy surrogate model with GAM...")
    print("Feature vector: [temperature_enabled, shear_enabled]")
    print("Combinations: [1,1]=Full, [1,0]=EB+Temp, [0,1]=EB+Shear, [0,0]=Euler-Bernoulli")
    
    combinatorial_results = calculate_combinatorial_accuracy(
        X_samples,
        beam_displacement_with_features,
        param_names,
        MODEL_CONFIG['feature_combinations'],
        MODEL_CONFIG['reference_features'],
        MODEL_CONFIG['combination_names']
    )
    
    # Prepare data for surrogate model training
    X_combined, y_combined, combination_labels = prepare_accuracy_data(
        X_samples, combinatorial_results, param_names
    )
    
    # Fit GAM surrogate for combinatorial accuracy prediction (main effects only)
    print(f"Fitting GAM surrogate with main effects only (no interactions)...")
    gam_combinatorial = fit_gam_model(X_combined, y_combined)
    
    # Calculate test results for evaluation and logging
    print("\nCalculating test results for evaluation...")
    y_test_results = calculate_combinatorial_accuracy(
        X_test,
        beam_displacement_with_features,
        param_names,
        MODEL_CONFIG['feature_combinations'],
        MODEL_CONFIG['reference_features'],
        MODEL_CONFIG['combination_names']
    )
    
    # Plot combinatorial accuracy surrogate performance
    combinatorial_performance = plot_combinatorial_accuracy_surrogate_performance(
        gam_combinatorial, X_combined, y_combined, combination_labels, X_test, y_test_results, 
        model_name="GAM"
    )
    
    # Log all training and testing data
    print("\n" + "="*60)
    print("LOGGING ALL DATA POINTS")
    print("="*60)
    log_results = log_training_testing_data(
        X_combined=X_combined,
        y_combined=y_combined,
        combination_labels=combination_labels,
        model=gam_combinatorial,
        X_test=X_test,
        y_test_results=y_test_results,
        param_names=param_names,
        output_dir='logs',
        model_name="GAM"
    )
    
    print("\nGAM accuracy analysis complete!")
    print("\nLog files created:")
    print(f"  - Training data: {log_results['training_file']}")
    print(f"  - Testing data: {log_results['testing_file']}")
    print(f"  - Combined data: {log_results['combined_file']}")
    print(f"  - Summary statistics: {log_results['summary_file']}")

if __name__ == "__main__":
    main()

