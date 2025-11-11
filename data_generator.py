"""
DATA GENERATOR MODULE
=====================

This module provides functions for generating parameter samples and calculating
accuracy metrics for model fidelity assessment. It is designed to be generic
and work with any physics model that follows the expected interface.

Key Functions:
- generate_samples: Generate parameter samples using various sampling strategies
- calculate_combinatorial_accuracy: Calculate accuracy for different model combinations
- prepare_accuracy_data: Prepare data for surrogate model training

Data Structure:
The module expects a physics model function that takes parameters as:
    model_func(*params, feature_vector) -> output
    
Parameters are expected to be in a dictionary format:
    PARAM_BOUNDS = {
        'param1': [min_val, max_val],
        'param2': [min_val, max_val],
        ...
    }

To adapt for different systems:
1. Update PARAM_BOUNDS dictionary with your parameter names and ranges
2. Ensure your physics model function matches the expected signature
3. Update parameter order in calculate_combinatorial_accuracy if needed
"""

import numpy as np
from scipy.stats import qmc
from pyDOE import lhs

# ============================================================================
# SAMPLING FUNCTIONS
# ============================================================================

def generate_samples(param_bounds, n_samples, method='halton', boundary_bias=0.1, 
                     extrapolation_fraction=0.0, random_seed=42):
    """
    Generate parameter samples using various sampling strategies.
    
    This function supports multiple sampling methods:
    - 'lhs': Latin Hypercube Sampling
    - 'halton': Halton sequence (quasi-random)
    - 'halton_boundary': Halton with boundary bias and optional extrapolation
    
    Args:
        param_bounds: Dictionary of parameter bounds {param_name: [min, max]}
        n_samples: Number of samples to generate
        method: Sampling method ('lhs', 'halton', 'halton_boundary')
        boundary_bias: Fraction of samples near boundaries (for halton_boundary)
        extrapolation_fraction: Fraction of samples from extrapolation range
        random_seed: Random seed for reproducibility
    
    Returns:
        X: Array of shape (n_samples, n_params) with parameter values
        param_names: List of parameter names in order
    """
    param_names = list(param_bounds.keys())
    n_params = len(param_names)
    
    np.random.seed(random_seed)
    
    if method == 'lhs':
        # Standard Latin Hypercube
        # Note: pyDOE.lhs() doesn't accept random_state, so we set np.random.seed() above
        X_base = lhs(n_params, samples=n_samples)
        
    elif method == 'halton':
        # Halton sequence
        sampler = qmc.Halton(d=n_params, seed=random_seed)
        X_base = sampler.random(n=n_samples)
        
    elif method == 'halton_boundary':
        # Halton with boundary bias and optional extrapolation
        n_interior = int(n_samples * (1 - boundary_bias - extrapolation_fraction))
        n_boundary = int(n_samples * boundary_bias)
        n_extrapolation = n_samples - n_interior - n_boundary
        
        # Interior samples (training range)
        sampler = qmc.Halton(d=n_params, seed=random_seed)
        X_interior = sampler.random(n=n_interior) if n_interior > 0 else np.empty((0, n_params))
        
        # Boundary samples (near edges of training range)
        X_boundary = np.random.rand(n_boundary, n_params) if n_boundary > 0 else np.empty((0, n_params))
        for i in range(n_boundary):
            for j in range(n_params):
                if np.random.rand() < 0.5:  # 50% chance to be at boundary
                    X_boundary[i, j] = np.random.choice([
                        np.random.uniform(0, 0.1), 
                        np.random.uniform(0.9, 1.0)
                    ])
        
        # Extrapolation samples (wider range)
        X_extrapolation = np.random.rand(n_extrapolation, n_params) if n_extrapolation > 0 else np.empty((0, n_params))
        for i in range(n_extrapolation):
            for j in range(n_params):
                if np.random.rand() < 0.3:  # 30% chance for extrapolation values
                    X_extrapolation[i, j] = np.random.choice([
                        np.random.uniform(-0.2, 0.0),  # Below training range
                        np.random.uniform(1.0, 1.2)    # Above training range
                    ])
        
        # Combine all samples
        if n_interior > 0 and n_boundary > 0 and n_extrapolation > 0:
            X_base = np.vstack([X_interior, X_boundary, X_extrapolation])
        elif n_interior > 0 and n_boundary > 0:
            X_base = np.vstack([X_interior, X_boundary])
        elif n_interior > 0:
            X_base = X_interior
        elif n_boundary > 0:
            X_base = X_boundary
        else:
            X_base = X_extrapolation
        
        np.random.shuffle(X_base)
        
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Scale to parameter bounds
    X_scaled = np.zeros_like(X_base)
    for i, param in enumerate(param_names):
        bounds = param_bounds[param]
        X_scaled[:, i] = bounds[0] + X_base[:, i] * (bounds[1] - bounds[0])
    
    return X_scaled, param_names

# ============================================================================
# ACCURACY CALCULATION FUNCTIONS
# ============================================================================

def calculate_combinatorial_accuracy(X_samples, model_func, param_names, 
                                     feature_combinations=None, 
                                     reference_features=None,
                                     combination_names=None):
    """
    Calculate accuracy for all combinations of optional features.
    
    This function evaluates how different model configurations (feature combinations)
    compare to a reference model. Accuracy is defined as 1 - relative_error.
    
    Args:
        X_samples: Array of shape (n_samples, n_params) with parameter values
        model_func: Physics model function that takes (*params, feature_vector) -> output
        param_names: List of parameter names in order
        feature_combinations: List of feature vectors to test (default: all combinations)
        reference_features: Reference feature vector for ground truth (default: [1, 1])
        combination_names: Names for each combination (optional)
    
    Returns:
        results: Dictionary with keys for each combination name containing:
            - 'combination': Feature vector [temp_enabled, shear_enabled]
            - 'ground_truth': Ground truth outputs from reference model
            - 'approximation': Approximate outputs from this combination
            - 'relative_error': Relative error compared to reference
            - 'accuracy_score': Accuracy score (1 - relative_error)
    """
    if reference_features is None:
        reference_features = [1, 1]
    
    if feature_combinations is None:
        feature_combinations = [[1, 1], [1, 0], [0, 1], [0, 0]]
        if combination_names is None:
            combination_names = ['Full Model', 'EB + Temp', 'EB + Shear', 'Euler-Bernoulli']
    
    if combination_names is None:
        combination_names = [f'Combination {i}' for i in range(len(feature_combinations))]
    
    # Ground truth: reference model
    ground_truth = []
    for sample in X_samples:
        # Unpack parameters: expects [P, L, I, E, G, A, κ, T] for beam model
        # For generic use, unpack as many parameters as needed
        if len(sample) == 8:
            P, L, I, E, G, A, κ, T = sample
            disp_ref = model_func(P, L, I, E, G, A, κ, T, reference_features)
        else:
            # Generic unpacking for other models
            disp_ref = model_func(*sample, reference_features)
        ground_truth.append(disp_ref)
    
    ground_truth = np.array(ground_truth)
    
    # Test all combinations
    results = {}
    
    for combo, name in zip(feature_combinations, combination_names):
        if combo == reference_features:
            # Skip reference combination (it would have 100% accuracy)
            continue
        
        approximation = []
        for sample in X_samples:
            # Unpack parameters: expects [P, L, I, E, G, A, κ, T] for beam model
            if len(sample) == 8:
                P, L, I, E, G, A, κ, T = sample
                disp_approx = model_func(P, L, I, E, G, A, κ, T, combo)
            else:
                # Generic unpacking for other models
                disp_approx = model_func(*sample, combo)
            approximation.append(disp_approx)
        
        approximation = np.array(approximation)
        
        # Calculate accuracy metrics
        relative_error = np.abs(ground_truth - approximation) / np.abs(ground_truth)
        
        results[name] = {
            'combination': combo,
            'ground_truth': ground_truth,
            'approximation': approximation,
            'relative_error': relative_error,
            'accuracy_score': 1 - relative_error  # Higher is better
        }
    
    return results

def prepare_accuracy_data(X_samples, combinatorial_results, param_names):
    """
    Prepare data for surrogate model training from combinatorial accuracy results.
    
    This function combines all combinations into a single dataset suitable for
    training a surrogate model that predicts accuracy given parameters and feature flags.
    
    Args:
        X_samples: Array of shape (n_samples, n_params) with parameter values
        combinatorial_results: Dictionary from calculate_combinatorial_accuracy
        param_names: List of parameter names in order
    
    Returns:
        X_combined: Array of shape (n_total_samples, n_params + n_features)
                   Each row: [param1, param2, ..., temp_enabled, shear_enabled]
        y_combined: Array of shape (n_total_samples,) with accuracy scores
        combination_labels: List of combination names for each sample
    """
    X_combined = []
    y_combined = []
    combination_labels = []
    
    for combo_name, results in combinatorial_results.items():
        # Add feature vector to input
        combo_vector = results['combination']
        
        for i, sample in enumerate(X_samples):
            # Input: [P, L, I, E, G, A, κ, T, temp_enabled, shear_enabled]
            extended_sample = np.append(sample, combo_vector)
            X_combined.append(extended_sample)
            
            # Target: accuracy score (1 - relative_error)
            y_combined.append(results['accuracy_score'][i])
            combination_labels.append(combo_name)
    
    X_combined = np.array(X_combined)
    y_combined = np.array(y_combined)
    
    return X_combined, y_combined, combination_labels

