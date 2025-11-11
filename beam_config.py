"""
BEAM CONFIGURATION MODULE
=========================

This module contains configuration parameters for the beam model analysis.
It defines parameter bounds, sampling configurations, and model settings.

To adapt for different systems:
1. Update PARAM_BOUNDS with your parameter names and ranges
2. Update TRAINING_BOUNDS and EXTRAPOLATION_BOUNDS as needed
3. Modify sampling configuration in DATA_CONFIG
4. Update feature combinations and names in MODEL_CONFIG
"""

import numpy as np

# ============================================================================
# PARAMETER BOUNDS
# ============================================================================

# Parameter bounds (Timoshenko-Ehrenfest beam with thermal effects)
# Training range (narrow) - for model training
TRAINING_BOUNDS = {
    'P': [90000, 110000],    # Force (N) - ±10% around 100 kN
    'L': [9, 11],            # Reference length (m) - ±10% around 10 m  
    'I': [0.020, 0.030],     # Moment of inertia (m^4) - ±10% around 0.025 m^4
    'E': [189e9, 231e9],     # Reference Young's modulus (Pa) - ±10% around 210 GPa
    'G': [72e9, 88e9],       # Reference shear modulus (Pa) - ±10% around 80 GPa
    'A': [0.045, 0.055],     # Cross-sectional area (m^2) - ±10% around 0.05 m^2
    'κ': [0.9, 1.1],         # Shear correction factor - ±10% around 1.0
    'T': [273, 373]          # Temperature (K) - 0°C to 100°C
}

# Extrapolation range (moderately wider) - for testing model generalization
EXTRAPOLATION_BOUNDS = {
    'P': [50000, 150000],    # Force (N) - ±50% around 100 kN
    'L': [5, 15],            # Reference length (m) - ±50% around 10 m  
    'I': [0.010, 0.040],     # Moment of inertia (m^4) - ±60% around 0.025 m^4
    'E': [150e9, 270e9],     # Reference Young's modulus (Pa) - ±30% around 210 GPa
    'G': [60e9, 100e9],      # Reference shear modulus (Pa) - ±25% around 80 GPa
    'A': [0.030, 0.070],     # Cross-sectional area (m^2) - ±40% around 0.05 m^2
    'κ': [0.7, 1.3],         # Shear correction factor - ±30% around 1.0
    'T': [250, 400]          # Temperature (K) - -23°C to 127°C
}

# ============================================================================
# DATA GENERATION CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    'n_samples': 5000,           # Number of training samples
    'n_test_samples': 200,       # Number of test samples for extrapolation
    'sampling_method': 'lhs',    # 'lhs', 'halton', 'halton_boundary'
    'boundary_bias': 0.1,        # Fraction of samples near boundaries
    'extrapolation_fraction': 0.1,  # Fraction of samples from extrapolation range
    'random_seed': 42,           # Random seed for reproducibility
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'reference_features': [1, 1],  # Reference model: Full Timoshenko with temperature
    'feature_combinations': [
        [1, 1],  # Full Model
        [1, 0],  # EB + Temp
        [0, 1],  # EB + Shear
        [0, 0]   # Euler-Bernoulli
    ],
    'combination_names': [
        'Full Model',
        'EB + Temp',
        'EB + Shear',
        'Euler-Bernoulli'
    ],
    'param_order': ['P', 'L', 'I', 'E', 'G', 'A', 'κ', 'T'],  # Parameter order
}

# ============================================================================
# MODEL-SPECIFIC CONFIGURATIONS
# ============================================================================

# GAM Configuration (Main Effects Only)
GAM_CONFIG = {
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

# GAM^2 Configuration
GAM2_CONFIG = {
    'gam_n_splines': 10,              # Number of splines per feature
    'gam_spline_order': 3,            # Spline order (cubic)
    'gam_fit_intercept': True,        # Fit intercept
    'gam_max_iter': 1000,             # Maximum iterations
    'gam_tol': 1e-4,                  # Convergence tolerance
    'gam_link': 'identity',           # Link function (identity for continuous output)
    'gam_interactions': True,         # Include first-level interactions
    'gam_interaction_terms': None,    # Specific interactions (None = smart selection)
    'gam_max_interactions': 35,       # Maximum number of interaction terms
    'gam_tune_regularization': True,  # Use grid search for regularization tuning
    'gam_lam_range': np.logspace(-3, 1, 8),  # Regularization range for tuning
    'gam_cv_folds': 3,                # Cross-validation folds for tuning
}

# MLP (Scikit-learn) Configuration
MLP_SKLEARN_CONFIG = {
    'nn_hidden_layers': (32, 16, 8, 4),  # Hidden layer sizes
    'nn_activation': 'relu',            # Activation function
    'nn_solver': 'adam',                # Optimizer
    'nn_alpha': 0.001,                  # L2 regularization
    'nn_learning_rate': 'adaptive',     # Learning rate schedule
    'nn_max_iter': 2000,                # Maximum iterations
    'nn_bounded_method': 'logit',       # 'logit' (recommended) or 'sigmoid' for [0,1] bounds
}

# Simple PyTorch NN (Feature Vector) Configuration
NN_FEATURE_VECTOR_CONFIG = {
    'hidden_layers': [64, 32, 32, 16],          # Hidden layer sizes for PyTorch NN
    'learning_rate': 0.001,                 # Learning rate for PyTorch NN
    'batch_size': 64,                       # Batch size for training
    'epochs': 120,                          # Number of training epochs
    'patience': 50,                         # Early stopping patience
    'dropout': 0.05,                        # Dropout rate
    'target_type': 'accuracy'               # 'accuracy' - what to train on
}

# PyTorch NN (Input Gated) Configuration
NN_INPUT_GATED_CONFIG = {
    'gated_hidden_layers': [8],        # Single hidden layer - enough for 4 combinations
    'gated_learning_rate': 0.01,        # Higher learning rate for faster training
    'gated_batch_size': 32,             # Smaller batch size for faster iteration
    'gated_epochs': 100,                 # Much fewer epochs for faster runs
    'gated_patience': 100,                # Less patience for faster convergence
    'gated_dropout': 0.0,               # No dropout for simple problem
    'gate_strength': 1.0               # Strength of gating (0.0 = no gate, 1.0 = complete gate)
}

# PyTorch NN (Intermediate Hidden Layer Gated) Configuration (Architecture 2)
NN_ARCH2_HIDDEN_GATED_CONFIG = {
    'gated_hidden_layers': [32, 16],    # Hidden layers before gating
    'gated_layer_size': 32,             # Size of the gated layer
    'gated_learning_rate': 0.01,        # Higher learning rate for faster training
    'gated_batch_size': 32,             # Smaller batch size for faster iteration
    'gated_epochs': 100,                # Much fewer epochs for faster runs
    'gated_patience': 20,               # Less patience for faster convergence
    'gated_dropout': 0.1,               # Light dropout for regularization
    'gate_strength': 1.0                # Strength of gating (0.0 = no gate, 1.0 = complete gate)
}

# PyTorch NN (Activation Layer Gated) Configuration (Architecture 3)
NN_ARCH3_ACTIVATION_GATED_CONFIG = {
    'gated_hidden_layers': [32],        # Hidden layers before activation layer
    'activation_layer_size': 32,        # Size of the activation suppression layer
    'gated_layer_size': 32,             # Size of the gated layer after activation
    'gated_learning_rate': 0.01,        # Higher learning rate for faster training
    'gated_batch_size': 32,             # Smaller batch size for faster iteration
    'gated_epochs': 100,                # Much fewer epochs for faster runs
    'gated_patience': 20,               # Less patience for faster convergence
    'gated_dropout': 0.1,               # Light dropout for regularization
    'suppression_threshold': 0.3,       # Threshold for low-impact input suppression
    'gate_strength': 1.0                # Strength of gating (0.0 = no gate, 1.0 = complete gate)
}

# Conditional Neural Network Configuration
CNN_CONDITIONAL_HEADS_CONFIG = {
    'cnn_shared_layers': [32, 16],      # Shared feature extraction layers
    'cnn_head_layers': [4, 2],         # Individual head layers for each feature combination
    'cnn_activation': 'relu',           # Activation function
    'cnn_dropout': 0.1,                 # Dropout for regularization
    'cnn_learning_rate': 0.01,         # Learning rate
    'cnn_batch_size': 16,               # Batch size for training
    'cnn_epochs': 100,                  # Training epochs
    'cnn_patience': 10,                 # Early stopping patience
    'cnn_random_state': 42,             # Random seed for reproducibility
}

# Gaussian Process Configuration
GP_IMPORTANCE_CONFIG = {
    'gp_kernel': 'rbf_white',  # 'rbf_white', 'matern_white', 'rbf', 'matern'
    'gp_length_scale': 1.0,    # Length scale for RBF/Matern kernel
    'gp_noise_level': 0.1,     # Noise level for WhiteKernel
    'gp_alpha': 1e-10,         # Regularization parameter
    'gp_nu': 2.5,              # Nu parameter for Matern kernel
}

# Graph Edit Importance Configuration
GRAPH_EDIT_CONFIG = {
    'graph_layout': 'spring',  # 'spring', 'bipartite', 'circular'
    'node_size_scale': 1000,
    'edge_width_scale': 2,
    'figsize': (16, 12)
}

