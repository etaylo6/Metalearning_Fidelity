
import numpy as np
# sklearn-free implementation - using numpy/scipy alternatives
from scipy.stats import qmc
from pyDOE import lhs
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# Import pygam for GAM^2
try:
    from pygam import LinearGAM, s, te, f, l
    PYGAN_AVAILABLE = True
    print("GAM^2: Using pygam for Generalized Additive Model with tensor product interactions")
except ImportError:
    raise ImportError("pygam is required for GAM^2. Install with: pip install pygam")

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

def r2_score(y_true, y_pred):
    """Calculate R² score (coefficient of determination)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1 - (ss_res / ss_tot)

def mean_squared_error(y_true, y_pred):
    """Calculate mean squared error"""
    return np.mean((y_true - y_pred) ** 2)

def mean_absolute_error(y_true, y_pred):
    """Calculate mean absolute error"""
    return np.mean(np.abs(y_true - y_pred))

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'n_samples': 5000,  # Reduced samples for faster training
    'sampling_method': 'lhs',  # 'lhs', 'halton', 'halton_boundary'
    'boundary_bias': 0.1,  # Fraction of samples near boundaries
    'random_seed': 42,
    
    # GAM^2 Configuration
    'gam_n_splines': 10,              # Number of splines per feature
    'gam_spline_order': 3,            # Spline order (cubic)
    'gam_fit_intercept': True,        # Fit intercept
    'gam_max_iter': 1000,             # Maximum iterations
    'gam_tol': 1e-4,                  # Convergence tolerance
    'gam_link': 'identity',           # Link function (identity for continuous output)
    'gam_interactions': True,         # Include first-level interactions
    'gam_interaction_terms': None,    # Specific interactions (None = smart selection)
    'gam_max_interactions': 35,       # Maximum number of interaction terms (increased)
    'gam_tune_regularization': True,  # Use grid search for regularization tuning
    'gam_lam_range': np.logspace(-3, 1, 8),  # Regularization range for tuning (reduced for speed)
    'gam_cv_folds': 3,                # Cross-validation folds for tuning
}

# Parameter bounds (Timoshenko-Ehrenfest beam with thermal effects)
# Training range (narrow)
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

# Extrapolation range (moderately wider)
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

# Use training bounds for current analysis
PARAM_BOUNDS = TRAINING_BOUNDS

# Temperature coefficients
TEMP_COEFFS = {
    'T_ref': 293,            # Reference temperature (20°C in Kelvin)
    'alpha_E': -0.0005,      # Temperature coefficient for Young's modulus (/K)
    'alpha_G': -0.0005,      # Temperature coefficient for shear modulus (/K) 
    'alpha_T': 0.000012      # Thermal expansion coefficient (/K) for steel
}

# ============================================================================
# BEAM MODELS
# ============================================================================
def beam_displacement_timoshenko(P, L, I, E, G, A, κ, T):
    # Temperature-dependent moduli
    E_T = E * (1 + TEMP_COEFFS['alpha_E'] * (T - TEMP_COEFFS['T_ref']))
    G_T = G * (1 + TEMP_COEFFS['alpha_G'] * (T - TEMP_COEFFS['T_ref']))
    
    # Temperature-dependent length (thermal expansion)
    L_T = L * (1 + TEMP_COEFFS['alpha_T'] * (T - TEMP_COEFFS['T_ref']))
    
    # Timoshenko-Ehrenfest displacement
    bending = (P * L_T**3) / (3 * E_T * I)
    shear = (P * L_T) / (κ * G_T * A)
    return bending + shear

def beam_displacement_euler_bernoulli(P, L, I, E, T):
    # Temperature-dependent modulus
    E_T = E * (1 + TEMP_COEFFS['alpha_E'] * (T - TEMP_COEFFS['T_ref']))
    
    # Temperature-dependent length (thermal expansion)
    L_T = L * (1 + TEMP_COEFFS['alpha_T'] * (T - TEMP_COEFFS['T_ref']))
    
    # Euler-Bernoulli displacement (bending only, no shear)
    displacement = (P * L_T**3) / (3 * E_T * I)
    return displacement

def beam_displacement_with_features(P, L, I, E, G, A, κ, T, feature_vector):
    """
    feature_vector: [temperature_enabled, shear_enabled]
    [1, 1] = Full Timoshenko-Ehrenfest with temperature (Full Model)
    [1, 0] = Euler-Bernoulli with temperature (EB + Temp)
    [0, 1] = Timoshenko-Ehrenfest without temperature (EB + Shear)
    [0, 0] = Euler-Bernoulli without temperature (Euler-Bernoulli)
    """
    temp_enabled, shear_enabled = feature_vector
    
    # Temperature effects
    if temp_enabled:
        # Use actual temperature
        E_eff = E * (1 + TEMP_COEFFS['alpha_E'] * (T - TEMP_COEFFS['T_ref']))
        G_eff = G * (1 + TEMP_COEFFS['alpha_G'] * (T - TEMP_COEFFS['T_ref']))
        L_eff = L * (1 + TEMP_COEFFS['alpha_T'] * (T - TEMP_COEFFS['T_ref']))
    else:
        # Use reference temperature (no thermal effects)
        E_eff = E * (1 + TEMP_COEFFS['alpha_E'] * (TEMP_COEFFS['T_ref'] - TEMP_COEFFS['T_ref']))
        G_eff = G * (1 + TEMP_COEFFS['alpha_G'] * (TEMP_COEFFS['T_ref'] - TEMP_COEFFS['T_ref']))
        L_eff = L * (1 + TEMP_COEFFS['alpha_T'] * (TEMP_COEFFS['T_ref'] - TEMP_COEFFS['T_ref']))
    
    # Shear effects
    if shear_enabled:
        # Timoshenko-Ehrenfest (bending + shear)
        bending = (P * L_eff**3) / (3 * E_eff * I)
        shear = (P * L_eff) / (κ * G_eff * A)
        return bending + shear
    else:
        # Euler-Bernoulli (bending only)
        return (P * L_eff**3) / (3 * E_eff * I)

# ============================================================================
# ACCURACY CALCULATIONS
# ============================================================================

def calculate_combinatorial_accuracy(X_samples, reference_features=[1, 1]):

    # Ground truth: full model with reference features
    ground_truth = []
    for sample in X_samples:
        P, L, I, E, G, A, κ, T = sample
        disp_ref = beam_displacement_with_features(P, L, I, E, G, A, κ, T, reference_features)
        ground_truth.append(disp_ref)
    
    ground_truth = np.array(ground_truth)
    
    # Test all combinations
    combinations = [[1, 1], [1, 0], [0, 1], [0, 0]]
    combination_names = ['Full Model', 'EB + Temp', 'EB + Shear', 'Euler-Bernoulli']
    
    results = {}
    
    for i, (combo, name) in enumerate(zip(combinations, combination_names)):
        if combo == reference_features:
            # Skip reference combination
            continue
            
        approximation = []
        for sample in X_samples:
            P, L, I, E, G, A, κ, T = sample
            disp_approx = beam_displacement_with_features(P, L, I, E, G, A, κ, T, combo)
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

# ============================================================================
# SAMPLING
# ============================================================================
def generate_samples(n_samples, method='halton_boundary'):
    param_names = list(PARAM_BOUNDS.keys())
    n_params = len(param_names)
    
    if method == 'lhs':
        # Standard Latin Hypercube
        np.random.seed(CONFIG['random_seed'])
        X_base = lhs(n_params, samples=n_samples)
        
    elif method == 'halton':
        # Halton sequence
        sampler = qmc.Halton(d=n_params, seed=CONFIG['random_seed'])
        X_base = sampler.random(n=n_samples)
        
    elif method == 'halton_boundary':
        # Halton with boundary bias and some extrapolation range
        boundary_bias = CONFIG['boundary_bias']
        extrapolation_fraction = 0.1  # 10% of samples from extrapolation range
        
        n_interior = int(n_samples * (1 - boundary_bias - extrapolation_fraction))
        n_boundary = int(n_samples * boundary_bias)
        n_extrapolation = n_samples - n_interior - n_boundary
        
        # Interior samples (training range)
        sampler = qmc.Halton(d=n_params, seed=CONFIG['random_seed'])
        X_interior = sampler.random(n=n_interior)
        
        # Boundary samples (near edges of training range)
        X_boundary = np.random.rand(n_boundary, n_params)
        for i in range(n_boundary):
            for j in range(n_params):
                if np.random.rand() < 0.5:  # 50% chance to be at boundary
                    X_boundary[i, j] = np.random.choice([np.random.uniform(0, 0.1), 
                                                       np.random.uniform(0.9, 1.0)])
        
        # Extrapolation samples (wider range)
        X_extrapolation = np.random.rand(n_extrapolation, n_params)
        # Some samples will be outside [0,1] range to represent extrapolation
        for i in range(n_extrapolation):
            for j in range(n_params):
                if np.random.rand() < 0.3:  # 30% chance for extrapolation values
                    X_extrapolation[i, j] = np.random.choice([
                        np.random.uniform(-0.2, 0.0),  # Below training range
                        np.random.uniform(1.0, 1.2)    # Above training range
                    ])
        
        X_base = np.vstack([X_interior, X_boundary, X_extrapolation])
        np.random.shuffle(X_base)
        
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    # Scale to parameter bounds
    X_scaled = np.zeros_like(X_base)
    for i, param in enumerate(param_names):
        bounds = PARAM_BOUNDS[param]
        X_scaled[:, i] = bounds[0] + X_base[:, i] * (bounds[1] - bounds[0])
    
    return X_scaled

# ============================================================================
# SURROGATE MODELING - GAM^2
# ============================================================================

def select_important_interactions(X_scaled, y, n_features, max_interactions, n_splines):
    """
    Select important interaction terms using a quick correlation/importance-based approach.
    
    Args:
        X_scaled: Scaled input features
        y: Target values
        n_features: Number of features
        max_interactions: Maximum number of interactions to select
        n_splines: Number of splines for terms
    
    Returns:
        List of (i, j) tuples for selected interactions
    """
    # Quick feature importance using simple linear correlation
    correlations = np.abs([np.corrcoef(X_scaled[:, i], y)[0, 1] for i in range(n_features)])
    
    # Score interaction pairs based on both features' importance
    interaction_scores = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            # Score: product of feature importances
            score = correlations[i] * correlations[j]
            interaction_scores.append((score, (i, j)))
    
    # Sort by score and select top interactions
    interaction_scores.sort(reverse=True)
    selected = [pair for _, pair in interaction_scores[:max_interactions]]
    
    return selected

def fit_gam2_model(X, y):
    """
    Fit a Generalized Additive Model with first-level interactions (GAM^2)
    with improved binary feature handling, smart interaction selection, and regularization tuning.
    
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
    
    # Use pygam for true GAM^2 (Generalized Additive Model with tensor product interactions)
    print(f"Fitting GAM^2 model using pygam...")
    n_features = X_scaled.shape[1]
    n_splines = CONFIG['gam_n_splines']
    
    # Identify binary/categorical features (last 2 are temperature_enabled and shear_enabled)
    # Features: [P, L, I, E, G, A, κ, T, temp_enabled, shear_enabled]
    continuous_features = list(range(n_features - 2))  # First 8 features are continuous
    binary_features = [n_features - 2, n_features - 1]  # Last 2 are binary flags
    
    # Build GAM formula with main effects
    # For continuous features: use smooth splines
    # For binary features: use linear terms (f() factor terms or linear)
    
    # Start with first continuous feature
    combined_terms = s(continuous_features[0], n_splines=n_splines, spline_order=CONFIG['gam_spline_order'])
    
    # Add remaining continuous features as smooth terms
    for i in continuous_features[1:]:
        combined_terms = combined_terms + s(i, n_splines=n_splines, spline_order=CONFIG['gam_spline_order'])
    
    # Add binary features as linear terms (more appropriate for binary variables)
    for i in binary_features:
        # Use linear term for binary features - they don't need smoothing
        combined_terms = combined_terms + l(i)
    
    # First-level interactions: smart selection
    interaction_count = 0
    if CONFIG['gam_interactions']:
        if CONFIG['gam_interaction_terms'] is None:
            # Smart selection: use importance-based selection
            max_interactions = CONFIG.get('gam_max_interactions', 35)
            total_possible = n_features * (n_features - 1) // 2
            
            # Select important interactions
            selected_interactions = select_important_interactions(X_scaled, y, n_features, max_interactions, n_splines)
            
            # Add selected interactions
            for i, j in selected_interactions:
                # Use tensor product for continuous-continuous interactions
                if i in continuous_features and j in continuous_features:
                    combined_terms = combined_terms + te(i, j, n_splines=[n_splines, n_splines], 
                                                       spline_order=[CONFIG['gam_spline_order'], CONFIG['gam_spline_order']])
                # Use smooth by factor for binary-continuous interactions
                elif (i in binary_features and j in continuous_features) or (j in binary_features and i in continuous_features):
                    # Smooth term for continuous feature modulated by binary factor
                    cont_idx = j if i in binary_features else i
                    bin_idx = i if i in binary_features else j
                    combined_terms = combined_terms + te(bin_idx, cont_idx, n_splines=[2, n_splines],
                                                       spline_order=[1, CONFIG['gam_spline_order']])
                # Interaction for binary-binary (use tensor with 2 splines for binary)
                else:
                    # Use tensor product with n_splines=2 for binary features
                    combined_terms = combined_terms + te(i, j, n_splines=[2, 2],
                                                       spline_order=[1, 1])
                
                interaction_count += 1
            
            if interaction_count < total_possible:
                print(f"  Note: Using {interaction_count} interaction terms (out of {total_possible} possible) based on importance")
        else:
            # Use specified interaction pairs
            for i, j in CONFIG['gam_interaction_terms']:
                combined_terms = combined_terms + te(i, j, n_splines=[n_splines, n_splines],
                                                   spline_order=[CONFIG['gam_spline_order'], CONFIG['gam_spline_order']])
                interaction_count += 1
    
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

def fit_combinatorial_accuracy_surrogate(X_samples, combinatorial_results):
    """Fit a GAM^2 surrogate model for combinatorial accuracy prediction"""
    
    # Combine all combinations into a single dataset
    X_combined = []
    y_combined = []
    combination_labels = []
    
    for combo_name, results in combinatorial_results.items():
        # Add feature vector to input (temperature_enabled, shear_enabled)
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
    
    # Fit GAM^2 surrogate for combinatorial accuracy prediction
    print(f"Fitting GAM^2 model with {X_combined.shape[1]} features...")
    print(f"  - Main effects: {X_combined.shape[1]} features")
    if CONFIG['gam_interactions']:
        n_interactions = X_combined.shape[1] * (X_combined.shape[1] - 1) // 2
        print(f"  - First-level interactions: {n_interactions} pairs")
    print(f"  - Total samples: {len(X_combined)}")
    
    gam2_accuracy = fit_gam2_model(X_combined, y_combined)
    
    return gam2_accuracy, X_combined, y_combined, combination_labels

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def log_training_testing_data(X_combined, y_combined, combination_labels, gam2_accuracy, 
                               X_test, y_test_results, output_dir='logs'):
    """
    Log all training and testing data points with their input spaces, accuracies, 
    model types, and predictions to CSV files.
    
    Args:
        X_combined: Combined training data (including feature vectors)
        y_combined: True accuracy scores for training data
        combination_labels: Model/combination names for training data
        gam2_accuracy: Trained GAM^2 model for predictions
        X_test: Test input data
        y_test_results: Test accuracy results from calculate_combinatorial_accuracy
        output_dir: Directory to save log files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ========== TRAINING DATA LOG ==========
    print(f"\nLogging training data...")
    
    # Get predictions for training data
    y_pred_train = gam2_accuracy.predict(X_combined)
    
    # Create DataFrame for training data
    training_records = []
    for i in range(len(X_combined)):
        record = {
            'Data_Type': 'Training',
            'Sample_Index': i,
            'P_Force_N': X_combined[i, 0],
            'L_Length_m': X_combined[i, 1],
            'I_MomentOfInertia_m4': X_combined[i, 2],
            'E_YoungsModulus_Pa': X_combined[i, 3],
            'G_ShearModulus_Pa': X_combined[i, 4],
            'A_CrossSectionalArea_m2': X_combined[i, 5],
            'kappa_ShearCorrectionFactor': X_combined[i, 6],
            'T_Temperature_K': X_combined[i, 7],
            'Temperature_Enabled': int(X_combined[i, 8]),
            'Shear_Enabled': int(X_combined[i, 9]),
            'Model_Type': combination_labels[i],
            'True_Accuracy': y_combined[i],
            'Predicted_Accuracy': y_pred_train[i],
            'Prediction_Error': abs(y_combined[i] - y_pred_train[i]),
            'Relative_Prediction_Error': abs(y_combined[i] - y_pred_train[i]) / y_combined[i] if y_combined[i] != 0 else 0
        }
        training_records.append(record)
    
    training_df = pd.DataFrame(training_records)
    training_file = os.path.join(output_dir, f'training_data_GAM2_{timestamp}.csv')
    training_df.to_csv(training_file, index=False)
    print(f"Training data logged to: {training_file}")
    print(f"Total training samples: {len(training_records)}")
    
    # ========== TESTING DATA LOG ==========
    print(f"\nLogging testing data...")
    
    # Get unique combinations from test results
    testing_records = []
    sample_counter = 0
    
    for combo_name, results in y_test_results.items():
        combo_vector = results['combination']
        y_test_accuracy = results['accuracy_score']
        
        # Create extended test data with feature vectors
        X_test_extended = np.column_stack([X_test, np.tile(combo_vector, (len(X_test), 1))])
        
        # Get predictions for test data
        y_pred_test = gam2_accuracy.predict(X_test_extended)
        
        for i in range(len(X_test)):
            record = {
                'Data_Type': 'Testing',
                'Sample_Index': sample_counter,
                'P_Force_N': X_test[i, 0],
                'L_Length_m': X_test[i, 1],
                'I_MomentOfInertia_m4': X_test[i, 2],
                'E_YoungsModulus_Pa': X_test[i, 3],
                'G_ShearModulus_Pa': X_test[i, 4],
                'A_CrossSectionalArea_m2': X_test[i, 5],
                'kappa_ShearCorrectionFactor': X_test[i, 6],
                'T_Temperature_K': X_test[i, 7],
                'Temperature_Enabled': int(combo_vector[0]),
                'Shear_Enabled': int(combo_vector[1]),
                'Model_Type': combo_name,
                'True_Accuracy': y_test_accuracy[i],
                'Predicted_Accuracy': y_pred_test[i],
                'Prediction_Error': abs(y_test_accuracy[i] - y_pred_test[i]),
                'Relative_Prediction_Error': abs(y_test_accuracy[i] - y_pred_test[i]) / y_test_accuracy[i] if y_test_accuracy[i] != 0 else 0
            }
            testing_records.append(record)
            sample_counter += 1
    
    testing_df = pd.DataFrame(testing_records)
    testing_file = os.path.join(output_dir, f'testing_data_GAM2_{timestamp}.csv')
    testing_df.to_csv(testing_file, index=False)
    print(f"Testing data logged to: {testing_file}")
    print(f"Total testing samples: {len(testing_records)}")
    
    # ========== COMBINED SUMMARY LOG ==========
    print(f"\nCreating combined summary...")
    
    # Combine both datasets
    combined_df = pd.concat([training_df, testing_df], ignore_index=True)
    combined_file = os.path.join(output_dir, f'combined_data_GAM2_{timestamp}.csv')
    combined_df.to_csv(combined_file, index=False)
    print(f"Combined data logged to: {combined_file}")
    print(f"Total samples: {len(combined_df)}")
    
    # ========== SUMMARY STATISTICS BY MODEL ==========
    summary_file = os.path.join(output_dir, f'summary_statistics_GAM2_{timestamp}.csv')
    
    summary_records = []
    for model_type in combined_df['Model_Type'].unique():
        model_data = combined_df[combined_df['Model_Type'] == model_type]
        
        train_data = model_data[model_data['Data_Type'] == 'Training']
        test_data = model_data[model_data['Data_Type'] == 'Testing']
        
        summary = {
            'Model_Type': model_type,
            'Total_Samples': len(model_data),
            'Training_Samples': len(train_data),
            'Testing_Samples': len(test_data),
            
            # Training metrics
            'Train_Mean_True_Accuracy': train_data['True_Accuracy'].mean() if len(train_data) > 0 else np.nan,
            'Train_Mean_Predicted_Accuracy': train_data['Predicted_Accuracy'].mean() if len(train_data) > 0 else np.nan,
            'Train_Mean_Prediction_Error': train_data['Prediction_Error'].mean() if len(train_data) > 0 else np.nan,
            'Train_R2_Score': r2_score(train_data['True_Accuracy'], train_data['Predicted_Accuracy']) if len(train_data) > 0 else np.nan,
            
            # Testing metrics
            'Test_Mean_True_Accuracy': test_data['True_Accuracy'].mean() if len(test_data) > 0 else np.nan,
            'Test_Mean_Predicted_Accuracy': test_data['Predicted_Accuracy'].mean() if len(test_data) > 0 else np.nan,
            'Test_Mean_Prediction_Error': test_data['Prediction_Error'].mean() if len(test_data) > 0 else np.nan,
            'Test_R2_Score': r2_score(test_data['True_Accuracy'], test_data['Predicted_Accuracy']) if len(test_data) > 0 else np.nan,
        }
        summary_records.append(summary)
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics logged to: {summary_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY MODEL TYPE (GAM^2)")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    return {
        'training_file': training_file,
        'testing_file': testing_file,
        'combined_file': combined_file,
        'summary_file': summary_file,
        'training_df': training_df,
        'testing_df': testing_df,
        'combined_df': combined_df,
        'summary_df': summary_df
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_combinatorial_accuracy_surrogate_performance(gam2_accuracy, X_combined, y_combined, combination_labels, X_test):
    """Plot GAM^2 combinatorial accuracy surrogate model performance with extrapolation testing"""
    
    # Get the actual combinations that were used (excluding reference)
    unique_combinations = np.unique(X_combined[:, -2:], axis=0)
    combination_names = []
    
    for combo in unique_combinations:
        if np.array_equal(combo, [1, 0]):
            combination_names.append('EB + Temp [1,0]')
        elif np.array_equal(combo, [0, 1]):
            combination_names.append('EB + Shear [0,1]')
        elif np.array_equal(combo, [0, 0]):
            combination_names.append('Euler-Bernoulli [0,0]')
    
    colors = ['red', 'green', 'purple']
    
    # Create subplot grid - 1 column per combination
    n_combinations = len(unique_combinations)
    n_cols = n_combinations
    n_rows = 2 if n_combinations > 2 else 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 8*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    # Optimize layout for better snipping/screenshots
    plt.rcParams['figure.autolayout'] = True
    
    for i, (combo, name, color) in enumerate(zip(unique_combinations, combination_names, colors)):
        ax = axes[i]
        
        # Get data for this combination
        mask = np.all(X_combined[:, -2:] == combo, axis=1)
        X_combo = X_combined[mask]
        y_combo = y_combined[mask]
        
        if len(X_combo) == 0:
            continue
        
        # Extrapolation testing setup
        X_test_extended = np.column_stack([X_test, np.tile(combo, (len(X_test), 1))])
        y_test_results = calculate_combinatorial_accuracy(X_test, reference_features=[1, 1])
        
        # Find the combination name for this combo
        combo_name = None
        for name_check, data in y_test_results.items():
            if np.array_equal(data['combination'], combo):
                combo_name = name_check
                break
        
        # GAM^2 predictions
        y_pred_train = gam2_accuracy.predict(X_combo)
        r2_train = r2_score(y_combo, y_pred_train)
        
        ax.scatter(y_combo, y_pred_train, alpha=0.6, color='black', s=30, label='Training')
        
        if combo_name is not None:
            y_test_combo = 1 - y_test_results[combo_name]['relative_error']  # Convert to accuracy scores
            y_pred_test = gam2_accuracy.predict(X_test_extended)
            r2_test = r2_score(y_test_combo, y_pred_test)
            ax.scatter(y_test_combo, y_pred_test, alpha=0.6, color=color, s=50, 
                      marker='^', label='Extrapolation')
        else:
            r2_test = np.nan
            y_test_combo = None
        
        # Perfect prediction line (on consistent 60-100% scale)
        ax.plot([0.6, 1.0], [0.6, 1.0], 'k--', alpha=0.8)
        
        # Set consistent axis limits (60% and up)
        ax.set_xlim([0.6, 1.0])
        ax.set_ylim([0.6, 1.0])
        
        ax.set_xlabel('True Accuracy Score')
        ax.set_ylabel('Predicted Accuracy Score')
        title = f'{name} - GAM²\nTrain R² = {r2_train:.4f}'
        if not np.isnan(r2_test):
            title += f', Extrap R² = {r2_test:.4f}'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Remove empty subplots
    for i in range(n_combinations, len(axes)):
        axes[i].remove()
    
    # Improve layout for better snipping
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.08, right=0.98, hspace=0.3, wspace=0.3)
    plt.show()
    
    # Summary statistics
    print("\n" + "="*70)
    print("GAM^2 SURROGATE PERFORMANCE")
    print("="*70)
    
    for i, (combo, name) in enumerate(zip(unique_combinations, combination_names)):
        mask = np.all(X_combined[:, -2:] == combo, axis=1)
        X_combo = X_combined[mask]
        y_combo = y_combined[mask]
        
        if len(X_combo) == 0:
            continue
            
        # Training performance
        y_pred_train = gam2_accuracy.predict(X_combo)
        r2_train = r2_score(y_combo, y_pred_train)
        mse_train = mean_squared_error(y_combo, y_pred_train)
        
        # Extrapolation performance
        X_test_extended = np.column_stack([X_test, np.tile(combo, (len(X_test), 1))])
        y_test_results = calculate_combinatorial_accuracy(X_test, reference_features=[1, 1])
        
        # Find the combination name for this combo
        combo_name = None
        for name_check, data in y_test_results.items():
            if np.array_equal(data['combination'], combo):
                combo_name = name_check
                break
        
        if combo_name is not None:
            y_test_combo = 1 - y_test_results[combo_name]['relative_error']  # Convert to accuracy scores
            y_pred_test = gam2_accuracy.predict(X_test_extended)
            r2_test = r2_score(y_test_combo, y_pred_test)
            mse_test = mean_squared_error(y_test_combo, y_pred_test)
            degradation = r2_train - r2_test
        else:
            r2_test = np.nan
            mse_test = np.nan
            degradation = np.nan
        
        print(f"{name}:")
        print(f"  Training R²: {r2_train:.4f}")
        if not np.isnan(r2_test):
            print(f"  Extrapolation R²: {r2_test:.4f}")
            print(f"  Degradation: {degradation:.4f}")
        print(f"  Training MSE: {mse_train:.6f}")
        if not np.isnan(mse_test):
            print(f"  Extrapolation MSE: {mse_test:.6f}")
        print(f"  Accuracy Range: [{np.min(y_combo):.4f}, {np.max(y_combo):.4f}]")
        print()
    
    return {
        'combinations': unique_combinations.tolist(),
        'combination_names': combination_names,
        'performance': {name: r2_score(y_combined[np.all(X_combined[:, -2:] == combo, axis=1)], 
                                      gam2_accuracy.predict(X_combined[np.all(X_combined[:, -2:] == combo, axis=1)]))
                       for combo, name in zip(unique_combinations, combination_names) if len(X_combined[np.all(X_combined[:, -2:] == combo, axis=1)]) > 0}
    }

# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def main():
    """Run accuracy surrogate analysis with GAM^2"""
    print("="*60)
    print("BEAM MODEL ACCURACY SURROGATE ANALYSIS - GAM^2")
    print("="*60)
    
    # Generate samples
    print(f"Generating {CONFIG['n_samples']} samples using {CONFIG['sampling_method']}...")
    X_samples = generate_samples(CONFIG['n_samples'], CONFIG['sampling_method'])
    
    # Generate test samples for evaluation with much wider range
    print("Evaluating on extrapolation points (wide range)...")
    print(f"Training range: ±10% around nominal values")
    print(f"Extrapolation range: ±25-100% around nominal values")
    
    # Use extrapolation bounds for test data
    original_bounds = PARAM_BOUNDS.copy()
    PARAM_BOUNDS.update(EXTRAPOLATION_BOUNDS)
    X_test = generate_samples(200, 'halton')  # More test points, different sampling
    PARAM_BOUNDS.update(original_bounds)  # Restore original bounds
    
    # Visualizations
    print("Generating visualizations...")
    
    # Combinatorial accuracy surrogate model
    print("Building combinatorial accuracy surrogate model with GAM^2...")
    print("Feature vector: [temperature_enabled, shear_enabled]")
    print("Combinations: [1,1]=Full, [1,0]=EB+Temp, [0,1]=EB+Shear, [0,0]=Euler-Bernoulli")
    
    # Calculate accuracy for all combinations (ground truth = full model [1,1])
    combinatorial_results = calculate_combinatorial_accuracy(X_samples, reference_features=[1, 1])
    
    # Fit combinatorial accuracy surrogate (GAM^2)
    print(f"Fitting GAM^2 surrogate with main effects and first-level interactions...")
    gam2_combinatorial, X_combined, y_combined, combination_labels = fit_combinatorial_accuracy_surrogate(
        X_samples, combinatorial_results
    )
    
    # Plot combinatorial accuracy surrogate performance
    combinatorial_performance = plot_combinatorial_accuracy_surrogate_performance(
        gam2_combinatorial, X_combined, y_combined, combination_labels, X_test
    )
    
    # Calculate test results for logging
    print("\nCalculating test results for logging...")
    y_test_results = calculate_combinatorial_accuracy(X_test, reference_features=[1, 1])
    
    # Log all training and testing data
    print("\n" + "="*60)
    print("LOGGING ALL DATA POINTS")
    print("="*60)
    log_results = log_training_testing_data(
        X_combined=X_combined,
        y_combined=y_combined,
        combination_labels=combination_labels,
        gam2_accuracy=gam2_combinatorial,
        X_test=X_test,
        y_test_results=y_test_results,
        output_dir='logs'
    )
    
    print("\nGAM^2 accuracy analysis complete!")
    print("\nLog files created:")
    print(f"  - Training data: {log_results['training_file']}")
    print(f"  - Testing data: {log_results['testing_file']}")
    print(f"  - Combined data: {log_results['combined_file']}")
    print(f"  - Summary statistics: {log_results['summary_file']}")

if __name__ == "__main__":
    main()

