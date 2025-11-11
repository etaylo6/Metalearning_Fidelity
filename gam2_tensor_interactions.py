"""
ACCURACY SURROGATE MODEL: GAM^2 WITH TENSOR PRODUCT INTERACTIONS
================================================================

This module implements a Generalized Additive Model with tensor product interactions (GAM^2)
for predicting model accuracy in combinatorial fidelity assessment.

Architecture:
- Uses pygam library for GAM^2 implementation
- Main effects: Smooth splines for continuous features, linear terms for binary features
- Interactions: Tensor product interactions between features (first-level)
- Smart interaction selection based on feature importance
- Regularization tuning via grid search
- Handles mixed continuous and binary features appropriately

Key Features:
1. Tensor product interactions for capturing feature interactions
2. Automatic selection of important interactions based on correlation
3. Different treatment for continuous vs binary features
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
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime

# Import shared modules
from beam_physics import beam_displacement_with_features
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, MODEL_CONFIG
from data_generator import generate_samples, calculate_combinatorial_accuracy, prepare_accuracy_data

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
# GAM^2 CONFIGURATION
# ============================================================================

CONFIG = {
    'random_seed': DATA_CONFIG['random_seed'],
    
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

# ============================================================================
# GAM^2 MODEL FUNCTIONS
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
    # For generic use: assume last N features are binary (default: 2)
    n_binary_features = 2  # Can be adjusted for different systems
    continuous_features = list(range(n_features - n_binary_features))
    binary_features = list(range(n_features - n_binary_features, n_features))
    
    # Build GAM formula with main effects
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

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def log_training_testing_data(X_combined, y_combined, combination_labels, model, 
                               X_test, y_test_results, param_names, output_dir='logs'):
    """
    Log all training and testing data points with their input spaces, accuracies, 
    model types, and predictions to CSV files.
    
    Args:
        X_combined: Combined training data (including feature vectors)
        y_combined: True accuracy scores for training data
        combination_labels: Model/combination names for training data
        model: Trained surrogate model for predictions
        X_test: Test input data
        y_test_results: Test accuracy results from calculate_combinatorial_accuracy
        param_names: List of parameter names for column headers
        output_dir: Directory to save log files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create parameter column names
    param_cols = [f'{p}_{i}' for i, p in enumerate(param_names)]
    if len(param_names) == 8:  # Beam model specific
        param_cols = ['P_Force_N', 'L_Length_m', 'I_MomentOfInertia_m4', 
                     'E_YoungsModulus_Pa', 'G_ShearModulus_Pa', 
                     'A_CrossSectionalArea_m2', 'kappa_ShearCorrectionFactor', 'T_Temperature_K']
    
    # ========== TRAINING DATA LOG ==========
    print(f"\nLogging training data...")
    
    # Get predictions for training data
    y_pred_train = model.predict(X_combined)
    
    # Create DataFrame for training data
    training_records = []
    for i in range(len(X_combined)):
        record = {
            'Data_Type': 'Training',
            'Sample_Index': i,
        }
        # Add parameter columns
        for j, col in enumerate(param_cols):
            record[col] = X_combined[i, j]
        # Add feature flags
        record['Temperature_Enabled'] = int(X_combined[i, len(param_names)])
        record['Shear_Enabled'] = int(X_combined[i, len(param_names) + 1])
        # Add model info
        record['Model_Type'] = combination_labels[i]
        record['True_Accuracy'] = y_combined[i]
        record['Predicted_Accuracy'] = y_pred_train[i]
        record['Prediction_Error'] = abs(y_combined[i] - y_pred_train[i])
        record['Relative_Prediction_Error'] = abs(y_combined[i] - y_pred_train[i]) / y_combined[i] if y_combined[i] != 0 else 0
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
        y_pred_test = model.predict(X_test_extended)
        
        for i in range(len(X_test)):
            record = {
                'Data_Type': 'Testing',
                'Sample_Index': sample_counter,
            }
            # Add parameter columns
            for j, col in enumerate(param_cols):
                record[col] = X_test[i, j]
            # Add feature flags
            record['Temperature_Enabled'] = int(combo_vector[0])
            record['Shear_Enabled'] = int(combo_vector[1])
            # Add model info
            record['Model_Type'] = combo_name
            record['True_Accuracy'] = y_test_accuracy[i]
            record['Predicted_Accuracy'] = y_pred_test[i]
            record['Prediction_Error'] = abs(y_test_accuracy[i] - y_pred_test[i])
            record['Relative_Prediction_Error'] = abs(y_test_accuracy[i] - y_pred_test[i]) / y_test_accuracy[i] if y_test_accuracy[i] != 0 else 0
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

def plot_combinatorial_accuracy_surrogate_performance(model, X_combined, y_combined, 
                                                       combination_labels, X_test, 
                                                       y_test_results):
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
        
        # Find the combination name for this combo
        combo_name = None
        for name_check, data in y_test_results.items():
            if np.array_equal(data['combination'], combo):
                combo_name = name_check
                break
        
        # GAM^2 predictions
        y_pred_train = model.predict(X_combo)
        r2_train = r2_score(y_combo, y_pred_train)
        
        ax.scatter(y_combo, y_pred_train, alpha=0.6, color='black', s=30, label='Training')
        
        if combo_name is not None:
            y_test_combo = 1 - y_test_results[combo_name]['relative_error']  # Convert to accuracy scores
            y_pred_test = model.predict(X_test_extended)
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
        y_pred_train = model.predict(X_combo)
        r2_train = r2_score(y_combo, y_pred_train)
        mse_train = mean_squared_error(y_combo, y_pred_train)
        
        # Extrapolation performance
        X_test_extended = np.column_stack([X_test, np.tile(combo, (len(X_test), 1))])
        
        # Find the combination name for this combo
        combo_name = None
        for name_check, data in y_test_results.items():
            if np.array_equal(data['combination'], combo):
                combo_name = name_check
                break
        
        if combo_name is not None:
            y_test_combo = 1 - y_test_results[combo_name]['relative_error']  # Convert to accuracy scores
            y_pred_test = model.predict(X_test_extended)
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
                                      model.predict(X_combined[np.all(X_combined[:, -2:] == combo, axis=1)]))
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
    print("Building combinatorial accuracy surrogate model with GAM^2...")
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
    
    # Fit GAM^2 surrogate for combinatorial accuracy prediction
    print(f"Fitting GAM^2 surrogate with main effects and first-level interactions...")
    gam2_combinatorial = fit_gam2_model(X_combined, y_combined)
    
    # Plot combinatorial accuracy surrogate performance
    combinatorial_performance = plot_combinatorial_accuracy_surrogate_performance(
        gam2_combinatorial, X_combined, y_combined, combination_labels, X_test, combinatorial_results
    )
    
    # Calculate test results for logging
    print("\nCalculating test results for logging...")
    y_test_results = calculate_combinatorial_accuracy(
        X_test,
        beam_displacement_with_features,
        param_names,
        MODEL_CONFIG['feature_combinations'],
        MODEL_CONFIG['reference_features'],
        MODEL_CONFIG['combination_names']
    )
    
    # Log all training and testing data
    print("\n" + "="*60)
    print("LOGGING ALL DATA POINTS")
    print("="*60)
    log_results = log_training_testing_data(
        X_combined=X_combined,
        y_combined=y_combined,
        combination_labels=combination_labels,
        model=gam2_combinatorial,
        X_test=X_test,
        y_test_results=y_test_results,
        param_names=param_names,
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

