"""
BENCHMARK ALL MODELS
====================

This script trains all accuracy surrogate models and aggregates performance metrics
for each feature combination. It collects:
- Training time for each model
- Performance metrics (R², MSE, MAE, max error, mean error) for each feature combination
- Training and extrapolation performance

Output:
- CSV file with aggregated results: `benchmark_results_<timestamp>.csv`
- Summary statistics printed to console
"""

import numpy as np
import pandas as pd
import time
import os
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Import shared modules
from beam_physics import beam_displacement_with_features
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, MODEL_CONFIG
from data_generator import generate_samples, calculate_combinatorial_accuracy, prepare_accuracy_data

# Import model training functions
from gam_main_effects import fit_gam_model
from gam2_tensor_interactions import fit_gam2_model
from nn_simple_feature_vector import fit_simple_neural_network
from nn_input_gated import fit_gated_neural_network
from nn_intermediate_gated import fit_intermediate_gated_neural_network
from nn_activation_gated import fit_activation_gated_neural_network
from nn_conditional_heads import fit_conditional_neural_network

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

MODELS = {
    'GAM': {
        'name': 'GAM with Main Effects Only',
        'fit_function': fit_gam_model,
        'needs_feature_combinations': False
    },
    'GAM2': {
        'name': 'GAM² with Tensor Interactions',
        'fit_function': fit_gam2_model,
        'needs_feature_combinations': False
    },
    'NN_Simple': {
        'name': 'Simple Neural Network',
        'fit_function': fit_simple_neural_network,
        'needs_feature_combinations': False
    },
    'NN_Input_Gated': {
        'name': 'Input-Gated Neural Network',
        'fit_function': fit_gated_neural_network,
        'needs_feature_combinations': False
    },
    'NN_Intermediate_Gated': {
        'name': 'Intermediate-Gated Neural Network',
        'fit_function': fit_intermediate_gated_neural_network,
        'needs_feature_combinations': False
    },
    'NN_Activation_Gated': {
        'name': 'Activation-Gated Neural Network',
        'fit_function': fit_activation_gated_neural_network,
        'needs_feature_combinations': False
    },
    'NN_Conditional': {
        'name': 'Conditional Neural Network',
        'fit_function': fit_conditional_neural_network,
        'needs_feature_combinations': True
    },
}

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model_performance(model, X_train, y_train, X_test, y_test_results):
    """
    Evaluate model performance for each feature combination.
    
    Args:
        model: Trained model with predict() method
        X_train: Training data with feature flags
        y_train: Training targets
        X_test: Test data (parameters only, no feature flags)
        y_test_results: Test results from calculate_combinatorial_accuracy
    
    Returns:
        dict: Performance metrics for each combination
    """
    results = {}
    
    # Get unique combinations from test results (these exclude the reference)
    unique_combinations = []
    combo_to_name = {}
    for combo_name, data in y_test_results.items():
        combo = data['combination']
        combo_tuple = tuple(combo.tolist() if hasattr(combo, 'tolist') else combo)
        if combo_tuple not in combo_to_name:
            unique_combinations.append(combo)
            combo_to_name[combo_tuple] = combo_name
    
    for combo in unique_combinations:
        # Convert combo to numpy array for comparison
        combo_arr = np.array(combo)
        combo_tuple = tuple(combo_arr.tolist())
        combo_name = combo_to_name.get(combo_tuple, f"Combination {combo}")
        
        # Training data for this combination
        mask_train = np.all(X_train[:, -2:] == combo_arr, axis=1)
        X_train_combo = X_train[mask_train]
        y_train_combo = y_train[mask_train]
        
        if len(X_train_combo) == 0:
            continue
        
        # Test data for this combination
        combo_name_test = None
        y_test_combo = None
        
        for name_check, data in y_test_results.items():
            data_combo = data['combination']
            data_combo_arr = np.array(data_combo.tolist() if hasattr(data_combo, 'tolist') else data_combo)
            if np.array_equal(data_combo_arr, combo_arr):
                combo_name_test = name_check
                y_test_combo = 1 - data['relative_error']  # Convert to accuracy scores
                break
        
        # Training predictions
        y_pred_train = model.predict(X_train_combo)
        
        # Training metrics
        train_r2 = r2_score(y_train_combo, y_pred_train)
        train_mse = mean_squared_error(y_train_combo, y_pred_train)
        train_mae = mean_absolute_error(y_train_combo, y_pred_train)
        train_errors = np.abs(y_train_combo - y_pred_train)
        train_max_error = np.max(train_errors)
        train_mean_error = np.mean(train_errors)
        
        # Test predictions and metrics
        if y_test_combo is not None:
            X_test_extended = np.column_stack([X_test, np.tile(combo_arr, (len(X_test), 1))])
            y_pred_test = model.predict(X_test_extended)
            
            test_r2 = r2_score(y_test_combo, y_pred_test)
            test_mse = mean_squared_error(y_test_combo, y_pred_test)
            test_mae = mean_absolute_error(y_test_combo, y_pred_test)
            test_errors = np.abs(y_test_combo - y_pred_test)
            test_max_error = np.max(test_errors)
            test_mean_error = np.mean(test_errors)
        else:
            test_r2 = np.nan
            test_mse = np.nan
            test_mae = np.nan
            test_max_error = np.nan
            test_mean_error = np.nan
        
        # Store results
        combo_key = f"[{int(combo_arr[0])},{int(combo_arr[1])}]"
        results[combo_key] = {
            'combination': combo_arr.tolist(),
            'combination_name': combo_name,
            'train_r2': train_r2,
            'train_mse': train_mse,
            'train_mae': train_mae,
            'train_max_error': train_max_error,
            'train_mean_error': train_mean_error,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_max_error': test_max_error,
            'test_mean_error': test_mean_error,
            'n_train_samples': len(X_train_combo),
            'n_test_samples': len(y_test_combo) if y_test_combo is not None else 0
        }
    
    return results

# ============================================================================
# BENCHMARKING FUNCTION
# ============================================================================

def benchmark_all_models(suppress_plots=True):
    """
    Train all models and collect performance metrics.
    
    Args:
        suppress_plots: If True, suppress matplotlib plots during training
    """
    if suppress_plots:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    
    print("="*80)
    print("BENCHMARKING ALL ACCURACY SURROGATE MODELS")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Generate shared data
    print("Generating shared training and test data...")
    print(f"Training samples: {DATA_CONFIG['n_samples']}")
    print(f"Test samples: {DATA_CONFIG['n_test_samples']}")
    print(f"Sampling method: {DATA_CONFIG['sampling_method']}")
    print()
    
    X_samples, param_names = generate_samples(
        TRAINING_BOUNDS,
        DATA_CONFIG['n_samples'],
        DATA_CONFIG['sampling_method'],
        DATA_CONFIG['boundary_bias'],
        DATA_CONFIG.get('extrapolation_fraction', 0.0),
        DATA_CONFIG['random_seed']
    )
    
    X_test, _ = generate_samples(
        EXTRAPOLATION_BOUNDS,
        DATA_CONFIG['n_test_samples'],
        'halton',
        0.0,
        0.0,
        DATA_CONFIG['random_seed'] + 1
    )
    
    # Calculate accuracy for all combinations
    print("Calculating combinatorial accuracy...")
    combinatorial_results = calculate_combinatorial_accuracy(
        X_samples,
        beam_displacement_with_features,
        param_names,
        MODEL_CONFIG['feature_combinations'],
        MODEL_CONFIG['reference_features'],
        MODEL_CONFIG['combination_names']
    )
    
    # Prepare training data
    X_combined, y_combined, combination_labels = prepare_accuracy_data(
        X_samples, combinatorial_results, param_names
    )
    
    # Calculate test results
    y_test_results = calculate_combinatorial_accuracy(
        X_test,
        beam_displacement_with_features,
        param_names,
        MODEL_CONFIG['feature_combinations'],
        MODEL_CONFIG['reference_features'],
        MODEL_CONFIG['combination_names']
    )
    
    print("Data generation complete!")
    print()
    
    # Benchmark each model
    all_results = []
    
    for model_key, model_info in MODELS.items():
        print("="*80)
        print(f"Training: {model_info['name']}")
        print("="*80)
        
        try:
            # Time training (suppress model output for cleaner benchmark output)
            start_time = time.time()
            
            # Special handling for conditional NN (needs feature_combinations)
            if model_info['needs_feature_combinations']:
                # Extract unique feature combinations from combinatorial_results
                feature_combinations_list = []
                seen_combos = set()
                for combo_name, results in combinatorial_results.items():
                    combo_vector = results['combination']
                    # Convert to tuple for set comparison
                    combo_tuple = tuple(combo_vector.tolist() if hasattr(combo_vector, 'tolist') else combo_vector)
                    if combo_tuple not in seen_combos:
                        feature_combinations_list.append(combo_vector)
                        seen_combos.add(combo_tuple)
                
                print(f"  Training with {len(feature_combinations_list)} feature combinations...")
                # Note: We don't suppress output here as training progress is useful
                model = model_info['fit_function'](
                    X_combined, 
                    y_combined,
                    feature_combinations_list
                )
            else:
                print(f"  Training model...")
                # Note: We don't suppress output here as training progress is useful
                model = model_info['fit_function'](X_combined, y_combined)
            
            training_time = time.time() - start_time
            
            print(f"  Training completed in {training_time:.2f} seconds")
            print()
            
            # Evaluate performance
            print("  Evaluating performance...")
            performance = evaluate_model_performance(
                model,
                X_combined,
                y_combined,
                X_test,
                y_test_results
            )
            
            # Store results
            for combo_key, metrics in performance.items():
                result_row = {
                    'model': model_key,
                    'model_name': model_info['name'],
                    'feature_combination': combo_key,
                    'combination_name': metrics['combination_name'],
                    'training_time_seconds': training_time,
                    'n_train_samples': metrics['n_train_samples'],
                    'n_test_samples': metrics['n_test_samples'],
                    'train_r2': metrics['train_r2'],
                    'train_mse': metrics['train_mse'],
                    'train_mae': metrics['train_mae'],
                    'train_max_error': metrics['train_max_error'],
                    'train_mean_error': metrics['train_mean_error'],
                    'test_r2': metrics['test_r2'],
                    'test_mse': metrics['test_mse'],
                    'test_mae': metrics['test_mae'],
                    'test_max_error': metrics['test_max_error'],
                    'test_mean_error': metrics['test_mean_error'],
                }
                all_results.append(result_row)
            
            print(f"Evaluation complete for {model_info['name']}")
            print()
            
        except Exception as e:
            print(f"ERROR: Failed to train {model_info['name']}")
            print(f"Error message: {str(e)}")
            print()
            import traceback
            traceback.print_exc()
            print()
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'benchmark_results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f'benchmark_results_{timestamp}.csv')
    results_df.to_csv(output_file, index=False)
    
    print("="*80)
    print("BENCHMARKING COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_file}")
    print()
    
    # Print summary statistics
    print_summary_statistics(results_df)
    
    return results_df

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary_statistics(results_df):
    """Print summary statistics for all models."""
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print()
    
    # Training time summary
    print("Training Time Summary:")
    print("-" * 80)
    time_summary = results_df.groupby('model_name')['training_time_seconds'].first().sort_values()
    for model, time in time_summary.items():
        print(f"  {model:40s}: {time:8.2f} seconds")
    print()
    
    # Performance by feature combination
    print("Performance by Feature Combination:")
    print("-" * 80)
    
    for combo in results_df['feature_combination'].unique():
        combo_results = results_df[results_df['feature_combination'] == combo]
        combo_name = combo_results['combination_name'].iloc[0]
        
        print(f"\n{combo_name} ({combo}):")
        print(f"  {'Model':40s} {'Train R²':>10s} {'Test R²':>10s} {'Train Max Err':>14s} {'Test Max Err':>13s} {'Train Mean Err':>15s} {'Test Mean Err':>14s}")
        print("  " + "-" * 118)
        
        for _, row in combo_results.iterrows():
            print(f"  {row['model_name']:40s} "
                  f"{row['train_r2']:10.4f} "
                  f"{row['test_r2']:10.4f} "
                  f"{row['train_max_error']:14.6f} "
                  f"{row['test_max_error']:13.6f} "
                  f"{row['train_mean_error']:15.6f} "
                  f"{row['test_mean_error']:14.6f}")
    
    print()
    
    # Aggregate statistics
    print("Aggregate Statistics (across all combinations):")
    print("-" * 80)
    
    # Calculate aggregate statistics
    agg_data = []
    for model_name in results_df['model_name'].unique():
        model_df = results_df[results_df['model_name'] == model_name]
        agg_data.append({
            'model_name': model_name,
            'train_r2_mean': model_df['train_r2'].mean(),
            'train_r2_std': model_df['train_r2'].std(),
            'test_r2_mean': model_df['test_r2'].mean(),
            'test_r2_std': model_df['test_r2'].std(),
            'train_max_error_mean': model_df['train_max_error'].mean(),
            'train_max_error_max': model_df['train_max_error'].max(),
            'test_max_error_mean': model_df['test_max_error'].mean(),
            'test_max_error_max': model_df['test_max_error'].max(),
            'train_mean_error_mean': model_df['train_mean_error'].mean(),
            'test_mean_error_mean': model_df['test_mean_error'].mean(),
            'training_time_seconds': model_df['training_time_seconds'].iloc[0]
        })
    
    agg_df = pd.DataFrame(agg_data)
    print(agg_df.to_string(index=False, float_format='%.6f'))
    print()
    
    # Best performing models
    print("Best Performing Models (by Test R²):")
    print("-" * 80)
    
    for combo in results_df['feature_combination'].unique():
        combo_df = results_df[results_df['feature_combination'] == combo]
        # Filter out NaN values for test_r2
        combo_df_valid = combo_df[combo_df['test_r2'].notna()]
        if len(combo_df_valid) > 0:
            best_idx = combo_df_valid['test_r2'].idxmax()
            best_row = combo_df_valid.loc[best_idx]
            print(f"  {combo:10s} ({best_row['model_name']:40s}): "
                  f"R² = {best_row['test_r2']:.4f}, "
                  f"Max Error = {best_row['test_max_error']:.6f}, "
                  f"Mean Error = {best_row['test_mean_error']:.6f}")
    print()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results_df = benchmark_all_models(suppress_plots=True)
    print("\nBenchmarking complete! Check the output file for detailed results.")

