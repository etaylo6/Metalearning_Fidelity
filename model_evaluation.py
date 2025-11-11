"""
MODEL EVALUATION AND POST-PROCESSING
====================================

This module provides shared functions for evaluating and visualizing model performance,
including plotting, logging, and summary statistics. It can be used with any accuracy
surrogate model that implements a predict() method.

Functions:
- plot_combinatorial_accuracy_surrogate_performance: Plot training and test performance
- log_training_testing_data: Log data to CSV files
- print_performance_summary: Print summary statistics to console
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_combination_names(unique_combinations):
    """
    Get combination names from combination vectors.
    
    Args:
        unique_combinations: Array of combination vectors
        
    Returns:
        List of combination names
    """
    combination_names = []
    for combo in unique_combinations:
        if np.array_equal(combo, [1, 0]):
            combination_names.append('EB + Temp [1,0]')
        elif np.array_equal(combo, [0, 1]):
            combination_names.append('EB + Shear [0,1]')
        elif np.array_equal(combo, [0, 0]):
            combination_names.append('Euler-Bernoulli [0,0]')
        else:
            combination_names.append(f'Combination {combo}')
    return combination_names

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_combinatorial_accuracy_surrogate_performance(model, X_combined, y_combined, 
                                                       combination_labels, X_test, 
                                                       y_test_results, model_name="Model"):
    """
    Plot combinatorial accuracy surrogate model performance with extrapolation testing.
    
    Args:
        model: Trained model with predict() method
        X_combined: Training data with feature flags
        y_combined: Training targets
        combination_labels: Labels for each training sample
        X_test: Test data (parameters only)
        y_test_results: Test results dictionary from calculate_combinatorial_accuracy
        model_name: Name of the model for plot titles (default: "Model")
    
    Returns:
        dict: Performance metrics for each combination
    """
    # Get the actual combinations that were used (excluding reference)
    unique_combinations = np.unique(X_combined[:, -2:], axis=0)
    combination_names = get_combination_names(unique_combinations)
    
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
        if i >= len(axes):
            break
            
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
        
        # Model predictions
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
        title = f'{name} - {model_name}\nTrain R² = {r2_train:.4f}'
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
    print(f"{model_name.upper()} SURROGATE PERFORMANCE")
    print("="*70)
    
    performance_dict = {}
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
        mae_train = mean_absolute_error(y_combo, y_pred_train)
        
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
            mae_test = mean_absolute_error(y_test_combo, y_pred_test)
            degradation = r2_train - r2_test
        else:
            r2_test = np.nan
            mse_test = np.nan
            mae_test = np.nan
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
        
        performance_dict[name] = {
            'train_r2': r2_train,
            'test_r2': r2_test,
            'train_mse': mse_train,
            'test_mse': mse_test,
            'train_mae': mae_train,
            'test_mae': mae_test,
        }
    
    return {
        'combinations': unique_combinations.tolist(),
        'combination_names': combination_names,
        'performance': performance_dict
    }

# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def log_training_testing_data(X_combined, y_combined, combination_labels, model, 
                               X_test, y_test_results, param_names, output_dir='logs', 
                               model_name="Model"):
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
        model_name: Name of the model for file names (default: "Model")
    
    Returns:
        dict: Dictionary with file paths and dataframes
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_clean = model_name.replace(' ', '_').replace('²', '2').replace('^', '')
    
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
    training_file = os.path.join(output_dir, f'training_data_{model_name_clean}_{timestamp}.csv')
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
    testing_file = os.path.join(output_dir, f'testing_data_{model_name_clean}_{timestamp}.csv')
    testing_df.to_csv(testing_file, index=False)
    print(f"Testing data logged to: {testing_file}")
    print(f"Total testing samples: {len(testing_records)}")
    
    # ========== COMBINED SUMMARY LOG ==========
    print(f"\nCreating combined summary...")
    
    # Combine both datasets
    combined_df = pd.concat([training_df, testing_df], ignore_index=True)
    combined_file = os.path.join(output_dir, f'combined_data_{model_name_clean}_{timestamp}.csv')
    combined_df.to_csv(combined_file, index=False)
    print(f"Combined data logged to: {combined_file}")
    print(f"Total samples: {len(combined_df)}")
    
    # ========== SUMMARY STATISTICS BY MODEL ==========
    summary_file = os.path.join(output_dir, f'summary_statistics_{model_name_clean}_{timestamp}.csv')
    
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
    print(f"SUMMARY STATISTICS BY MODEL TYPE ({model_name.upper()})")
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

