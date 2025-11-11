"""
ACCURACY SURROGATE MODEL: NEURAL NETWORK WITH INPUT-LEVEL GATING
================================================================

This module implements a Gated Neural Network with input-level gating for predicting
model accuracy in combinatorial fidelity assessment. The gating mechanism is applied
directly to the input layer based on physics model flags.

Architecture:
- PyTorch-based neural network
- Input-level gating: Gate mask applied to inputs before network processing
- Physics-informed gating: Gates inputs based on temperature and shear model flags
- Shared hidden layers: All inputs pass through same hidden layers after gating
- Sigmoid output: Bounded [0, 1] predictions

Key Features:
1. Physics-informed gating: Suppresses irrelevant inputs based on feature flags
2. Input-level gating: Applied before network processing for efficiency
3. Adaptive gating strength: Configurable gate strength parameter
4. Shared feature extraction: All gated inputs processed through shared layers
5. Early stopping: Prevents overfitting during training

Gating Strategy:
- Temperature disabled (temp_enabled=0): Suppresses temperature parameter (T)
- Shear disabled (shear_enabled=0): Suppresses shear parameters (G, A, κ)
- Gate strength: Controls how strongly inputs are suppressed (0.0 = no gate, 1.0 = complete gate)

Input Structure:
- Continuous parameters: [P, L, I, E, G, A, κ, T]
- Binary features: [temperature_enabled, shear_enabled]
- Total input dimension: 10 (8 continuous + 2 binary)

Output:
- Accuracy score in [0, 1] range
- 1.0 = perfect accuracy (no error)
- 0.0 = complete error

To adapt for different systems:
1. Update gating logic in GatedNeuralNetwork.forward() if input structure changes
2. Modify gate_mask indices if parameter order changes
3. Adjust CONFIG dictionary with appropriate hyperparameters
4. Update parameter indices in gating logic for different physics models
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import shared modules
from beam_physics import beam_displacement_with_features
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, MODEL_CONFIG
from data_generator import generate_samples, calculate_combinatorial_accuracy, prepare_accuracy_data

# ============================================================================
# GATED NEURAL NETWORK CONFIGURATION
# ============================================================================

CONFIG = {
    'random_seed': DATA_CONFIG['random_seed'],
    
    # Gated Neural Network Configuration
    'gated_hidden_layers': [8],        # Single hidden layer - enough for 4 combinations
    'gated_learning_rate': 0.01,        # Higher learning rate for faster training
    'gated_batch_size': 32,             # Smaller batch size for faster iteration
    'gated_epochs': 100,                # Much fewer epochs for faster runs
    'gated_patience': 100,              # Less patience for faster convergence
    'gated_dropout': 0.0,               # No dropout for simple problem
    'gate_strength': 1.0                # Strength of gating (0.0 = no gate, 1.0 = complete gate)
}

# ============================================================================
# GATED NEURAL NETWORK MODEL
# ============================================================================

class GatedNeuralNetwork(nn.Module):
    """
    Neural Network with mode-dependent gating for physics-informed feature selection.
    
    This architecture applies gating at the input level, suppressing irrelevant
    parameters based on feature flags before processing through the network.
    """
    
    def __init__(self, input_dim, hidden_layers, dropout=0.2, gate_strength=0.7):
        super(GatedNeuralNetwork, self).__init__()
        self.gate_strength = gate_strength
        
        # Define which inputs are relevant for each physics model
        # Input order: [P, L, I, E, G, A, κ, T, temp_enabled, shear_enabled]
        # For generic use: parameter indices can be configured
        self.temperature_param_idx = 7  # T parameter index
        self.shear_param_indices = [4, 5, 6]  # G, A, κ parameter indices
        self.feature_flag_indices = [input_dim - 2, input_dim - 1]  # Last 2 are feature flags
        
        # Shared layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()  # Output bounded [0,1]
        )
    
    def forward(self, x):
        # Extract feature flags (last 2 elements)
        feature_flags = x[:, -2:]  # [temp_enabled, shear_enabled]
        temp_enabled = feature_flags[:, 0:1]  # Extract temp flag
        shear_enabled = feature_flags[:, 1:2]  # Extract shear flag
        
        # Create gating mask based on feature flags
        gate_mask = torch.ones_like(x)
        
        # For temperature (temp_enabled=0), zero out temperature parameter (T)
        temp_mask = (temp_enabled == 0).float()  # 1 if no temp, 0 if temp enabled
        gate_mask[:, self.temperature_param_idx] = 1 - self.gate_strength * temp_mask.squeeze()
        
        # For Euler-Bernoulli (shear_enabled=0), completely zero out G, A, κ influence
        euler_mask = (shear_enabled == 0).float()  # 1 if Euler-Bernoulli, 0 if Timoshenko
        for idx in self.shear_param_indices:
            gate_mask[:, idx] = 1 - self.gate_strength * euler_mask.squeeze()
        
        # Apply gating
        gated_input = x * gate_mask
        
        # Forward pass
        features = self.shared_layers(gated_input)
        output = self.output_layer(features)
        
        return output

def fit_gated_neural_network(X, y):
    """
    Fit a Gated Neural Network that gates inputs based on physics model.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (accuracy scores in [0, 1])
    
    Returns:
        GatedNNWrapper object with predict() method interface
    """
    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG['random_seed'])
    np.random.seed(CONFIG['random_seed'])
    
    # Normalize inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=CONFIG['gated_batch_size'], shuffle=True)
    
    # Initialize model
    model = GatedNeuralNetwork(
        input_dim=X_scaled.shape[1],
        hidden_layers=CONFIG['gated_hidden_layers'],
        dropout=CONFIG['gated_dropout'],
        gate_strength=CONFIG['gate_strength']
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['gated_learning_rate'])
    
    # Training with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG['gated_epochs']):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= CONFIG['gated_patience']:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Wrapper class for sklearn-like interface
    class GatedNNWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
            self.model.eval()  # Set to evaluation mode
            
        def predict(self, X, return_std=False):
            with torch.no_grad():
                X_scaled = self.scaler.transform(X)
                X_tensor = torch.FloatTensor(X_scaled)
                predictions = self.model(X_tensor).squeeze().numpy()
                
                if return_std:
                    # Gated NN doesn't provide uncertainty, return zeros
                    return predictions, np.zeros(len(X))
                else:
                    return predictions
    
    return GatedNNWrapper(model, scaler)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_combinatorial_accuracy_surrogate_performance(gated_nn_accuracy, X_combined, y_combined, 
                                                       combination_labels, X_test, 
                                                       y_test_results):
    """Plot Gated Neural Network combinatorial accuracy surrogate model performance with extrapolation testing"""
    
    # Get the actual combinations that were used (excluding reference)
    unique_combinations = np.unique(X_combined[:, -2:], axis=0)
    combination_names = []
    
    for combo in unique_combinations:
        if np.array_equal(combo, [1, 0]):
            combination_names.append('Euler-Bernoulli [1,0]')
        elif np.array_equal(combo, [0, 1]):
            combination_names.append('No Temperature [0,1]')
        elif np.array_equal(combo, [0, 0]):
            combination_names.append('Minimal Model [0,0]')
    
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
        
        # Gated Neural Network predictions
        y_pred_train = gated_nn_accuracy.predict(X_combo)
        r2_train = r2_score(y_combo, y_pred_train)
        
        ax.scatter(y_combo, y_pred_train, alpha=0.6, color='black', s=30, label='Training')
        
        if combo_name is not None:
            y_test_combo = 1 - y_test_results[combo_name]['relative_error']  # Convert to accuracy scores
            y_pred_test = gated_nn_accuracy.predict(X_test_extended)
            r2_test = r2_score(y_test_combo, y_pred_test)
            ax.scatter(y_test_combo, y_pred_test, alpha=0.6, color=color, s=50, 
                      marker='^', label='Extrapolation')
        else:
            r2_test = np.nan
            y_test_combo = None
        
        # Perfect prediction line
        all_true = np.concatenate([y_combo, y_test_combo if y_test_combo is not None else []])
        all_pred = np.concatenate([y_pred_train, y_pred_test if not np.isnan(r2_test) else []])
        min_val = min(np.min(all_true), np.min(all_pred))
        max_val = max(np.max(all_true), np.max(all_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        
        ax.set_xlabel('True Accuracy Score')
        ax.set_ylabel('Predicted Accuracy Score')
        title = f'{name} - Gated Neural Network\nTrain R² = {r2_train:.4f}'
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
    print("GATED NEURAL NETWORK SURROGATE PERFORMANCE")
    print("="*70)
    
    for i, (combo, name) in enumerate(zip(unique_combinations, combination_names)):
        mask = np.all(X_combined[:, -2:] == combo, axis=1)
        X_combo = X_combined[mask]
        y_combo = y_combined[mask]
        
        if len(X_combo) == 0:
            continue
            
        # Training performance
        y_pred_train = gated_nn_accuracy.predict(X_combo)
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
            y_pred_test = gated_nn_accuracy.predict(X_test_extended)
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
                                      gated_nn_accuracy.predict(X_combined[np.all(X_combined[:, -2:] == combo, axis=1)]))
                       for combo, name in zip(unique_combinations, combination_names) if len(X_combined[np.all(X_combined[:, -2:] == combo, axis=1)]) > 0}
    }

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run accuracy surrogate analysis"""
    print("="*60)
    print("BEAM MODEL ACCURACY SURROGATE ANALYSIS - INPUT-GATED NEURAL NETWORK")
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
    print("Building combinatorial accuracy surrogate model...")
    print("Feature vector: [temperature_enabled, shear_enabled]")
    print("Combinations: [1,1]=Full, [1,0]=Euler-Bernoulli, [0,1]=No Temperature, [0,0]=Minimal")
    
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
    
    # Fit combinatorial accuracy surrogate (Gated Neural Network)
    print(f"Fitting Gated Neural Network surrogate with mode-dependent input gating...")
    gated_nn_combinatorial = fit_gated_neural_network(X_combined, y_combined)
    
    # Plot combinatorial accuracy surrogate performance
    combinatorial_performance = plot_combinatorial_accuracy_surrogate_performance(
        gated_nn_combinatorial, X_combined, y_combined, combination_labels, X_test, combinatorial_results
    )
    
    print("\nAccuracy analysis complete!")
    print("Architecture: Input-level gating with physics-informed feature suppression")
    print(f"Gate strength: {CONFIG['gate_strength']}")
    print(f"Hidden layers: {CONFIG['gated_hidden_layers']}")

if __name__ == "__main__":
    main()

