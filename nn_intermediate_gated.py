"""
ACCURACY SURROGATE MODEL: PyTorch Neural Network with Intermediate-Layer Gating
==============================================================================

This script implements a PyTorch Neural Network with gating applied in an intermediate
hidden layer. Unlike input-level gating, this architecture allows the network to first
process inputs through initial hidden layers, then apply physics-informed gating to
the intermediate features before continuing through additional layers.

Architecture Description:
- **Model Type**: Feedforward Neural Network with intermediate-layer gating
- **Input Features**: Beam parameters (P, L, I, E, G, A, κ, T) concatenated with
  binary feature flags ([temperature_enabled, shear_enabled]).
- **Pre-Gate Layers**: Initial hidden layers that process raw inputs and extract features
- **Gated Layer**: An intermediate layer where physics-informed gating is applied based
  on feature flags. Temperature and shear-related features are suppressed when their
  corresponding flags are disabled.
- **Post-Gate Layers**: Additional hidden layers that process the gated features
- **Output Layer**: A single output neuron with a Sigmoid activation to bound predictions
  between 0 and 1 (representing accuracy).
- **Optimizer**: Adam optimizer.
- **Regularization**: Dropout layers and early stopping.
- **Advantages**: Allows the network to learn non-linear feature representations before
  applying gating, potentially capturing more complex relationships. Doesn't require
  explicit knowledge of which parameters map to which features.
- **Disadvantages**: More complex than input-level gating, and the gating mechanism
  operates on learned features rather than raw parameters, which may be less interpretable.

Modularization:
- Uses `beam_physics.py` for beam displacement calculations.
- Uses `data_generator.py` for sample generation and accuracy calculation.
- Uses `beam_config.py` for centralized configuration management.

To run this script:
1. Ensure `torch`, `numpy`, `scikit-learn`, `matplotlib` are installed.
2. Ensure `beam_physics.py`, `data_generator.py`, `beam_config.py` are in the Python path.
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
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, MODEL_CONFIG, NN_ARCH2_HIDDEN_GATED_CONFIG
from data_generator import generate_samples, calculate_combinatorial_accuracy, prepare_accuracy_data

# Consolidate configuration
CONFIG = {**DATA_CONFIG, **NN_ARCH2_HIDDEN_GATED_CONFIG}

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class IntermediateGatedNeuralNetwork(nn.Module):
    """
    Neural Network with intermediate-layer gating for physics-informed feature selection.
    
    This architecture applies gating in an intermediate hidden layer, allowing the network
    to first learn feature representations before applying physics-informed gating.
    
    Architecture: Input -> Pre-Gate Layers -> Gated Layer -> Post-Gate Layers -> Output
    """
    
    def __init__(self, input_dim, hidden_layers, gated_layer_size, dropout=0.1, gate_strength=1.0):
        super(IntermediateGatedNeuralNetwork, self).__init__()
        self.gate_strength = gate_strength
        self.gated_layer_size = gated_layer_size
        
        # Hidden layers before gating
        pre_gate_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            pre_gate_layers.append(nn.Linear(prev_dim, hidden_dim))
            pre_gate_layers.append(nn.ReLU())
            pre_gate_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.pre_gate_layers = nn.Sequential(*pre_gate_layers)
        
        # Gated layer - this is the key architectural difference
        self.gated_layer = nn.Linear(prev_dim, gated_layer_size)
        
        # Hidden layers after gating
        post_gate_layers = []
        prev_dim = gated_layer_size
        for hidden_dim in [16, 8]:  # Fixed post-gate architecture
            post_gate_layers.append(nn.Linear(prev_dim, hidden_dim))
            post_gate_layers.append(nn.ReLU())
            post_gate_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.post_gate_layers = nn.Sequential(*post_gate_layers)
        
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
        
        # Forward pass through pre-gate layers
        pre_gate_features = self.pre_gate_layers(x)
        
        # Apply gated layer (Architecture 2 characteristic)
        gated_features = self.gated_layer(pre_gate_features)
        
        # Create gating mask based on feature flags for the gated features
        gate_mask = torch.ones_like(gated_features)
        
        # For temperature (temp_enabled=0), suppress temperature-related features
        temp_mask = (temp_enabled == 0).float()  # 1 if no temp, 0 if temp enabled
        # Suppress roughly 1/4 of the gated features for temperature
        temp_suppression = self.gate_strength * temp_mask
        # Expand temp_suppression to match the feature slice dimensions
        temp_suppression_expanded = temp_suppression.expand(-1, self.gated_layer_size//4)
        gate_mask[:, :self.gated_layer_size//4] *= (1 - temp_suppression_expanded)
        
        # For Euler-Bernoulli (shear_enabled=0), suppress shear-related features
        euler_mask = (shear_enabled == 0).float()  # 1 if Euler-Bernoulli, 0 if Timoshenko
        # Suppress roughly 1/4 of the gated features for shear effects
        shear_suppression = self.gate_strength * euler_mask
        # Expand shear_suppression to match the feature slice dimensions
        shear_suppression_expanded = shear_suppression.expand(-1, self.gated_layer_size//4)
        gate_mask[:, self.gated_layer_size//4:self.gated_layer_size//2] *= (1 - shear_suppression_expanded)
        
        # Apply gating to the intermediate layer
        gated_features = gated_features * gate_mask
        
        # Forward pass through post-gate layers
        post_gate_features = self.post_gate_layers(gated_features)
        
        # Final output
        output = self.output_layer(post_gate_features)
        
        return output

def fit_intermediate_gated_neural_network(X, y):
    """
    Fit an Intermediate Gated Neural Network for accuracy prediction.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (accuracy scores in [0, 1])
    
    Returns:
        IntermediateGatedNNWrapper object with predict() method interface
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
    model = IntermediateGatedNeuralNetwork(
        input_dim=X_scaled.shape[1],
        hidden_layers=CONFIG['gated_hidden_layers'],
        gated_layer_size=CONFIG['gated_layer_size'],
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
    class IntermediateGatedNNWrapper:
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
    
    return IntermediateGatedNNWrapper(model, scaler)

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_combinatorial_accuracy_surrogate_performance(model, X_combined, y_combined, 
                                                       combination_labels, X_test, 
                                                       y_test_results):
    """Plot Intermediate Gated Neural Network combinatorial accuracy surrogate model performance with extrapolation testing"""
    
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
        
        # Intermediate Gated Neural Network predictions
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
        
        # Perfect prediction line
        all_true = np.concatenate([y_combo, y_test_combo if y_test_combo is not None else []])
        all_pred = np.concatenate([y_pred_train, y_pred_test if not np.isnan(r2_test) else []])
        min_val = min(np.min(all_true), np.min(all_pred))
        max_val = max(np.max(all_true), np.max(all_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
        
        ax.set_xlabel('True Accuracy Score')
        ax.set_ylabel('Predicted Accuracy Score')
        title = f'{name} - Intermediate Gated NN\nTrain R² = {r2_train:.4f}'
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
    print("INTERMEDIATE GATED NEURAL NETWORK - SURROGATE PERFORMANCE")
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
    """Run accuracy surrogate analysis"""
    print("="*60)
    print("BEAM MODEL ACCURACY SURROGATE ANALYSIS - INTERMEDIATE GATED NN")
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
    
    # Generate test samples for evaluation with much wider range
    print("Evaluating on extrapolation points (wide range)...")
    print(f"Training range: ±10% around nominal values")
    print(f"Extrapolation range: ±25-100% around nominal values")
    
    X_test, _ = generate_samples(
        EXTRAPOLATION_BOUNDS,
        CONFIG['n_test_samples'],
        'halton',
        0.0,
        0.0,
        CONFIG['random_seed'] + 1  # Different seed for test data
    )
    
    # Calculate accuracy for all combinations using shared function
    print("Building combinatorial accuracy surrogate model...")
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
    
    # Fit combinatorial accuracy surrogate (Intermediate Gated Neural Network)
    print(f"Fitting Intermediate Gated Neural Network surrogate with intermediate-layer gating...")
    intermediate_gated_nn = fit_intermediate_gated_neural_network(X_combined, y_combined)
    
    # Calculate test results for visualization
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
        intermediate_gated_nn, X_combined, y_combined, combination_labels, X_test, y_test_results
    )
    
    print("\nAccuracy analysis complete!")
    print("Architecture: Intermediate-layer gating with physics-informed feature suppression")
    print(f"Gate strength: {CONFIG['gate_strength']}")
    print(f"Pre-gate hidden layers: {CONFIG['gated_hidden_layers']}")
    print(f"Gated layer size: {CONFIG['gated_layer_size']}")
    print("\nAdvantages:")
    print("  - Doesn't require information about subsystems")
    print("  - Can learn about non-linear trends in input space")
    print("\nDisadvantages:")
    print("  - Adds a lot of complexity")
    print("  - Variables can be wrongly considered")

if __name__ == "__main__":
    main()

