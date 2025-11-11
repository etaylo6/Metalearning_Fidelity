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
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import shared modules
from beam_physics import beam_displacement_with_features
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, MODEL_CONFIG
from data_generator import generate_samples, calculate_combinatorial_accuracy, prepare_accuracy_data
from model_evaluation import plot_combinatorial_accuracy_surrogate_performance

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
    
    # Calculate test results for evaluation
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
        gated_nn_combinatorial, X_combined, y_combined, combination_labels, X_test, y_test_results,
        model_name="Input-Gated Neural Network"
    )
    
    print("\nAccuracy analysis complete!")
    print("Architecture: Input-level gating with physics-informed feature suppression")
    print(f"Gate strength: {CONFIG['gate_strength']}")
    print(f"Hidden layers: {CONFIG['gated_hidden_layers']}")

if __name__ == "__main__":
    main()

