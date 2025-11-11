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
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import shared modules
from beam_physics import beam_displacement_with_features
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, MODEL_CONFIG, NN_ARCH2_HIDDEN_GATED_CONFIG
from data_generator import generate_samples, calculate_combinatorial_accuracy, prepare_accuracy_data
from model_evaluation import plot_combinatorial_accuracy_surrogate_performance

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
        intermediate_gated_nn, X_combined, y_combined, combination_labels, X_test, y_test_results,
        model_name="Intermediate-Gated Neural Network"
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

