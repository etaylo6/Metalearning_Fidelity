"""
ACCURACY SURROGATE MODEL: PyTorch Neural Network with Simple Feature Vector
==========================================================================

This script implements a simple PyTorch Neural Network that treats feature flags
([temperature_enabled, shear_enabled]) as regular input features alongside the
physical parameters. This is the simplest approach where the network learns to
associate feature flags with accuracy predictions without any special gating mechanisms.

Architecture Description:
- **Model Type**: Feedforward Neural Network (Multi-Layer Perceptron)
- **Input Features**: Beam parameters (P, L, I, E, G, A, κ, T) concatenated with
  binary feature flags ([temperature_enabled, shear_enabled]).
- **Hidden Layers**: Configurable number and size of hidden layers with ReLU activation
  and dropout for regularization.
- **Output Layer**: A single output neuron with a Sigmoid activation to bound predictions
  between 0 and 1 (representing accuracy).
- **Optimizer**: Adam optimizer.
- **Regularization**: Dropout layers and early stopping based on validation loss.
- **Advantages**: Simple architecture, easy to implement and understand, can capture
  non-linear relationships between inputs and accuracy.
- **Disadvantages**: Does not explicitly model physics-informed relationships between
  feature flags and parameters. The network must learn these relationships implicitly,
  which may require more data than gated architectures.

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
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, MODEL_CONFIG, NN_FEATURE_VECTOR_CONFIG
from data_generator import generate_samples, calculate_combinatorial_accuracy, prepare_accuracy_data
from model_evaluation import plot_combinatorial_accuracy_surrogate_performance

# Consolidate configuration
CONFIG = {**DATA_CONFIG, **NN_FEATURE_VECTOR_CONFIG}

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class SimpleNeuralNetwork(nn.Module):
    """
    Simple Neural Network for accuracy prediction with feature vectors.
    
    This architecture treats feature flags as regular input features, allowing
    the network to learn implicit relationships between features and accuracy.
    
    Input: [P, L, I, E, G, A, κ, T, temp_enabled, shear_enabled]
    Output: accuracy (0 to 1)
    """
    
    def __init__(self, input_dim, hidden_layers, dropout=0.1):
        super(SimpleNeuralNetwork, self).__init__()
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer with sigmoid activation for [0,1] bounded output
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.network(x)
        # Ensure output is between 0 and 1 (accuracy range)
        return torch.sigmoid(output)

def fit_simple_neural_network(X, y):
    """
    Fit a Simple PyTorch Neural Network for accuracy prediction.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (accuracy scores in [0, 1])
    
    Returns:
        SimpleNNWrapper object with predict() method interface
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
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # Initialize model
    model = SimpleNeuralNetwork(
        input_dim=X_scaled.shape[1],
        hidden_layers=CONFIG['hidden_layers'],
        dropout=CONFIG['dropout']
    )
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG['epochs']):
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
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']}, Loss: {avg_loss:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch+1}, best loss: {best_loss:.6f}")
            break
    
    # Print final status
    if patience_counter < CONFIG['patience']:
        print(f"Training completed, final loss: {avg_loss:.6f}")
    
    # Wrapper class for sklearn-like interface
    class SimpleNNWrapper:
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
                    # NN doesn't provide uncertainty, return zeros
                    return predictions, np.zeros(len(X))
                else:
                    return predictions
    
    return SimpleNNWrapper(model, scaler)

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run accuracy surrogate analysis"""
    print("="*60)
    print("BEAM MODEL ACCURACY SURROGATE ANALYSIS - SIMPLE FEATURE VECTOR NN")
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
    
    # Fit combinatorial accuracy surrogate (Simple Neural Network)
    print(f"Fitting Simple Neural Network surrogate with feature vector approach...")
    simple_nn_combinatorial = fit_simple_neural_network(X_combined, y_combined)
    
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
        simple_nn_combinatorial, X_combined, y_combined, combination_labels, X_test, y_test_results,
        model_name="Simple Neural Network"
    )
    
    print("\nAccuracy analysis complete!")
    print("Architecture: Simple feedforward neural network with feature flags as inputs")
    print(f"Hidden layers: {CONFIG['hidden_layers']}")
    print(f"Dropout: {CONFIG['dropout']}")

if __name__ == "__main__":
    main()

