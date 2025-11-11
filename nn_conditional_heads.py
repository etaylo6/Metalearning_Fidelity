"""
ACCURACY SURROGATE MODEL: PyTorch Neural Network with Conditional Prediction Heads
=================================================================================

This script implements a Conditional Neural Network that uses separate prediction heads
for each feature combination. This architecture models how parameter importance changes
based on which physics features are enabled, allowing each feature combination to have
its own specialized prediction pathway.

Architecture Description:
- **Model Type**: Feedforward Neural Network with conditional prediction heads
- **Input Features**: Beam parameters (P, L, I, E, G, A, κ, T) - feature flags are used
  to route to the appropriate prediction head, not as input features.
- **Shared Feature Extraction**: All inputs pass through shared hidden layers that
  extract common features regardless of feature combination.
- **Conditional Prediction Heads**: Each feature combination (e.g., [1,1], [1,0], [0,1], [0,0])
  has its own dedicated prediction head (set of hidden layers + output layer). This allows
  the network to learn different parameter-accuracy relationships for different model
  fidelity levels.
- **Routing Mechanism**: Based on the feature flags, each sample is routed to the
  appropriate prediction head for forward pass.
- **Output Layer**: Each head produces a single output with Sigmoid activation to bound
  predictions between 0 and 1 (representing accuracy).
- **Optimizer**: Adam optimizer.
- **Regularization**: Dropout layers and early stopping.
- **Advantages**: Explicitly models how parameter importance changes with feature inclusion,
  potentially leading to more accurate predictions when different physics models have
  significantly different parameter sensitivities. Each head can specialize for its
  specific feature combination.
- **Disadvantages**: Requires more parameters than shared-head architectures, and may
  require more data to train effectively. The number of heads grows with the number of
  feature combinations.

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
from beam_config import TRAINING_BOUNDS, EXTRAPOLATION_BOUNDS, DATA_CONFIG, MODEL_CONFIG, CNN_CONDITIONAL_HEADS_CONFIG
from data_generator import generate_samples, calculate_combinatorial_accuracy, prepare_accuracy_data
from model_evaluation import plot_combinatorial_accuracy_surrogate_performance

# Consolidate configuration
CONFIG = {**DATA_CONFIG, **CNN_CONDITIONAL_HEADS_CONFIG}

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class ConditionalNeuralNetwork(nn.Module):
    """
    Conditional Neural Network with separate prediction heads for each feature combination.
    
    This architecture uses shared feature extraction layers followed by separate
    prediction heads for each feature combination, allowing the model to learn
    different parameter-accuracy relationships for different model fidelity levels.
    """
    def __init__(self, input_dim, feature_combinations, shared_layers, head_layers, dropout=0.2):
        super(ConditionalNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.feature_combinations = feature_combinations
        self.num_combinations = len(feature_combinations)
        
        # Shared feature extraction layers
        self.shared_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for layer_size in shared_layers:
            self.shared_layers.append(nn.Linear(prev_dim, layer_size))
            prev_dim = layer_size
        
        # Individual prediction heads for each feature combination
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_combinations):
            head = nn.ModuleList()
            head_prev_dim = prev_dim
            
            for head_layer_size in head_layers:
                head.append(nn.Linear(head_prev_dim, head_layer_size))
                head_prev_dim = head_layer_size
            
            # Final output layer (single output for accuracy)
            head.append(nn.Linear(head_prev_dim, 1))
            self.prediction_heads.append(head)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x, feature_flags):
        """
        Forward pass through conditional neural network.
        
        Args:
            x: Input parameters [batch_size, input_dim]
            feature_flags: Feature combination flags [batch_size, 2]
        
        Returns:
            Predictions for each sample using appropriate head
        """
        batch_size = x.size(0)
        
        # Shared feature extraction
        shared_features = x
        for layer in self.shared_layers:
            shared_features = self.activation(layer(shared_features))
            shared_features = self.dropout(shared_features)
        
        # Route to appropriate prediction head based on feature flags
        predictions = torch.zeros(batch_size, device=x.device)
        
        for i in range(batch_size):
            # Find which combination this sample belongs to
            combo_idx = self._get_combination_index(feature_flags[i])
            
            # Forward through appropriate head
            head_features = shared_features[i]
            head_layers = self.prediction_heads[combo_idx]
            
            for layer in head_layers[:-1]:  # All but final layer
                head_features = self.activation(layer(head_features))
                head_features = self.dropout(head_features)
            
            # Final output (no activation for regression, sigmoid applied at end)
            predictions[i] = head_layers[-1](head_features).squeeze()
        
        # Apply sigmoid to bound outputs to [0,1]
        return torch.sigmoid(predictions)
    
    def _get_combination_index(self, feature_flag):
        """Get the index of the feature combination."""
        flag_list = feature_flag.cpu().numpy().tolist()
        for i, combo in enumerate(self.feature_combinations):
            # Handle both list and numpy array comparisons
            combo_list = combo.tolist() if hasattr(combo, 'tolist') else combo
            if flag_list == combo_list:
                return i
        raise ValueError(f"Unknown feature combination: {flag_list}")

def fit_conditional_neural_network(X, y, feature_combinations):
    """
    Fit a Conditional Neural Network for accuracy prediction.
    
    Args:
        X: Input features (n_samples, n_features) - includes feature flags in last 2 columns
        y: Target values (accuracy scores in [0, 1])
        feature_combinations: List of feature combination vectors
    
    Returns:
        CNNWrapper object with predict() method interface
    """
    # Set random seeds for reproducibility
    torch.manual_seed(CONFIG['cnn_random_state'])
    np.random.seed(CONFIG['cnn_random_state'])
    
    # Separate input parameters from feature flags
    X_params = X[:, :-2]  # All but last 2 columns (feature flags)
    X_flags = X[:, -2:]   # Last 2 columns (feature flags)
    
    # Normalize input parameters
    scaler = StandardScaler()
    X_params_scaled = scaler.fit_transform(X_params)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_params_scaled)
    flags_tensor = torch.FloatTensor(X_flags)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, flags_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=CONFIG['cnn_batch_size'], shuffle=True)
    
    # Initialize model
    input_dim = X_params_scaled.shape[1]
    model = ConditionalNeuralNetwork(
        input_dim=input_dim,
        feature_combinations=feature_combinations,
        shared_layers=CONFIG['cnn_shared_layers'],
        head_layers=CONFIG['cnn_head_layers'],
        dropout=CONFIG['cnn_dropout']
    )
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['cnn_learning_rate'])
    
    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(CONFIG['cnn_epochs']):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_flags, batch_y in dataloader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch_X, batch_flags)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        
        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['cnn_epochs']}, Loss: {avg_loss:.6f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['cnn_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Create wrapper for sklearn-compatible interface
    class CNNWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
            self.model.eval()  # Set to evaluation mode
        
        def predict(self, X, return_std=False):
            # Separate parameters and flags
            X_params = X[:, :-2]
            X_flags = X[:, -2:]
            
            # Scale parameters
            X_params_scaled = self.scaler.transform(X_params)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_params_scaled)
            flags_tensor = torch.FloatTensor(X_flags)
            
            # Predict
            with torch.no_grad():
                predictions = self.model(X_tensor, flags_tensor)
                predictions_np = predictions.numpy()
            
            if return_std:
                # Conditional NN doesn't provide uncertainty, return zeros
                return predictions_np, np.zeros(len(X))
            else:
                return predictions_np
    
    print(f"Conditional NN training completed. Best loss: {best_loss:.6f}")
    return CNNWrapper(model, scaler)

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    """Run accuracy surrogate analysis"""
    print("="*60)
    print("BEAM MODEL ACCURACY SURROGATE ANALYSIS - CONDITIONAL NEURAL NETWORK")
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
    
    # Extract unique feature combinations for the conditional network
    feature_combinations = []
    for combo_name, results in combinatorial_results.items():
        combo_vector = results['combination']
        # Convert to list for comparison if it's a numpy array
        combo_list = combo_vector.tolist() if hasattr(combo_vector, 'tolist') else combo_vector
        if combo_list not in [combo.tolist() if hasattr(combo, 'tolist') else combo for combo in feature_combinations]:
            feature_combinations.append(combo_vector)
    
    # Fit combinatorial accuracy surrogate (Conditional Neural Network)
    print(f"Training Conditional Neural Network with {len(feature_combinations)} feature combinations:")
    for i, combo in enumerate(feature_combinations):
        combo_display = combo.tolist() if hasattr(combo, 'tolist') else combo
        print(f"  Head {i}: {combo_display}")
    
    conditional_nn = fit_conditional_neural_network(X_combined, y_combined, feature_combinations)
    
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
        conditional_nn, X_combined, y_combined, combination_labels, X_test, y_test_results,
        model_name="Conditional Neural Network"
    )
    
    print("\nAccuracy analysis complete!")
    print("Architecture: Conditional neural network with separate prediction heads")
    print(f"Number of prediction heads: {len(feature_combinations)}")
    print(f"Shared layers: {CONFIG['cnn_shared_layers']}")
    print(f"Head layers: {CONFIG['cnn_head_layers']}")
    print("\nAdvantages:")
    print("  - Explicitly models how parameter importance changes with feature inclusion")
    print("  - Each head can specialize for its specific feature combination")
    print("  - Potentially more accurate when different physics models have different sensitivities")
    print("\nDisadvantages:")
    print("  - Requires more parameters than shared-head architectures")
    print("  - May require more data to train effectively")

if __name__ == "__main__":
    main()

