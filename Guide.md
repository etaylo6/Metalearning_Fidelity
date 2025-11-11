# Model Guide: Beam Fidelity Evaluation Surrogate Models

## Overview

This directory contains multiple surrogate modeling approaches for predicting beam model accuracy and parameter importance. All models share the same data generation pipeline and physics calculations, ensuring consistent comparisons and reproducible results.

## Table of Contents

1. [Accuracy Surrogate Models](#accuracy-surrogate-models)
2. [Importance Surrogate Models](#importance-surrogate-models)
3. [Analysis Tools](#analysis-tools)
4. [Shared Infrastructure](#shared-infrastructure)
5. [Model Comparison Matrix](#model-comparison-matrix)
6. [Dependencies](#dependencies)
7. [Quick Start Guide](#quick-start-guide)
8. [Configuration](#configuration)
9. [Output and Results](#output-and-results)
10. [Adapting for Different Systems](#adapting-for-different-systems)

---

## Accuracy Surrogate Models

These models predict the **accuracy** (1 - relative error) of different beam model fidelity levels compared to a reference model. All accuracy models output values in the range [0, 1], where 1.0 represents perfect accuracy.

### 1. GAM² with Tensor Product Interactions
**File**: `gam2_tensor_interactions.py`

**What it does**: Uses a Generalized Additive Model (GAM) with tensor product interactions to predict accuracy. This is a statistical, interpretable model that captures non-linear relationships and interactions between parameters.

**Key Features**:
- Interpretable component functions (you can see how each parameter affects accuracy)
- Handles interactions between parameters automatically
- Different treatment for continuous parameters (smooth splines) vs binary features (linear terms)
- Regularization tuning to prevent overfitting
- Provides uncertainty estimates (via GAM framework)

**Architecture**:
- Main effects: Smooth splines for continuous parameters, linear terms for binary features
- Interactions: Tensor product interactions selected based on importance
- Output: Bounded [0, 1] predictions via clipping

**Advantages**:
- Highly interpretable
- Can capture complex interactions
- Robust to overfitting with proper regularization
- Good for exploratory analysis

**Disadvantages**:
- Can be slower than neural networks for large datasets
- Requires tuning of interaction terms
- Less flexible than deep neural networks for very complex relationships

**Dependencies**: `pygam`, `numpy`, `matplotlib`, `pandas`

---

### 2. Simple Neural Network with Feature Vector
**File**: `nn_simple_feature_vector.py`

**What it does**: A standard feedforward neural network that treats feature flags (temperature_enabled, shear_enabled) as regular input features alongside physical parameters. This is the simplest neural network approach.

**Key Features**:
- Simple architecture: all inputs treated equally
- No special handling of feature flags
- Standard MLP with ReLU activations and dropout
- Early stopping for regularization

**Architecture**:
- Input: [P, L, I, E, G, A, κ, T, temp_enabled, shear_enabled]
- Hidden layers: [64, 32, 32, 16] (configurable)
- Output: Single neuron with Sigmoid activation

**Advantages**:
- Simple to understand and implement
- Fast training and prediction
- Good baseline performance
- No assumptions about parameter relationships

**Disadvantages**:
- Must learn all relationships from data
- May require more data than physics-informed models
- Less interpretable than GAM models
- Doesn't explicitly model physics constraints

**Dependencies**: `torch`, `numpy`, `scikit-learn`, `matplotlib`

---

### 3. Input-Gated Neural Network
**File**: `nn_input_gated.py`

**What it does**: A neural network that applies physics-informed gating at the input layer. When temperature is disabled, the temperature parameter (T) is suppressed. When shear is disabled, shear-related parameters (G, A, κ) are suppressed.

**Key Features**:
- Physics-informed: explicitly models which parameters are relevant for which features
- Input-level gating: applies gating before network processing
- Configurable gate strength (0.0 = no gate, 1.0 = complete suppression)
- Shared hidden layers after gating

**Architecture**:
- Input gating: Masks parameters based on feature flags
- Shared layers: Process gated inputs through hidden layers
- Output: Single neuron with Sigmoid activation

**Gating Logic**:
- `temp_enabled=0`: Suppresses T parameter
- `shear_enabled=0`: Suppresses G, A, κ parameters

**Advantages**:
- Incorporates physics knowledge
- Can generalize better with less data
- More interpretable than simple NN (you know which parameters are gated)
- Efficient (gating applied once at input)

**Disadvantages**:
- Requires explicit knowledge of parameter-feature relationships
- Gating might be too rigid if relationships are more nuanced
- Less flexible than learned feature representations

**Dependencies**: `torch`, `numpy`, `scikit-learn`, `matplotlib`

---

### 4. Intermediate-Layer Gated Neural Network
**File**: `nn_intermediate_gated.py`

**What it does**: A neural network that applies gating in an intermediate hidden layer rather than at the input. The network first learns feature representations, then applies physics-informed gating to those representations.

**Key Features**:
- Two-stage processing: feature extraction → gating → prediction
- Learns non-linear feature representations before gating
- Gating applied to learned features, not raw parameters
- More flexible than input-level gating

**Architecture**:
- Pre-gate layers: Extract features from raw inputs
- Gated layer: Apply physics-informed gating to features
- Post-gate layers: Process gated features to prediction

**Advantages**:
- Learns non-linear feature representations
- More flexible than input-level gating
- Doesn't require explicit parameter-feature mapping
- Can capture complex relationships

**Disadvantages**:
- More complex architecture
- Less interpretable (gating on learned features, not raw parameters)
- Requires more hyperparameter tuning
- May be harder to debug

**Dependencies**: `torch`, `numpy`, `scikit-learn`, `matplotlib`

---

### 5. Activation-Suppressed and Gated Neural Network
**File**: `nn_activation_gated.py`

**What it does**: A two-stage neural network that first applies adaptive activation suppression (suppresses low-impact inputs), then applies physics-informed gating. This combines adaptive feature selection with physics-based constraints.

**Key Features**:
- Two-stage process: activation suppression → physics-based gating
- Adaptive suppression: learns which features are important
- Physics-informed gating: applies constraints based on feature flags
- Most sophisticated architecture

**Architecture**:
- Pre-activation layers: Extract initial features
- Activation layer: Suppress low-impact features based on magnitude
- Gated layer: Apply physics-informed gating
- Post-gate layers: Process to prediction

**Advantages**:
- Combines adaptive and physics-informed selection
- Most flexible architecture
- Can handle complex feature interactions
- Potentially best performance with sufficient data

**Disadvantages**:
- Most complex architecture
- Requires most hyperparameter tuning
- Hardest to interpret
- May overfit with insufficient data
- Slowest to train

**Dependencies**: `torch`, `numpy`, `scikit-learn`, `matplotlib`

---

### 6. Conditional Neural Network with Separate Heads
**File**: `nn_conditional_heads.py`

**What it does**: A neural network with separate prediction heads for each feature combination. Each feature combination (e.g., [1,1], [1,0], [0,1], [0,0]) gets its own specialized prediction pathway, allowing the model to learn different parameter-accuracy relationships for different fidelity levels.

**Key Features**:
- Separate heads: Each feature combination has its own prediction head
- Shared feature extraction: Common features extracted before head-specific processing
- Explicit modeling: Each head specializes for its feature combination
- Routing mechanism: Samples routed to appropriate head based on feature flags

**Architecture**:
- Shared layers: Extract common features from parameters
- Conditional heads: Separate prediction pathways for each combination
- Routing: Feature flags determine which head to use

**Advantages**:
- Explicitly models combination-specific relationships
- Each head can specialize for its combination
- More accurate when combinations have very different sensitivities
- Interpretable (you can analyze each head separately)

**Disadvantages**:
- Requires more parameters (one head per combination)
- Needs more data to train effectively
- Number of heads grows with number of combinations
- More complex than shared-head architectures

**Dependencies**: `torch`, `numpy`, `scikit-learn`, `matplotlib`

---

## Importance Surrogate Models

These models predict **parameter importance** (local Sobol sensitivity) rather than accuracy. They answer the question: "How sensitive is the output to changes in each parameter?"

### 7. Gaussian Process Regression for Parameter Importance
**File**: `importance_surrogate_gaussian_process.py`

**What it does**: Uses Gaussian Process Regression (GPR) to predict local Sobol importance (sensitivity) for each parameter. Calculates how much a relative change in each parameter affects the output.

**Key Features**:
- Predicts parameter importance, not accuracy
- Provides uncertainty quantification (predictive standard deviation)
- Separate GPR model for each parameter
- Bayesian approach with natural uncertainty handling

**Architecture**:
- Input: Beam parameters [P, L, I, E, G, A, κ, T]
- Output: Local Sobol importance for each parameter
- Kernel: Configurable (RBF, Matern, with optional WhiteKernel)
- Multi-output: One GPR model per parameter

**Advantages**:
- Provides uncertainty quantification
- Good for small datasets
- Handles non-linear relationships
- Bayesian approach with natural uncertainty

**Disadvantages**:
- Computationally expensive (O(n³) scaling)
- Slow for large datasets (>1000 samples)
- Requires kernel hyperparameter tuning
- Separate model per parameter increases complexity

**Dependencies**: `scikit-learn`, `numpy`, `matplotlib`

---

## Analysis Tools

### 8. Graph Edit Importance Analysis
**File**: `graph_edit_importance.py`

**What it does**: Creates and visualizes a bipartite graph representation of the beam model system. Shows relationships between variables and model components, useful for understanding system structure and dependencies.

**Key Features**:
- Bipartite graph: Variables and model components as node types
- Visualization: Shows system structure and dependencies
- Subgraph analysis: Analyzes different model combinations
- Structural importance: Understands which components connect which variables

**Note**: This is a visualization/analysis tool, not a predictive model.

**Dependencies**: `networkx`, `numpy`, `matplotlib`

---

## Shared Infrastructure

All models use the same shared infrastructure for consistency and reproducibility:

### `beam_physics.py`
- Contains beam displacement calculation functions
- `beam_displacement_timoshenko()`: Full Timoshenko-Ehrenfest model
- `beam_displacement_euler_bernoulli()`: Euler-Bernoulli model
- `beam_displacement_with_features()`: Flexible model with feature flags

### `data_generator.py`
- `generate_samples()`: Generates parameter samples using various strategies (LHS, Halton, Halton with boundary bias)
- `calculate_combinatorial_accuracy()`: Calculates accuracy for all feature combinations
- `prepare_accuracy_data()`: Prepares data for surrogate model training

### `beam_config.py`
- `TRAINING_BOUNDS`: Parameter bounds for training data
- `EXTRAPOLATION_BOUNDS`: Parameter bounds for test data
- `DATA_CONFIG`: Data generation configuration (sampling method, number of samples, etc.)
- `MODEL_CONFIG`: Model feature combinations and names
- Model-specific configurations (hyperparameters for each model)

---

## Model Comparison Matrix

| Model | Type | Interpretability | Complexity | Training Speed | Data Requirements | Physics-Informed | Uncertainty |
|-------|------|------------------|------------|----------------|-------------------|------------------|-------------|
| GAM² | Statistical | ⭐⭐⭐⭐⭐ | Medium | Medium | Medium | Partial | Yes |
| Simple NN | Neural Network | ⭐⭐ | Low | Fast | High | No | No |
| Input-Gated NN | Neural Network | ⭐⭐⭐ | Medium | Fast | Medium | Yes | No |
| Intermediate-Gated NN | Neural Network | ⭐⭐ | High | Medium | Medium | Partial | No |
| Activation-Gated NN | Neural Network | ⭐ | Very High | Slow | High | Partial | No |
| Conditional NN | Neural Network | ⭐⭐⭐ | High | Medium | High | Explicit | No |
| GP Importance | Statistical | ⭐⭐⭐⭐ | Medium | Slow | Low | No | Yes |

**Legend**:
- ⭐⭐⭐⭐⭐ = Excellent
- ⭐⭐⭐⭐ = Very Good
- ⭐⭐⭐ = Good
- ⭐⭐ = Fair
- ⭐ = Poor

---

## Dependencies

All dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### Core Dependencies
- `numpy`: Numerical computations
- `scipy`: Scientific computing
- `matplotlib`: Plotting and visualization
- `pandas`: Data manipulation (used by GAM² for logging)

### Machine Learning
- `scikit-learn`: Machine learning utilities (used by neural networks and GP)
- `torch`: PyTorch for neural network models

### Statistical Modeling
- `pygam`: Generalized Additive Models (used by GAM²)

### Sampling
- `pyDOE`: Design of experiments and Latin Hypercube Sampling

### Graph Analysis
- `networkx`: Graph analysis and visualization

---

## Quick Start Guide

### Prerequisites

Install all dependencies:
```bash
pip install -r requirements.txt
```

### Running an Accuracy Surrogate Model

1. **Choose a model** based on the comparison matrix and model descriptions above

2. **Run the model**:
   ```bash
   python gam2_tensor_interactions.py
   # or
   python nn_simple_feature_vector.py
   # or
   python nn_input_gated.py
   # etc.
   ```

3. **View results**:
   - Performance plots (training vs extrapolation) displayed automatically
   - Summary statistics printed to console (R² scores, MSE, etc.)
   - For GAM²: CSV logs saved in `logs/` directory with timestamp

### Running Parameter Importance Analysis

```bash
python importance_surrogate_gaussian_process.py
```

This will:
- Calculate local Sobol importance for each parameter
- Train GPR models for each parameter
- Generate plots showing importance predictions with uncertainty
- Print summary statistics

### Running Graph Analysis

```bash
python graph_edit_importance.py
```

This will:
- Create bipartite graph representation
- Visualize system structure
- Generate subgraphs for different model combinations
- Show variable-component dependencies

---

## Configuration

All models use centralized configuration in `beam_config.py`:

### Data Configuration
- `n_samples`: Number of training samples (default: 5000)
- `n_test_samples`: Number of test samples (default: 200)
- `sampling_method`: 'lhs', 'halton', or 'halton_boundary'
- `boundary_bias`: Fraction of samples near boundaries
- `random_seed`: Random seed for reproducibility

### Model Configuration
- `MODEL_CONFIG`: Feature combinations and names
- Model-specific hyperparameters (see each model file for details)

### Parameter Bounds
- `TRAINING_BOUNDS`: Parameter ranges for training data
- `EXTRAPOLATION_BOUNDS`: Parameter ranges for test data (wider range)

To modify configuration, edit `beam_config.py` before running any model.

---

## Output and Results

### Accuracy Models
- **Plots**: Training vs extrapolation performance for each feature combination
- **Statistics**: R² scores, MSE, accuracy ranges
- **Predictions**: Accuracy scores in [0, 1] range
- **Logs** (GAM² only): CSV files with training/testing data and predictions

### Importance Models
- **Plots**: Importance predictions with uncertainty bars
- **Statistics**: R² scores for each parameter's importance prediction
- **Uncertainty**: Predictive standard deviation for each prediction

### Graph Analysis
- **Graphs**: Bipartite graph visualizations
- **Subgraphs**: Graphs for different model combinations
- **Metrics**: Graph density, node counts, edge counts

---

## Adapting for Different Systems

To use these models with a different physical system:

1. **Update `beam_physics.py`**:
   - Replace beam model functions with your physics model
   - Update parameter order and names
   - Modify feature vector structure if needed

2. **Update `beam_config.py`**:
   - Replace `TRAINING_BOUNDS` with your parameter bounds
   - Update `EXTRAPOLATION_BOUNDS` for test data
   - Modify `MODEL_CONFIG` for your feature combinations

3. **Update `data_generator.py`** (if needed):
   - Modify `calculate_combinatorial_accuracy()` if parameter count changes
   - Update parameter unpacking to match your model signature

4. **Update model files** (if needed):
   - Modify visualization functions if parameter names change
   - Update logging functions if parameter structure changes
   - Adjust hyperparameters in model-specific configs

---

## Best Practices

1. **Start simple**: Begin with GAM² or Simple NN to establish a baseline
2. **Compare models**: Run multiple models and compare performance
3. **Check extrapolation**: Always evaluate on extrapolation data to test generalization
4. **Tune hyperparameters**: Adjust model-specific hyperparameters for your data
5. **Visualize results**: Use the built-in plotting functions to understand model behavior
6. **Use consistent data**: All models use the same data generation, ensuring fair comparison

---

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Memory errors**: Reduce `n_samples` in `beam_config.py`
3. **Slow training**: Reduce model complexity or number of samples
4. **Poor performance**: Try a different model or tune hyperparameters
5. **Data generation errors**: Check parameter bounds in `beam_config.py`

### Getting Help

- Check model docstrings for detailed architecture descriptions
- Review `beam_config.py` for configuration options
- Examine example outputs in `logs/` directory (for GAM²)
- Compare multiple models to understand differences

---

## Summary

This directory provides a comprehensive suite of surrogate modeling approaches for beam fidelity evaluation. Each model offers different capabilities:

- **GAM²**: Highly interpretable statistical model with interaction terms
- **Simple NN**: Standard neural network baseline
- **Input-Gated NN**: Physics-informed neural network with input-level gating
- **Intermediate-Gated NN**: Neural network with intermediate-layer gating
- **Activation-Gated NN**: Advanced neural network with activation suppression and gating
- **Conditional NN**: Neural network with separate prediction heads per combination
- **GP Importance**: Gaussian Process model for parameter sensitivity analysis
- **Graph Analysis**: Visualization tool for system structure analysis

All models share the same data generation and physics calculations, ensuring consistent, reproducible comparisons. Refer to the comparison matrix and individual model descriptions to understand their differences and select the appropriate model for your needs.
