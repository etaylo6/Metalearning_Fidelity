"""
BEAM PHYSICS MODULE
===================

This module contains the physics models for beam displacement calculations.
It implements the Timoshenko-Ehrenfest and Euler-Bernoulli beam theories with
optional thermal effects.

The module is designed to be generic and can be adapted for different physical
systems by modifying the model functions and parameter definitions.

Key Functions:
- beam_displacement_timoshenko: Full Timoshenko-Ehrenfest model with thermal effects
- beam_displacement_euler_bernoulli: Euler-Bernoulli model with thermal effects
- beam_displacement_with_features: Flexible model that can enable/disable features

Parameter Structure:
The beam model expects parameters in the order: [P, L, I, E, G, A, κ, T]
- P: Applied force (N)
- L: Beam length (m)
- I: Moment of inertia (m^4)
- E: Young's modulus (Pa)
- G: Shear modulus (Pa)
- A: Cross-sectional area (m^2)
- κ: Shear correction factor
- T: Temperature (K)

Feature Vector:
- feature_vector: [temperature_enabled, shear_enabled]
  - [1, 1]: Full Timoshenko-Ehrenfest with temperature
  - [1, 0]: Euler-Bernoulli with temperature
  - [0, 1]: Timoshenko-Ehrenfest without temperature
  - [0, 0]: Euler-Bernoulli without temperature
"""

import numpy as np

# ============================================================================
# TEMPERATURE COEFFICIENTS
# ============================================================================

TEMP_COEFFS = {
    'T_ref': 293,            # Reference temperature (20°C in Kelvin)
    'alpha_E': -0.0005,      # Temperature coefficient for Young's modulus (/K)
    'alpha_G': -0.0005,      # Temperature coefficient for shear modulus (/K) 
    'alpha_T': 0.000012      # Thermal expansion coefficient (/K) for steel
}

# ============================================================================
# BEAM MODEL FUNCTIONS
# ============================================================================

def beam_displacement_timoshenko(P, L, I, E, G, A, κ, T):
    """
    Timoshenko-Ehrenfest beam tip displacement with thermal effects.
    
    This is the full physics model that includes:
    - Bending deformation (Euler-Bernoulli component)
    - Shear deformation (Timoshenko component)
    - Temperature-dependent material properties
    - Thermal expansion effects
    
    Args:
        P: Applied force (N)
        L: Beam length (m)
        I: Moment of inertia (m^4)
        E: Young's modulus (Pa)
        G: Shear modulus (Pa)
        A: Cross-sectional area (m^2)
        κ: Shear correction factor
        T: Temperature (K)
    
    Returns:
        Displacement at beam tip (m)
    """
    # Temperature-dependent moduli
    E_T = E * (1 + TEMP_COEFFS['alpha_E'] * (T - TEMP_COEFFS['T_ref']))
    G_T = G * (1 + TEMP_COEFFS['alpha_G'] * (T - TEMP_COEFFS['T_ref']))
    
    # Temperature-dependent length (thermal expansion)
    L_T = L * (1 + TEMP_COEFFS['alpha_T'] * (T - TEMP_COEFFS['T_ref']))
    
    # Timoshenko-Ehrenfest displacement
    bending = (P * L_T**3) / (3 * E_T * I)
    shear = (P * L_T) / (κ * G_T * A)
    return bending + shear

def beam_displacement_euler_bernoulli(P, L, I, E, T):
    """
    Euler-Bernoulli beam tip displacement with thermal effects.
    
    This model includes:
    - Bending deformation only (no shear)
    - Temperature-dependent material properties
    - Thermal expansion effects
    
    Args:
        P: Applied force (N)
        L: Beam length (m)
        I: Moment of inertia (m^4)
        E: Young's modulus (Pa)
        T: Temperature (K)
    
    Returns:
        Displacement at beam tip (m)
    """
    # Temperature-dependent modulus
    E_T = E * (1 + TEMP_COEFFS['alpha_E'] * (T - TEMP_COEFFS['T_ref']))
    
    # Temperature-dependent length (thermal expansion)
    L_T = L * (1 + TEMP_COEFFS['alpha_T'] * (T - TEMP_COEFFS['T_ref']))
    
    # Euler-Bernoulli displacement (bending only, no shear)
    displacement = (P * L_T**3) / (3 * E_T * I)
    return displacement

def beam_displacement_with_features(P, L, I, E, G, A, κ, T, feature_vector):
    """
    Calculate beam displacement with optional features controlled by feature_vector.
    
    This is a flexible wrapper that allows enabling/disabling specific physics
    features (temperature effects, shear effects) to model different fidelity levels.
    
    Args:
        P: Applied force (N)
        L: Beam length (m)
        I: Moment of inertia (m^4)
        E: Young's modulus (Pa)
        G: Shear modulus (Pa)
        A: Cross-sectional area (m^2)
        κ: Shear correction factor
        T: Temperature (K)
        feature_vector: [temperature_enabled, shear_enabled]
            - temperature_enabled: 1 to enable thermal effects, 0 to disable
            - shear_enabled: 1 to enable shear deformation, 0 to disable
    
    Returns:
        Displacement at beam tip (m)
    
    Feature Combinations:
        [1, 1] = Full Timoshenko-Ehrenfest with temperature
        [1, 0] = Euler-Bernoulli with temperature
        [0, 1] = Timoshenko-Ehrenfest without temperature
        [0, 0] = Euler-Bernoulli without temperature
    """
    temp_enabled, shear_enabled = feature_vector
    
    # Temperature effects
    if temp_enabled:
        # Use actual temperature
        E_eff = E * (1 + TEMP_COEFFS['alpha_E'] * (T - TEMP_COEFFS['T_ref']))
        G_eff = G * (1 + TEMP_COEFFS['alpha_G'] * (T - TEMP_COEFFS['T_ref']))
        L_eff = L * (1 + TEMP_COEFFS['alpha_T'] * (T - TEMP_COEFFS['T_ref']))
    else:
        # Use reference temperature (no thermal effects)
        E_eff = E * (1 + TEMP_COEFFS['alpha_E'] * (TEMP_COEFFS['T_ref'] - TEMP_COEFFS['T_ref']))
        G_eff = G * (1 + TEMP_COEFFS['alpha_G'] * (TEMP_COEFFS['T_ref'] - TEMP_COEFFS['T_ref']))
        L_eff = L * (1 + TEMP_COEFFS['alpha_T'] * (TEMP_COEFFS['T_ref'] - TEMP_COEFFS['T_ref']))
    
    # Shear effects
    if shear_enabled:
        # Timoshenko-Ehrenfest (bending + shear)
        bending = (P * L_eff**3) / (3 * E_eff * I)
        shear = (P * L_eff) / (κ * G_eff * A)
        return bending + shear
    else:
        # Euler-Bernoulli (bending only)
        return (P * L_eff**3) / (3 * E_eff * I)

