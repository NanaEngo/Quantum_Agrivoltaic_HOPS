"""
Hamiltonian Factory for Quantum Agrivoltaics.

This module provides functions to create system Hamiltonians, specifically the
Fenna-Matthews-Olsen (FMO) complex, and helper functions for spectral densities.
"""

import numpy as np

def create_fmo_hamiltonian(include_reaction_center=False):
    """
    Create the FMO Hamiltonian matrix based on standard parameters from the literature.
    
    Mathematical Framework:
    The Fenna-Matthews-Olsen (FMO) complex is modeled as an excitonic system
    with the Hamiltonian:
    
    H_FMO = Σᵢ εᵢ |i⟩⟨i| + Σᵢⱼ Jᵢⱼ |i⟩⟨j|
    
    Parameters:
    include_reaction_center (bool): Whether to include the reaction center state
    
    Returns:
    H (2D array): Hamiltonian matrix in units of cm^-1
    site_energies (1D array): Site energies in cm^-1
    """
    if include_reaction_center:
        # Include 8 sites with reaction center
        site_energies = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440, 11700])  # Last is RC
    else:
        # Standard 7-site FMO complex
        site_energies = np.array([12410, 12530, 12210, 12320, 12480, 12630, 12440])
    
    n_sites = len(site_energies)
    H = np.zeros((n_sites, n_sites))
    
    # Set diagonal elements (site energies)
    np.fill_diagonal(H, site_energies)
    
    # Off-diagonal elements (couplings) - symmetric matrix
    couplings = {
        (0, 1): -87.7, (0, 2): 5.5, (0, 3): -5.9, (0, 4): 6.7, (0, 5): -13.7, (0, 6): -9.9,
        (1, 2): 30.8, (1, 3): 8.2, (1, 4): 0.7, (1, 5): 11.8, (1, 6): 4.3,
        (2, 3): -53.5, (2, 4): -2.2, (2, 5): -9.6, (2, 6): 6.0,
        (3, 4): -70.7, (3, 5): -17.0, (3, 6): -63.3,
        (4, 5): 81.1, (4, 6): -1.3,
        (5, 6): 39.7
    }
    
    for (i, j), value in couplings.items():
        if i < n_sites and j < n_sites:
            H[i, j] = value
            H[j, i] = value  # Ensure Hermitian
    
    return H, site_energies

def spectral_density_drude_lorentz(omega, lambda_reorg, gamma, temperature):
    """Calculate Drude-Lorentz spectral density."""
    kT = 0.695 * temperature
    J = 2 * lambda_reorg * gamma * omega / (omega**2 + gamma**2)
    n_th = 1.0 / (np.exp(np.maximum(omega, 1e-10) / kT) - 1)
    J *= (1 + n_th) if np.any(omega >= 0) else n_th - 1
    return J
