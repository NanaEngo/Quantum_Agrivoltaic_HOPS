import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Any, List, Optional
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class Spectroscopy2DES:
    """
    Simulation of Two-Dimensional Electronic Spectroscopy (2DES) signals.
    
    Mathematical Framework:
    2DES signals S(ωτ, T, ωt) are calculated using the third-order response functions
    R^(3)(τ, T, t). The 2D spectrum at a fixed waiting time T is given by:
    
    S(ωτ, T, ωt) = Re ∫∫ dτ dt exp(iωτ τ + iωt t) Σ_i R_i(τ, T, t)
    
    This implementation uses a simplified excitonic model where peaks represent:
    1. Diagonal peaks: Absorption/Emission of individual sites
    2. Cross-peaks: Coupling and energy transfer between sites
    3. Oscillations: Coherent excitonic beats (if T is varied)
    """
    
    def __init__(self, system_size: int):
        self.system_size = system_size
        logger.info(f"Spectroscopy2DES initialized for {system_size} sites.")
        
    def simulate_2d_spectrum(self, 
                             hamiltonian: np.ndarray, 
                             waiting_time: float = 0.0,
                             resolution: int = 128,
                             linewidth: float = 50.0) -> Dict[str, Any]:
        """
        Generates a simplified 2D spectrum based on the excitonic Hamiltonian.
        
        Parameters:
        - hamiltonian: System Hamiltonian in cm^-1
        - waiting_time: Population time T in fs
        - resolution: Grid size for ωτ and ωt axes
        - linewidth: Gaussian broadening width in cm^-1
        """
        # Eigenvalues identify the main absorption peaks
        energies, states = np.linalg.eigh(hamiltonian)
        
        # Frequency axes (cm^-1)
        w_min = np.min(energies) - 500
        w_max = np.max(energies) + 500
        w_axis = np.linspace(w_min, w_max, resolution)
        
        W_tau, W_t = np.meshgrid(w_axis, w_axis)
        spectrum = np.zeros((resolution, resolution))
        
        # 1. Diagonal Peaks (Absorption and Stimulated Emission)
        for i, E in enumerate(energies):
            # Intensity modulated by a simplified decay
            intensity = 1.0 * np.exp(-waiting_time / 1000.0)
            spectrum += intensity * np.exp(-((W_tau - E)**2 + (W_t - E)**2) / (2 * linewidth**2))
            
        # 2. Cross-Peaks (Indicate coupling)
        # We use the off-diagonal elements of the Hamiltonian to scale cross-peak intensity
        for i in range(len(energies)):
            for j in range(len(energies)):
                if i == j: continue
                
                # Check coupling in the original site basis
                # Simplified: use average coupling strength from Hamiltonian
                coupling = np.abs(np.mean(np.diag(hamiltonian, k=1))) / 100.0
                
                # Cross-peak intensity grows if energy transfer occurs
                # Simplified: transfer ~ (1 - exp(-T/rate))
                transfer_intensity = coupling * (1 - np.exp(-waiting_time / 500.0))
                
                spectrum += transfer_intensity * np.exp(-((W_tau - energies[i])**2 + (W_t - energies[j])**2) / (2 * linewidth**2))
        
        return {
            'w_axis': w_axis,
            'spectrum': spectrum,
            'waiting_time': waiting_time,
            'eigenenergies': energies
        }
    
    def plot_2d_spectrum(self, 
                         results: Dict[str, Any], 
                         output_dir: str = '../Graphics/',
                         filename: str = 'spectroscopy_2des') -> str:
        """
        Plots the 2D spectrum as a contour plot.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 7))
        plt.contourf(results['w_axis'], results['w_axis'], results['spectrum'].T, 50, cmap='nipy_spectral')
        plt.colorbar(label='Signal Intensity (arb. units)')
        
        # Draw diagonal line
        plt.plot(results['w_axis'], results['w_axis'], 'w--', alpha=0.5)
        
        plt.xlabel(r'Excitation Frequency $\omega_{\tau}$ (cm$^{-1}$)')
        plt.ylabel(r'Detection Frequency $\omega_t$ (cm$^{-1}$)')
        plt.title(f'2D Electronic Spectrum (T = {results["waiting_time"]} fs)')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_filename = f"{filename}_{timestamp}.pdf"
        filepath = os.path.join(output_dir, full_filename)
        
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        
        logger.info(f"2DES plot saved to {filepath}")
        return filepath
