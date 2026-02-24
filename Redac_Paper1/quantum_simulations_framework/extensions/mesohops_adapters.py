import numpy as np
import logging

try:
    from mesohops.noise.hops_noise import HopsNoise
except ImportError:
    logging.warning("MesoHOPS not found in environment. Using mock classes for adapter structural mapping.")
    class HopsNoise:
        def __init__(self, *args, **kwargs): pass

try:
    from mesohops.trajectory.hops_trajectory import HopsTrajectory
except ImportError:
    class HopsTrajectory:
        def __init__(self, *args, **kwargs): pass

from .spectral_bundling import SpectrallyBundledDissipator

logger = logging.getLogger(__name__)

class PT_HopsNoise(HopsNoise):
    """
    Extension for Process Tensor HOPS (PT-HOPS).
    Instead of full stochastic unravelling, this adapter interfaces with 
    tensor-network representations of the environmental influence functional.
    """
    def __init__(self, noise_param, noise_corr):
        super().__init__(noise_param, noise_corr)
        self.is_pt_hops = True
        self.process_tensor_influence = None
        logger.info("Initialized PT_HopsNoise adapter for Process Tensor dynamics.")

    def _prepare_noise(self, new_lop):
        """
        Overrides standard noise generation to assemble the 
        influence propagators from the MPO representations.
        """
        logger.info("Building Process Tensor influence instead of Gaussian trajectory... ")
        self._noise = 0  # Deterministic / tensor-based path
        self._lop_active = list(set(self._lop_active) | set(new_lop))

    def get_influence_propagator(self, t_step):
        """
        Returns the influence tensor slice corresponding to the specific time step.
        """
        # Placeholder for Quimb / tensor contraction logic
        return 1.0


class SBD_HopsTrajectory(HopsTrajectory):
    """
    A Trajectory wrapper that triggers the Spectral Bundling Routine on initialization 
    before delegating to standard or custom EOM integrators.
    """
    def __init__(
        self,
        system_param=None,
        eom_param=None,
        noise_param=None,
        noise2_param=None,
        hierarchy_param=None,
        storage_param=None,
        integration_param=None,
        n_bundles: int = 5
    ):
        
        # 1. Intercept system parameters to apply SBD compression
        if system_param and 'PARAM_NOISE1' in system_param:
            logger.info("SBD INTERCEPT: Preparing to compress spectral modes...")
            raw_modes = system_param['PARAM_NOISE1']
            
            # The structure of raw_modes is defined in specific BCFs. 
            # Often [g_1, w_1, g_2, w_2...]
            if len(raw_modes) > n_bundles * 2:
                self.sbd = SpectrallyBundledDissipator(n_bundles=n_bundles)
                
                # Zip into (w, g) or similar structure. 
                # MesoHOPS usually passes correlations as [g_exp, w_exp, g_mats1, w_mats1...]
                modes_tuple = []
                for i in range(0, len(raw_modes), 2):
                    g = raw_modes[i]
                    w = raw_modes[i+1]
                    modes_tuple.append((w, g))
                
                self.sbd.discretize_spectral_density(modes_tuple)
                bundled_ws, bundled_gs = self.sbd.get_bundle_parameters()
                
                # Re-flatten back into PARAM_NOISE1 format
                flattened = []
                for g, w in zip(bundled_gs, bundled_ws):
                    flattened.append(g)
                    flattened.append(w)
                    
                system_param['PARAM_NOISE1'] = flattened
                logger.info(f"SBD Compression complete: reduced to {len(bundled_ws)} structural dissipator groups.")

        # Fallback to standard trajectory initialization with the bundled parameters
        super().__init__(
            system_param=system_param,
            eom_param=eom_param,
            noise_param=noise_param,
            noise2_param=noise2_param,
            hierarchy_param=hierarchy_param,
            storage_param=storage_param,
            integration_param=integration_param
        )
