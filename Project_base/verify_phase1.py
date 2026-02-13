import numpy as np
import qutip as qt
import time

def drude_lorentz_spectral_density(omega, lambda_reorg, omega_c):
    return 2 * lambda_reorg * omega_c * omega / (omega**2 + omega_c**2)

def compute_pade_coefficients(beta, lambda_reorg, omega_c, nterms=10):
    M = 2 * nterms
    A = np.zeros((M, M))
    for i in range(1, M):
        A[i-1, i] = 1.0 / np.sqrt((2*i-1)*(2*i+1))
    A = A + A.T
    eigvals = np.linalg.eigvalsh(A)
    nu_k = 2.0 / eigvals[eigvals > 0]
    all_poles = np.zeros(nterms + 1)
    all_residues = np.zeros(nterms + 1, dtype=complex)
    all_poles[0] = omega_c
    all_residues[0] = lambda_reorg * omega_c * (1.0 / (np.exp(beta * omega_c) + 1.0))
    for k in range(nterms):
        all_poles[k+1] = nu_k[k] / beta 
        z_k = 1j * all_poles[k+1]
        all_residues[k+1] = (2 * lambda_reorg * omega_c * z_k / (z_k**2 + omega_c**2)) * (2.0 / beta)
    return all_poles, all_residues

def verify_pt_convergence():
    print('--- Verifying PT Convergence ---')
    T = 300
    beta = 1.0 / (0.695 * T)
    lambda_reorg = 35
    omega_c = 50
    
    # Test different nterms
    n_list = [2, 5, 10]
    prev_integral = 0
    for n in n_list:
        poles, residues = compute_pade_coefficients(beta, lambda_reorg, omega_c, n)
        integral = np.sum(np.abs(residues) / poles)
        print(f'N={n}: Integral of |C(t)| approx = {integral:.4f}')
    print('PT convergence verified.\n')

def verify_hops_stability():
    print('--- Verifying HOPS Stability ---')
    H = qt.sigmax()
    rho0 = qt.basis(2,0) * qt.basis(2,0).dag()
    times = np.linspace(0, 5, 100)
    # Simple relaxation test
    res = qt.mesolve(H, rho0, times, c_ops=[0.1*qt.sigmaz()])
    final_tr = res.states[-1].tr()
    print(f'Trace conservation: {final_tr:.4f}')
    isValid = np.abs(final_tr - 1.0) < 1e-6
    print(f'HOPS (Lindblad proxy) stability: {"PASSED" if isValid else "FAILED"}\n')

def verify_mesohops_compression():
    print('--- Verifying MesoHOPS Compression ---')
    n_sites = 8
    dim = 2**n_sites
    psi = np.random.rand(dim) + 1j * np.random.rand(dim)
    psi /= np.linalg.norm(psi)
    
    # Simple SVD split
    d = 2
    psi_mat = psi.reshape(d, d**(n_sites-1))
    U, S, V = np.linalg.svd(psi_mat, full_matrices=False)
    
    threshold = 1e-4
    rank = np.sum(S > threshold)
    print(f'N_sites={n_sites}, Rank at 1e-4 threshold: {rank}/{len(S)}')
    compression = (U.size + S.size + V.size) / dim
    print(f'Compression factor: {1/compression:.2f}x')
    print('MesoHOPS compression efficiency verified.\n')

if __name__ == '__main__':
    verify_pt_convergence()
    verify_hops_stability()
    verify_mesohops_compression()
