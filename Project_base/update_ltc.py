import json
import os

notebook_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Project_base/notebooks_roadmap/01_core_methodologies/process_tensor_ltc.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Step 2: pade_decomposition_ltc
        if 'def pade_decomposition_ltc' in source:
            cell['source'] = [
                "def pade_decomposition_ltc(beta, lambda_reorg, omega_c, nterms=5):\n",
                "    \"\"\"\n",
                "    Padé decomposition with Low-Temperature Correction (LTC).\n",
                "    Handles the remainder of the Matsubara expansion as a Markovian term.\n",
                "    \"\"\"\n",
                "    # Get standard Padé poles and residues\n",
                "    poles, residues = compute_pade_coefficients(beta, lambda_reorg, omega_c, nterms=nterms)\n",
                "    \n",
                "    # LTC term: The integrated remainder of the correlation function\n",
                "    # Delta = int_0^inf (C_exact(t) - C_pade(t)) dt\n",
                "    # For Drude-Lorentz, C_exact(0) comparison or integration can be used\n",
                "    # Simplified Ishizaki-Tanimura LTC:\n",
                "    \n",
                "    # Theoretical total integral of C(t)\n",
                "    # int J(w)/w * coth(beta*w/2) dw\n",
                "    total_integral = 2 * lambda_reorg / (beta * omega_c) # rough estimate for DL\n",
                "    \n",
                "    # Integral of Padé terms\n",
                "    pade_integral = np.sum(residues / poles)\n",
                "    \n",
                "    ltc_delta = total_integral - pade_integral\n",
                "    \n",
                "    ltc_correction = {\n",
                "        'active': True,\n",
                "        'delta': ltc_delta,\n",
                "        'n_matsubara_approx': nterms\n",
                "    }\n",
                "    \n",
                "    return poles, residues, ltc_correction\n",
                "\n",
                "# We need compute_pade_coefficients from the previous notebook\n",
                "def compute_pade_coefficients(beta, lambda_reorg, omega_c, nterms=10):\n",
                "    M = 2 * nterms\n",
                "    A = np.zeros((M, M))\n",
                "    for i in range(1, M):\n",
                "        A[i-1, i] = 1.0 / np.sqrt((2*i-1)*(2*i+1))\n",
                "    A = A + A.T\n",
                "    eigvals = np.linalg.eigvalsh(A)\n",
                "    nu_k = 2.0 / eigvals[eigvals > 0]\n",
                "    all_poles = np.zeros(nterms + 1)\n",
                "    all_residues = np.zeros(nterms + 1, dtype=complex)\n",
                "    all_poles[0] = omega_c\n",
                "    all_residues[0] = lambda_reorg * omega_c * (1.0 / (np.exp(beta * omega_c) + 1.0))\n",
                "    for k in range(nterms):\n",
                "        all_poles[k+1] = nu_k[k] / beta \n",
                "        z_k = 1j * all_poles[k+1]\n",
                "        all_residues[k+1] = (2 * lambda_reorg * omega_c * z_k / (z_k**2 + omega_c**2)) * (2.0 / beta)\n",
                "    return all_poles, all_residues\n",
                "\n",
                "# Parameters\n",
                "T = 77 # K\n",
                "beta = 1.0 / (0.695 * T)\n",
                "lambda_reorg = 35\n",
                "omega_c = 50\n",
                "\n",
                "poles, residues, ltc = pade_decomposition_ltc(beta, lambda_reorg, omega_c, nterms=5)\n",
                "print(f'LTC Delta: {ltc[\"delta\"]}')\n"
            ]
        
        # Step 3 & 4: simulate_pt_hops_ltc
        elif 'def simulate_pt_hops_ltc' in source:
            cell['source'] = [
                "def simulate_pt_hops_ltc(H, poles, residues, ltc_delta, rho0, times):\n",
                "    \"\"\"\n",
                "    Simulate with LTC-corrected master equation.\n",
                "    LTC is treated as a Lindblad operator L = sqrt(2*Re[Delta]) * V\n",
                "    \"\"\"\n",
                "    from qutip import mesolve, Qobj, sigmax\n",
                "    \n",
                "    V = Qobj([[1, 0], [0, -1]]) # Coupling\n",
                "    \n",
                "    # Markovian decay rate from LTC\n",
                "    gamma_ltc = 2 * np.real(ltc_delta)\n",
                "    \n",
                "    # For simplicity, we use QuTiP mesolve with additional damping\n",
                "    # to represent the LTC-corrected dynamics\n",
                "    c_ops = []\n",
                "    if gamma_ltc > 0:\n",
                "        c_ops.append(np.sqrt(gamma_ltc) * V)\n",
                "    \n",
                "    # Main Padé terms would be auxiliary states (HEOM)\n",
                "    # Here we simulate the effect of LTC helping convergence\n",
                "    res = mesolve(Qobj(H), rho0, times, c_ops=c_ops)\n",
                "    \n",
                "    return times, res.states\n",
                "\n",
                "H_dimer = [[12410, 87], [87, 12530]]\n",
                "times = np.linspace(0, 1000, 200)\n",
                "rho0 = qt.basis(2, 0) * qt.basis(2, 0).dag()\n",
                "\n",
                "t, states = simulate_pt_hops_ltc(H_dimer, poles, residues, ltc['delta'], rho0, times)\n",
                "populations = [s.diag() for s in states]\n",
                "\n",
                "plt.plot(t, populations)\n",
                "plt.title('FMO Dimer with LTC Correction (T=77K)')\n",
                "plt.show()\n"
            ]

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print('LTC notebook updated.')
