import json
import os

notebook_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Project_base/notebooks_roadmap/01_core_methodologies/hops_hierarchy.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Step 3: HEOMSolver placeholder
        if 'class HEOMSolver' in source:
            cell['source'] = [
                "class HOPSSolver:\n",
                "    \"\"\"\n",
                "    Stochastic HOPS solver (Hierarchy of Pure States).\n",
                "    \"\"\"\n",
                "    def __init__(self, H_sys, L, poles, residues, max_depth):\n",
                "        self.H_sys = H_sys\n",
                "        self.L = L\n",
                "        self.poles = poles\n",
                "        self.residues = residues\n",
                "        self.max_depth = max_depth\n",
                "        self.n_sys = H_sys.shape[0]\n",
                "        self.n_modes = len(poles)\n",
                "        \n",
                "    def generate_noise(self, times):\n",
                "        dt = times[1] - times[0]\n",
                "        n_steps = len(times)\n",
                "        # Generate Gaussian noise for each mode\n",
                "        noise = np.zeros((self.n_modes, n_steps), dtype=complex)\n",
                "        for k in range(self.n_modes):\n",
                "            # Variance of the noise according to correlation function\n",
                "            std = np.sqrt(np.abs(self.residues[k]) / (2 * dt))\n",
                "            noise[k] = std * (np.random.normal(size=n_steps) + 1j * np.random.normal(size=n_steps))\n",
                "        return noise\n",
                "\n",
                "    def propagate_trajectory(self, times, psi0, noise):\n",
                "        dt = times[1] - times[0]\n",
                "        psi_t = [psi0]\n",
                "        curr_psi = psi0\n",
                "        \n",
                "        # Simplified HOPS (first order hierarchy)\n",
                "        for i in range(1, len(times)):\n",
                "            # Drift term from bath poles\n",
                "            drift = 0\n",
                "            for k in range(self.n_modes):\n",
                "                drift += noise[k, i] * self.L\n",
                "            \n",
                "            # Effective Hamiltonian excitation\n",
                "            H_eff = self.H_sys - 1j * drift\n",
                "            \n",
                "            # Update wavefunction\n",
                "            d_psi = -1j * H_eff @ curr_psi * dt\n",
                "            curr_psi = curr_psi + d_psi\n",
                "            curr_psi = curr_psi / np.linalg.norm(curr_psi)\n",
                "            psi_t.append(curr_psi)\n",
                "            \n",
                "        return np.array(psi_t)\n",
                "\n",
                "print('HOPS solver implemented')"
            ]
        
        # Step 4: simple_HEOM_example
        elif 'def simple_HEOM_example' in source:
            cell['source'] = [
                "def run_hops_demo(n_traj=10):\n",
                "    times = np.linspace(0, 100, 200)\n",
                "    # Use poles from Padé notebook\n",
                "    poles = np.array([50.0, 100.0])\n",
                "    residues = np.array([10.0 + 5j, 5.0 + 2j])\n",
                "    \n",
                "    solver = HOPSSolver(H_sys, sigma_z, poles, residues, max_depth=1)\n",
                "    \n",
                "    all_pops = []\n",
                "    for _ in range(n_traj):\n",
                "        noise = solver.generate_noise(times)\n",
                "        psi_path = solver.propagate_trajectory(times, psi0, noise)\n",
                "        all_pops.append(np.abs(psi_path[:, 0])**2)\n",
                "    \n",
                "    avg_pop = np.mean(all_pops, axis=0)\n",
                "    \n",
                "    plt.plot(times, avg_pop, label='Average Population (HOPS)')\n",
                "    plt.plot(times, all_pops[0], alpha=0.3, label='Sample Trajectory')\n",
                "    plt.xlabel('Time (fs)')\n",
                "    plt.ylabel('Population')\n",
                "    plt.legend()\n",
                "    plt.show()\n",
                "\n",
                "run_hops_demo()"
            ]

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print('HOPS notebook updated.')
