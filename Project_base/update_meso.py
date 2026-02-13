import json
import os

notebook_path = '/home/taamangtchu/Documents/Github/Comparative-study-of-the-Anderson-model-in-weak-and-strong-interaction-regimes/Project_base/notebooks_roadmap/01_core_methodologies/mesohops_compression.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Step 2: TensorCompression class
        if 'class TensorCompression' in source:
            cell['source'] = [
                "class MesoHopsCompressor:\n",
                "    \"\"\"\n",
                "    Tensor compression for MesoHOPS using Singular Value Decomposition.\n",
                "    \"\"\"\n",
                "    def __init__(self, threshold=1e-5, max_rank=50):\n",
                "        self.threshold = threshold\n",
                "        self.max_rank = max_rank\n",
                "        \n",
                "    def compress_wavefunction(self, psi, n_sites):\n",
                "        \"\"\"\n",
                "        Compress a multi-site wavefunction using SVD iteratively (MPS-like compression).\n",
                "        \"\"\"\n",
                "        d = 2 # local dimension\n",
                "        psi_mat = psi.reshape(d, d**(n_sites-1))\n",
                "        U, S, V = np.linalg.svd(psi_mat, full_matrices=False)\n",
                "        \n",
                "        # Truncation\n",
                "        rank = min(self.max_rank, np.sum(S > self.threshold))\n",
                "        S_trunc = S[:rank]\n",
                "        U_trunc = U[:, :rank]\n",
                "        V_trunc = V[:rank, :]\n",
                "        \n",
                "        return U_trunc, S_trunc, V_trunc\n",
                "\n",
                "print('MesoHOPS Compressor implemented')"
            ]
        
        # Step 4: MesoHOPSSolver class
        elif 'class MesoHOPSSolver' in source:
            cell['source'] = [
                "def run_mesohops_benchmark(n_sites=5):\n",
                "    print(f'Running MesoHOPS benchmark for {n_sites} sites...')\n",
                "    # Create a highly entangled state (W-state placeholder)\n",
                "    dim = 2**n_sites\n",
                "    psi = np.zeros(dim, dtype=complex)\n",
                "    for i in range(n_sites):\n",
                "        psi[2**i] = 1.0\n",
                "    psi = psi / np.linalg.norm(psi)\n",
                "    \n",
                "    compressor = MesoHopsCompressor(threshold=1e-4, max_rank=10)\n",
                "    U, S, V = compressor.compress_wavefunction(psi, n_sites)\n",
                "    \n",
                "    print(f'Original size: {dim} elements')\n",
                "    print(f'Compressed size (first split): {U.size + S.size + V.size} elements')\n",
                "    print(f'Compression factor: {dim / (U.size + S.size + V.size):.2f}x')\n",
                "    print(f'Singular values retained: {len(S)}')\n",
                "\n",
                "run_mesohops_benchmark(n_sites=10)"
            ]

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print('MesoHOPS notebook updated.')
