"""
Orca Wrapper for DFT-based Quantum Agrivoltaics.

This module provides a wrapper for executing Orca DFT calculations and
parsing outputs for nucleophilic/electrophilic reactivity analysis.
"""

import logging
import os
import re
import subprocess
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class OrcaRunner:
    """
    Wrapper for Orca DFT execution and parsing.
    """

    def __init__(
        self,
        orca_path: str = "/home/taamangtchu/orca_6_1_0/orca",
        work_dir: str = "orca_work",
        nprocs: int = 4,
    ):
        self.orca_path = orca_path
        self.work_dir = work_dir
        self.nprocs = nprocs

        if not os.path.exists(self.orca_path):
            logger.warning(
                f"Orca executable not found at {self.orca_path}. Calculations will fail."
            )

        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def generate_input(
        self,
        molecule_name: str,
        coords: str,
        method: str = "wB97X-D4",
        basis: str = "def2-SVP",
        charge: int = 0,
        multiplicity: int = 1,
        extra_keywords: str = "TightSCF",
    ):
        """
        Generate Orca input file (.inp).
        """
        input_content = f"""! {method} {basis} {extra_keywords}
%pal nprocs {self.nprocs} end

* xyz {charge} {multiplicity}
{coords}
*
"""
        filepath = os.path.join(self.work_dir, f"{molecule_name}.inp")
        with open(filepath, "w") as f:
            f.write(input_content)
        return filepath

    def run_calculation(self, input_file: str) -> str:
        """
        Execute Orca calculation.
        """
        output_file = input_file.replace(".inp", ".out")
        cwd = os.getcwd()
        try:
            logger.info(f"Running Orca calculation: {input_file}")
            # Run Orca from within the work directory to keep root clean
            os.chdir(self.work_dir)

            inp_base = os.path.basename(input_file)
            out_base = os.path.basename(output_file)

            with open(out_base, "w") as out:
                subprocess.run([self.orca_path, inp_base], stdout=out, check=True)

            return output_file
        except KeyboardInterrupt:
            # Preserve user interrupt
            raise
        except subprocess.CalledProcessError as e:
            logger.error(f"Orca calculation failed (non-zero exit): {e}")
            raise
        except (FileNotFoundError, PermissionError, OSError) as e:
            logger.error(f"Orca execution environment error: {e}")
            raise
        finally:
            # Ensure we always restore the working directory
            try:
                os.chdir(cwd)
            except OSError as e:
                logger.debug(f"Failed to restore working directory: {e}", exc_info=True)

    def parse_mulliken_charges(self, output_file: str) -> np.ndarray:
        """
        Parse Mulliken populations/charges from Orca output.
        """
        charges = []
        if not os.path.exists(output_file):
            return np.array([])

        with open(output_file, "r") as f:
            lines = f.readlines()

        found = False
        for i, line in enumerate(lines):
            if "MULLIKEN ATOMIC CHARGES" in line:
                found = True
                # Skip header lines
                for j in range(i + 2, len(lines)):
                    if lines[j].strip() == "" or "Sum of atomic charges" in lines[j]:
                        break
                    parts = lines[j].split()
                    if len(parts) >= 4:
                        charges.append(float(parts[-1]))
                break

        if not found:
            logger.warning("Mulliken charges not found in output file.")

        return np.array(charges)

    def parse_orbital_energies(self, output_file: str) -> Dict[str, float]:
        """
        Parse HOMO and LUMO energies from Orca output.
        """
        homo = None
        lumo = None

        if not os.path.exists(output_file):
            return {"homo": 0.0, "lumo": 0.0}

        with open(output_file, "r") as f:
            content = f.read()

        # Look for the orbital energy table
        # We look for the last occurrence of the orbital table to get converged values
        tables = re.findall(r"ORBITAL ENERGIES\s+-+.*?\n(.*?)\n\s*\n", content, re.DOTALL)
        if not tables:
            return {"homo": 0.0, "lumo": 0.0}

        last_table = tables[-1]
        lines = last_table.strip().split("\n")

        for _i, line in enumerate(lines):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    occ = float(parts[1])
                    energy = float(parts[3])  # Energy in eV

                    if occ > 0:
                        homo = energy
                    elif lumo is None:
                        lumo = energy
                        break
                except ValueError:
                    continue

        return {
            "homo": homo if homo is not None else 0.0,
            "lumo": lumo if lumo is not None else 0.0,
            "gap": (lumo - homo) if (lumo is not None and homo is not None) else 0.0,
        }

    def calculate_reactivity_descriptors(
        self, molecule_name: str, coords: str, method: str = "wB97X-D4", basis: str = "def2-SVP"
    ) -> Dict[str, Any]:
        """
        Perform 3-state DFT calculations (N, N+1, N-1) to obtain Fukui functions
        and frontier orbital energies.
        """
        # 1. Neutral (N)
        logger.info(f"Computing Neutral state for {molecule_name}")
        inp_n = self.generate_input(
            f"{molecule_name}_N", coords, method, basis, charge=0, multiplicity=1
        )
        out_n = self.run_calculation(inp_n)
        q_n = self.parse_mulliken_charges(out_n)
        orbitals = self.parse_orbital_energies(out_n)

        # 2. Anion (N+1)
        logger.info(f"Computing Anion state for {molecule_name}")
        inp_p1 = self.generate_input(
            f"{molecule_name}_P1", coords, method, basis, charge=-1, multiplicity=2
        )
        out_p1 = self.run_calculation(inp_p1)
        q_p1 = self.parse_mulliken_charges(out_p1)

        # 3. Cation (N-1)
        logger.info(f"Computing Cation state for {molecule_name}")
        inp_m1 = self.generate_input(
            f"{molecule_name}_M1", coords, method, basis, charge=1, multiplicity=2
        )
        out_m1 = self.run_calculation(inp_m1)
        q_m1 = self.parse_mulliken_charges(out_m1)

        # Calculate Fukui functions
        # f+ = q_N - q_{N+1}
        # f- = q_{N-1} - q_N
        f_plus = q_n - q_p1
        f_minus = q_m1 - q_n
        f_radical = 0.5 * (f_plus + f_minus)

        return {
            "fukui_nucleophilic": f_plus,
            "fukui_electrophilic": f_minus,
            "fukui_radical": f_radical,
            "homo_ev": orbitals["homo"],
            "lumo_ev": orbitals["lumo"],
            "gap_ev": orbitals["gap"],
        }
