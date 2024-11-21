# this file contains functions to read the interpolation.bt2 file from BoltzTraP2 and extract the relevant information

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import json
import lzma

# function to read BoltzTraP2 output files
def read_boltztrap_interpolation(
        filepath: str = 'interpolation.bt2'
) -> list:
    """
    Read the interpolation.bt2 file from BoltzTraP2
    """
    data_bt = lzma.open(filepath).read()
    data_bt = json.loads(data_bt.decode('utf-8'))

    return data_bt

# function to get the Fermi energy from the interpolation.bt2 file
def get_fermi_energy(
        data_bt: list
) -> float:
    """
    Get the Fermi energy from the interpolation.bt2 file
    """
    fermi_energy = data_bt[0]['fermi']

    hartree_to_ev = 27.211386245981

    return fermi_energy * hartree_to_ev

# lattvec = np.array(data_bt[0]['atoms']['cell']['data'])# * a_0
# reciprocal_lattice = 2*np.pi*np.linalg.inv(lattvec).T
# fermi_energy = data_bt[0]['fermi']

# function to get the lattice vectors from the interpolation.bt2 file
def get_lattice_vectors(
        data_bt: list
) -> np.ndarray:
    """
    Get the lattice vectors from the interpolation.bt2 file
    """
    lattice_vectors = np.array(data_bt[0]['atoms']['cell']['data'])

    return lattice_vectors

# function to calculate the reciprocal lattice vectors
def calculate_reciprocal_lattice_vectors(
        lattice_vectors: np.ndarray
) -> np.ndarray:
    """
    Calculate the reciprocal lattice vectors
    """
    reciprocal_lattice = 2*np.pi*np.linalg.inv(lattice_vectors).T

    return reciprocal_lattice

# function to get the equivalences from the interpolation.bt2 file
def get_equivalences(
        data_bt: list
) -> List[np.ndarray]:
    """
    Get the equivalences from the interpolation.bt2 file
    """
    equivalences = [np.array(equiv['data']) for equiv in data_bt[1]]

    return equivalences

# function to get the interpolation coefficients from the interpolation.bt2 file
def get_interpolation_coefficients(
        data_bt: list
) -> np.ndarray:
    """
    Get the interpolation coefficients from the interpolation.bt2 file
    """
    interpolation_coefficients = np.array(data_bt[2]['real']) + 1j*np.array(data_bt[2]['imag'])

    return interpolation_coefficients


# Function to read a BXSF file
def load_skeaf(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a .skeaf file and return the band structure, fermi energy, and reciprocal lattice vectors
    """
    
    with open(filepath
                ) as f:
            lines = f.readlines()
    if 'Fermi Energy' not in lines[1]:
        raise ValueError('Fermi energy not found in the file')
    fermi_energy = float(lines[1].split()[-1])

    # number on line 6
    number = int(lines[6].split()[-1])

    nk = np.array([int(n) for n in lines[7].split()])
    reciprocal_lattice = np.array([list(map(float, line.split())) for line in lines[9:12]])
    a_0 = 0.5291772105  # Bohr radius in Angstrom
    reciprocal_lattice *=  2 * np.pi / a_0
    nband = int(lines[12].split()[-1])
    energies = []
    for line in lines[13:]:
        if 'END_BANDGRID_3D' in line:
            break
        energies.extend(list(map(float, line.split())))
    energies = np.array(energies)
    if len(energies) != nk[0]*nk[1]*nk[2]:
        raise ValueError('Number of energies does not match the number of k-points and bands')
    kpoints = [np.array([i, j, k]) for i in range(nk[0]) for j in range(nk[1]) for k in range(nk[2])]
    kpoints = np.array(kpoints)
    k_step = reciprocal_lattice / nk
    # kpoints = kpoints.dot(k_step)
    k_grid = kpoints.reshape(nk[0], nk[1], nk[2], 3)
    energy_grid = energies.reshape(nk[0], nk[1], nk[2])

    return energy_grid, fermi_energy, reciprocal_lattice