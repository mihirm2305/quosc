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