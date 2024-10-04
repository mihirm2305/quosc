import numpy as np
from joblib import Parallel, delayed
from typing import List, Tuple
from . import load_data

# function to generate k-points (in fractional coordinates)
def generate_kpoints(
        n_kpoints: int | Tuple[int, int, int],
        bounds: Tuple[int, int] | Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (-0.5, 0.5)
) -> np.ndarray:
    """
    Generate a grid of k-points in fractional coordinates.
    """

    if isinstance(n_kpoints, int):
        n_kpoints = (n_kpoints, n_kpoints, n_kpoints)

    if isinstance(bounds, tuple):
        bounds = [bounds, bounds, bounds]
    
    kpoints = np.mgrid[
        bounds[0][0]:bounds[0][1]:n_kpoints[0]*1j,
        bounds[1][0]:bounds[1][1]:n_kpoints[1]*1j,
        bounds[2][0]:bounds[2][1]:n_kpoints[2]*1j
    ].reshape(3, -1).T

    return kpoints

# function to calculate the energy bands
def get_band_structure(
        kp: np.ndarray,
        equivalences: List[np.ndarray],
        coeffs: np.ndarray,
        chunk_size: int = 10000,
        print_progress: bool = True
) -> np.ndarray:
    """
    Sample the energy bands at particular k-points using the BoltzTraP2 interpolation coefficients.
    """

    # helper function to calculate the energy phase
    def _calculate_energy_phase(equiv, kp_chunk, tpii):
        phase0 = np.exp(tpii * kp_chunk @ equiv.T)
        phase = np.sum(phase0, axis=1)
        return phase

    tpii = 2j * np.pi
    
    n_kpoints = len(kp)
    n_chunks = int(np.ceil(n_kpoints / chunk_size))
    
    egrid_total = []
    
    for i in range(n_chunks):
        kp_chunk = kp[i * chunk_size:(i + 1) * chunk_size]

        # Parallelize the energy phase calculations for each chunk
        phases = Parallel(n_jobs=-1)(
            delayed(_calculate_energy_phase)(equiv, kp_chunk, tpii) for equiv in equivalences
        )
        
        phases = np.array(phases)

        nstar = np.array([len(equiv) for equiv in equivalences])
        
        # Normalize
        phases /= nstar[:, np.newaxis]
        
        # Calculate energies
        egrid = coeffs @ phases
        egrid_total.append(egrid.real)
        
        # Print progress
        if print_progress:
            progress = (i + 1) / n_chunks * 100
            print(f'Processing chunk {i + 1}/{n_chunks} - {progress:.2f}% complete', end='\r')

    if print_progress:
        print()  # Move to the next line after progress is complete
    
    # Combine all chunks
    egrid_total = np.concatenate(egrid_total, axis=-1)
    
    return egrid_total

# function to reshape the energy bands into a 3D array
def reshape_bands(
        egrid: np.ndarray,
        n_kpoints: int | Tuple[int, int, int]
) -> np.ndarray:
    """
    Reshape the energy bands into a 3D array.
    i.e. (n_bands, n_kpoints) -> (n_bands, n_kpoints_x, n_kpoints_y, n_kpoints_z)
    """
    
    if isinstance(n_kpoints, int):
        n_kpoints = (n_kpoints, n_kpoints, n_kpoints)

    egrid = egrid.reshape(-1, *n_kpoints)
    
    return egrid

# wrapper function to generate k-points and calculate the energy bands
def calculate_bands(
        n_kpoints: int | Tuple[int, int, int],
        filepath: str = 'interpolation.bt2',
        bounds: Tuple[int, int] | Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]] = (-0.5, 0.5),
        chunk_size: int = 10000,
        print_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate k-points and calculate the energy bands.
    """

    # Load the BoltzTraP2 data
    data_bt = load_data.read_boltztrap_interpolation(filepath)
    
    # Get the equivalences and interpolation coefficients
    equivalences = load_data.get_equivalences(data_bt)
    coeffs = load_data.get_interpolation_coefficients(data_bt)
    
    # Generate k-points
    kp = generate_kpoints(n_kpoints, bounds)
    
    # Calculate the energy bands
    egrid = get_band_structure(kp, equivalences, coeffs, chunk_size, print_progress)
    
    # Reshape the energy bands
    egrid = reshape_bands(egrid, n_kpoints)

    # convert from Hartree to eV
    egrid *= 27.211386
    
    return egrid