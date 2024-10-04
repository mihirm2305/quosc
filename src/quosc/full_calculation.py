from typing import Iterable, Tuple
import numpy as np
from .initialisation.create_mesh import create_trimesh, tile_mesh
from .slicing.slice_mesh import * # generate_heights, calculate_sections, get_section_data
from .slicing.group_sections import * # group_by_centre, split_groups_by_area
from .slicing.extremal_orbits import * # calculate_extremal_orbits, filter_extremal_orbits, group_extremal_orbits
from .slicing.extremal_orbits import calculate_extremal_orbits
from joblib import Parallel, delayed
import trimesh
from sklearn.cluster import DBSCAN
import pandas as pd
import warnings


def calculate_single_frequencies_curvatures(
            normal: np.ndarray,
            mesh: trimesh.Trimesh,
            reciprocal_lattice: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the frequencies and curvatures (and counts) for a single normal.
        """
        
        try:
            heights = generate_heights(reciprocal_lattice)
            sections = calculate_sections(mesh, normal, heights)
            areas, heights, centres = get_section_data(sections, normal, heights)

            grouped_areas, grouped_heights, grouped_centres, grouped_indices = group_by_centre(areas, heights, centres, reciprocal_lattice)
            fine_areas, fine_heights, fine_centres, fine_indices = split_groups_by_area(grouped_areas, grouped_heights, grouped_centres, grouped_indices, reciprocal_lattice)

            extremal_freqs, extremal_curvs, extremal_centres, extremal_indices = calculate_extremal_orbits(fine_areas, fine_heights, fine_centres, fine_indices)

            filtered_freqs, filtered_curvs, filtered_centres = filter_extremal_orbits(extremal_freqs, extremal_curvs, extremal_centres, reciprocal_lattice)

            final_freqs, final_curvs, final_centres, final_counts = group_extremal_orbits(filtered_freqs, filtered_curvs, filtered_centres)

            return final_freqs, final_curvs, final_counts
        
        except:
            warnings.warn("No extremal orbits found.")
            return np.array([]), np.array([]), np.array([])


def calculate_frequencies_curvatures(
        band_indices: Iterable[int] | int,
        band_energies: np.ndarray,
        reciprocal_lattice: np.ndarray,
        fermi_energy: float,
        start_normal: np.ndarray,
        end_normal: np.ndarray,
        num_points: int,
        save: bool = True,
        filename: str = 'angle_sweep.csv'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the frequencies and curvatures from the band energies.
    """
    
    if isinstance(band_indices, int):
        band_indices = [band_indices]

    if np.any(np.array(band_indices) >= band_energies.shape[0]):
        raise ValueError("Some band indices are out of range.")
    
    # normalise the normals and create an array of normal vectors between the start and end normals at equal angles
    start_normal = np.array(start_normal, dtype=float)
    end_normal = np.array(end_normal, dtype=float)
    start_normal /= np.linalg.norm(start_normal)
    end_normal /= np.linalg.norm(end_normal)
    angle = np.arccos(np.dot(start_normal, end_normal))
    thetas = np.linspace(0, angle, num_points)

    normals = generate_interpolated_vectors(start_normal, end_normal, num_points)

    frequencies = []
    curvatures = []
    all_counts = []
    angles = []
    bands = []

    print(f'{len(band_indices)} bands to process')

    for i, band_index in enumerate(band_indices):

        tiled_band_energy = tile_mesh(band_energies)

        mesh = create_trimesh(band_index, tiled_band_energy, reciprocal_lattice, fermi_energy)
        results = Parallel(n_jobs=-1)(delayed(calculate_single_frequencies_curvatures)(normal, mesh, reciprocal_lattice) for normal in normals)

        for result, theta in zip(results, thetas):
            freqs, curvs, counts = result

            frequencies.extend(freqs)
            curvatures.extend(curvs)
            all_counts.extend(counts)
            angles.extend([theta] * len(freqs))
            bands.extend([band_index] * len(freqs))


        print(f'Processed band {i + 1}/{len(band_indices)}')
    
    # package in a dataframe
    df = pd.DataFrame({'band': bands, 'angle': angles, 'frequency': frequencies, 'curvature': curvatures, 'count': all_counts})

    if save:
        df.to_csv(filename, index=False)

    return df

    
# same but instead of different normals, use different fermi energies (i.e., energy sweep)
def calculate_frequencies_curvatures_energy_sweep(
        band_indices: Iterable[int] | int,
        band_energies: np.ndarray,
        reciprocal_lattice: np.ndarray,
        fermi_energies: Tuple[float, float],
        normal: np.ndarray,
        num_points: int,
        save: bool = True,
        filename: str = 'energy_sweep.csv'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the frequencies and curvatures from the band energies.
    """
    
    if isinstance(band_indices, int):
        band_indices = [band_indices]

    if np.any(np.array(band_indices) >= band_energies.shape[0]):
        raise ValueError("Some band indices are out of range.")
    
    # only one normal is used - no need to generate normals

    unique_energies = np.linspace(fermi_energies[0], fermi_energies[1], num_points)

    frequencies = []
    curvatures = []
    energies = []
    bands = []

    def _single_energy_calc(
            fermi_energy: float,
            band_index: int,
            band_energies: np.ndarray,
            reciprocal_lattice: np.ndarray,
            save: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the frequencies and curvatures for a single normal.
        """
        try:
            tiled_band_energy = tile_mesh(band_energies)
            mesh = create_trimesh(band_index, tiled_band_energy, reciprocal_lattice, fermi_energy)
        
        except:
            warnings.warn("Couldn't create Fermi surface mesh - probably Fermi energy not in band energy range.")
            return np.array([]), np.array([]), np.array([])
            
        return calculate_single_frequencies_curvatures(normal, mesh, reciprocal_lattice)
    
    print(f'{len(band_indices)} bands to process')

    for i, band_index in enumerate(band_indices):

            results = Parallel(n_jobs=-1)(delayed(_single_energy_calc)(fermi_energy, band_index, band_energies, reciprocal_lattice) for fermi_energy in unique_energies)

            for result, fermi_energy in zip(results, unique_energies):
                freqs, curvs, counts = result
                # band_frequencies.extend(freqs)
                # band_curvatures.extend(curvs)
                # band_fermi_energies.extend([fermi_energy] * len(freqs))

                frequencies.extend(freqs)
                curvatures.extend(curvs)
                energies.extend([fermi_energy] * len(freqs))
                bands.extend([band_index] * len(freqs))

            print(f'Processed band {i + 1}/{len(band_indices)}')

    # package in a dataframe
    df = pd.DataFrame({'band': bands, 'energy': energies, 'frequency': frequencies, 'curvature': curvatures})

    if save:
        df.to_csv(filename, index=False)

    return df    
            


def slerp(
        v1: np.ndarray,
        v2: np.ndarray,
        t: float
) -> np.ndarray:
    """
    Perform spherical linear interpolation between two vectors v1 and v2 at interpolation factor t.
    """

    # Compute the dot product between the two vectors
    dot_product = np.dot(v1, v2)
    # Clip to ensure numerical stability in arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angle between the two vectors
    theta = np.arccos(dot_product)

    # If the vectors are nearly identical, avoid division by zero
    if np.isclose(theta, 0):
        return v1

    # Compute the slerp
    v = (np.sin((1 - t) * theta) * v1 + np.sin(t * theta) * v2) / np.sin(theta)

    return v

def generate_interpolated_vectors(
        v1: np.ndarray,
        v2: np.ndarray,
        N: int
) -> np.ndarray:
    """
    Generate N evenly spaced vectors between two unit vectors v1 and v2.
    """

    interpolated_vectors = []

    if N == 1:
        return np.array([v1])

    for i in range(N):
        t = i / (N - 1)  # Fractional distance along the arc
        interpolated_vector = slerp(v1, v2, t)
        interpolated_vectors.append(interpolated_vector)
    return np.array(interpolated_vectors)
