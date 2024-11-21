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
            reciprocal_lattice: np.ndarray,
            height_bz_divisions: int = 100,
            area_grouping_tolerance: float = 0.05,
            frequency_tolerance: float = 0.02,
            curvature_tolerance: float = 0.6
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the frequencies and curvatures (and counts) for a single normal.
        """
        
        try:
            heights = generate_heights(reciprocal_lattice, bz_divisions=height_bz_divisions)
            sections = calculate_sections(mesh, normal, heights)
            areas, heights, centres = get_section_data(sections, normal, heights)

            grouped_areas, grouped_heights, grouped_centres, grouped_indices = group_by_centre(areas, heights, centres, reciprocal_lattice, tolerance=area_grouping_tolerance)
            fine_areas, fine_heights, fine_centres, fine_indices = split_groups_by_area(grouped_areas, grouped_heights, grouped_centres, grouped_indices, reciprocal_lattice)

            extremal_freqs, extremal_curvs, extremal_centres, extremal_indices = calculate_extremal_orbits(fine_areas, fine_heights, fine_centres, fine_indices)

            filtered_freqs, filtered_curvs, filtered_centres = filter_extremal_orbits(extremal_freqs, extremal_curvs, extremal_centres, reciprocal_lattice)

            final_freqs, final_curvs, final_centres, final_counts = group_extremal_orbits(filtered_freqs, filtered_curvs, filtered_centres, tolerances=(frequency_tolerance, curvature_tolerance))

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
        filename: str = 'angle_sweep.csv',
        tiling: int | Tuple[int, int, int] = 3,
        endpoints_included: bool = True,
        height_bz_divisions: int = 100,
        area_grouping_tolerance: float = 0.05,
        frequency_tolerance: float = 0.02,
        curvature_tolerance: float = 0.6
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

        tiled_band_energy = tile_mesh(band_energies, tiling=tiling, endpoints_included=endpoints_included)
        mesh = create_trimesh(band_index, tiled_band_energy, reciprocal_lattice, fermi_energy, tiling=tiling)
        
        results = Parallel(n_jobs=-1)(delayed(calculate_single_frequencies_curvatures)\
                                      (normal, mesh, reciprocal_lattice, height_bz_divisions, \
                                       area_grouping_tolerance, frequency_tolerance, curvature_tolerance)\
                                        for normal in normals)

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

def calculate_frequencies_curvatures_masses(
    band_indices: Iterable[int] | int,
    band_energies: np.ndarray,
    reciprocal_lattice: np.ndarray,
    fermi_energy: float,
    start_normal: np.ndarray,
    end_normal: np.ndarray,
    num_points: int,
    delta_E: float = 1e-5,
    save: bool = True,
    filename: str = 'angle_sweep.csv',
    tiling: int | Tuple[int, int, int] = 3,
    endpoints_included: bool = True,
    height_bz_divisions: int = 100,
    area_grouping_tolerance: float = 0.05,
    frequency_tolerance: float = 0.02,
    curvature_tolerance: float = 0.6
) -> pd.DataFrame:
    
    print(f"Calculating frequencies, curvatures, and masses for {num_points} angles.")
    print('Step 1/3 in progress...')

    df = calculate_frequencies_curvatures(
        band_indices = band_indices,
        band_energies = band_energies,
        reciprocal_lattice = reciprocal_lattice,
        fermi_energy = fermi_energy,
        start_normal = start_normal,
        end_normal = end_normal,
        num_points = num_points,
        save=False,
        tiling=tiling,
        endpoints_included=endpoints_included,
        height_bz_divisions=height_bz_divisions,
        area_grouping_tolerance=area_grouping_tolerance,
        frequency_tolerance=frequency_tolerance,
        curvature_tolerance=curvature_tolerance
    )

    print('Step 1/3 complete.')
    print('Step 2/3 in progress...')
    
    df_plus = calculate_frequencies_curvatures(
        band_indices = band_indices,
        band_energies = band_energies,
        reciprocal_lattice = reciprocal_lattice,
        fermi_energy = fermi_energy + delta_E/2,
        start_normal = start_normal,
        end_normal = end_normal,
        num_points = num_points,
        save=False,
        tiling=tiling,
        endpoints_included=endpoints_included,
        height_bz_divisions=height_bz_divisions,
        area_grouping_tolerance=area_grouping_tolerance,
        frequency_tolerance=frequency_tolerance,
        curvature_tolerance=curvature_tolerance        
    )

    print('Step 2/3 complete.')
    print('Step 3/3 in progress...')

    df_minus = calculate_frequencies_curvatures(
        band_indices = band_indices,
        band_energies = band_energies,
        reciprocal_lattice = reciprocal_lattice,
        fermi_energy = fermi_energy - delta_E/2,
        start_normal = start_normal,
        end_normal = end_normal,
        num_points = num_points,
        save=False,
        tiling=tiling,
        endpoints_included=endpoints_included,
        height_bz_divisions=height_bz_divisions,
        area_grouping_tolerance=area_grouping_tolerance,
        frequency_tolerance=frequency_tolerance,
        curvature_tolerance=curvature_tolerance
    )

    print('Step 3/3 complete.')

    def grad_to_mass(grad: float | np.ndarray) -> float | np.ndarray:
        """
        Convert a gradient to a mass.
        """
        hbar = 1.0545718e-34 # J s
        e = 1.60217662e-19 # C
        m_e = 9.10938356e-31 # kg

        area = grad / (hbar / (2 * np.pi * e))
        mass = area * hbar**2 / (2 * np.pi * e * m_e)

        return mass
    
    masses = []

    for band in df['band'].unique():

        for angle in df['angle'].unique():


            df_angle = df[(df['angle'] == angle) & (df['band'] == band)]
            df_angle_plus = df_plus[(df_plus['angle'] == angle) & (df_plus['band'] == band)]
            df_angle_minus = df_minus[(df_minus['angle'] == angle) & (df_minus['band'] == band)]

            # if df_angle is empty, skip to the next angle
            if df_angle.empty:
                # print(f"Skipping angle {np.rad2deg(angle):.0f} due to empty df_angle.")
                continue

            for row in df_angle.itertuples():
                freq = row.frequency
                curv = row.curvature

                # find all the rows in the plus and minus dataframes that are within 2% of the frequency and 60% of the curvature and the same count
                plus_condition = (np.abs(df_angle_plus['frequency'] - freq) < 0.01 * freq) & (np.abs(df_angle_plus['curvature'] - curv) < 0.1 * np.abs(curv)) & (df_angle_plus['count'] == row.count)
                minus_condition = (np.abs(df_angle_minus['frequency'] - freq) < 0.01 * freq) & (np.abs(df_angle_minus['curvature'] - curv) < 0.1 * np.abs(curv)) & (df_angle_minus['count'] == row.count)

                df_angle_plus_filtered = df_angle_plus[plus_condition]
                df_angle_minus_filtered = df_angle_minus[minus_condition]

                # if either filtered dataframe is empty, skip to the next row
                if df_angle_plus_filtered.empty or df_angle_minus_filtered.empty:
                    # print(f"Skipping angle {np.rad2deg(angle):.0f}, frequency {freq} due to empty filtered dataframes.")
                    masses.append(np.nan)
                    continue

                # if the length of either filtered dataframe is greater than 1, choose the one with the closest frequency
                if len(df_angle_plus_filtered) > 1:
                    closest_plus_index = np.abs(df_angle_plus_filtered['frequency'].values - freq).argmin()
                    df_angle_plus_filtered = df_angle_plus_filtered.iloc[[closest_plus_index]]

                if len(df_angle_minus_filtered) > 1:
                    closest_minus_index = np.abs(df_angle_minus_filtered['frequency'].values - freq).argmin()
                    df_angle_minus_filtered = df_angle_minus_filtered.iloc[[closest_minus_index]]

                # now we can calculate the gradient and mass
                freq_plus = df_angle_plus_filtered['frequency'].values[0]
                freq_minus = df_angle_minus_filtered['frequency'].values[0]

                gradient = (freq_plus - freq_minus) / (1e-5)
                mass = grad_to_mass(gradient)

                masses.append(mass)

    # add plot_mass to df as 'mass' column
    df['mass'] = pd.Series(masses)

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
