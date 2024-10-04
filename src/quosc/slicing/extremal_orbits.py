from scipy.signal import find_peaks
from typing import List, Tuple
import numpy as np
from warnings import warn
from ..utils import area_to_freq

def calculate_extremal_orbits(
        grouped_areas: List[np.ndarray],
        grouped_heights: List[np.ndarray],
        grouped_centres: List[np.ndarray],
        grouped_indices: List[np.ndarray],
        return_frequencies: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the frequency and curvature from the extremal areas.
    """
    
    frequencies = []
    curvatures = []
    all_centres = []
    all_indices = []
    
    for areas, heights, centres, indices in zip(grouped_areas, grouped_heights, grouped_centres, grouped_indices):
        if len(areas) < 3:
            continue

        # if there are duplicate heights, average the areas and give warnings
        if len(np.unique(heights)) != len(heights):
            warn("Duplicate heights found in the heights array. Averaging the areas for the same height.")
            new_areas = []
            new_heights = []
            new_centres = []
            new_indices = []
            for height in np.unique(heights):
                mask = heights == height
                new_areas.append(np.mean(areas[mask]))
                new_heights.append(height)
                new_centres.append(np.mean(centres[mask], axis=0))
                new_indices.append(indices[mask][0])

            areas = np.array(new_areas)
            heights = np.array(new_heights)
            centres = np.array(new_centres)
            indices = np.array(new_indices)

        if len(areas) < 3:
            continue
        
        # Find the peaks and valleys
        peaks, _ = find_peaks(areas)
        valleys, _ = find_peaks(-areas)

        # Find the extremal areas
        extremal_indices = np.concatenate([peaks, valleys])
        extremal_areas = areas[extremal_indices]
        if return_frequencies:
            extremal_freqs = area_to_freq(extremal_areas)
        else:
            extremal_freqs = extremal_areas

        # find the curvature (second derivative of the area wrt height)
        curvature = np.gradient(np.gradient(areas, heights), heights)
        extremal_curvatures = curvature[extremal_indices]

        extremal_centres = centres[extremal_indices]

        frequencies.extend(extremal_freqs)
        curvatures.extend(extremal_curvatures)
        all_centres.extend(extremal_centres)
        all_indices.extend(indices[extremal_indices])

    # process the curvatures into a more useful form
    curvatures = np.array(curvatures)
    curvatures = area_to_freq(curvatures)

    return np.array(frequencies), curvatures, np.array(all_centres), np.array(all_indices)

# TODO: calculate masses


def filter_extremal_orbits(
        frequencies: np.ndarray,
        curvatures: np.ndarray,
        centres: np.ndarray,
        reciprocal_lattice: np.ndarray,
        shift: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter the extremal orbits based on if the centre is within the bounds.
    """

    # convert centres to fractional coordinates
    reciprocal_lattice = np.array(reciprocal_lattice)
    reciprocal_lattice_inv = np.linalg.inv(reciprocal_lattice)
    fractional_centres = np.dot(centres, reciprocal_lattice_inv)

    # shift the centres by the shift (to avoid rounding errors for special points)
    shifted_centres = fractional_centres + shift

    # if 0,0,0 is the nearest reciprocal lattice point, then the centre is within the bounds
    shifted_centres = np.round(shifted_centres)
    mask = np.all(shifted_centres == 0, axis=1)

    return frequencies[mask], curvatures[mask], centres[mask]

def group_extremal_orbits(
        frequencies: np.ndarray,
        curvatures: np.ndarray,
        centres: np.ndarray,
        # indices: np.ndarray,
        tolerances: float | Tuple[float, float] = (0.02, 0.6)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    If there are multiple extremal orbits with the similar frequencies and curvatures, group them together and average.

    Args:
        frequencies: The frequencies of the extremal orbits.
        curvatures: The curvatures of the extremal orbits.
        centres: The centres of the extremal orbits.
        reciprocal_lattice: The reciprocal lattice vectors.

    Returns:
        The grouped frequencies, curvatures, centres, and the counts of the grouped orbits.
    """

    # Group the extremal orbits with similar frequencies and curvatures
    if isinstance(tolerances, float):
        tolerances = (tolerances, tolerances)

    grouped_frequencies = []
    grouped_curvatures = []
    grouped_centres = []

    for i, (freq, curvature, centre) in enumerate(zip(frequencies, curvatures, centres)):
        if i == 0:
            grouped_frequencies.append([freq])
            grouped_curvatures.append([curvature])
            grouped_centres.append([centre])
            continue

        # Check if the frequency and curvature is within the tolerance
        is_grouped = False
        for j, (grouped_freq, grouped_curvature) in enumerate(zip(grouped_frequencies, grouped_curvatures)):
            if (np.abs((freq - np.mean(grouped_freq))/np.mean(grouped_freq)) < tolerances[0]) and (np.abs((curvature - np.mean(grouped_curvature))/np.mean(grouped_curvature)) < tolerances[1]):
                grouped_freq.append(freq)
                grouped_curvature.append(curvature)
                grouped_centres[j].append(centre)
                is_grouped = True
                break

        if not is_grouped:
            grouped_frequencies.append([freq])
            grouped_curvatures.append([curvature])
            grouped_centres.append([centre])

    # Average the grouped frequencies and curvatures
    counts = np.array([len(group) for group in grouped_frequencies])
    grouped_frequencies = np.array([np.mean(group) for group in grouped_frequencies])
    grouped_curvatures = np.array([np.mean(group) for group in grouped_curvatures])
    grouped_centres = [np.array(group) for group in grouped_centres]

    return grouped_frequencies, grouped_curvatures, grouped_centres, counts

