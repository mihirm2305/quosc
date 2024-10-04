import numpy as np
from typing import Tuple


def area_to_freq(area: float | np.ndarray) -> float | np.ndarray:
    """
    Convert an area to a frequency.

    Args:
        area: The area in reciprocal space to convert to a frequency (in Angstrom^(-2)).

    Returns:
        The frequency (in Teslas) corresponding to the area.
    """
    hbar = 1.0545718e-34 # J s
    e = 1.60217662e-19 # C

    freq = area * (hbar / (2 * np.pi * e)) * 1e20
    return freq


def freq_to_area(freq: float | np.ndarray) -> float | np.ndarray:
    """
    Convert a frequency to an area.

    Args:
        freq: The frequency (in Teslas) to convert to an area.

    Returns:
        The area in reciprocal space corresponding to the frequency (in Angstrom^(-2)).
    """
    hbar = 1.0545718e-34 # J s
    e = 1.60217662e-19 # C

    area = freq / (hbar / (2 * np.pi * e)) / 1e20
    return area


# def calculate_single_frequencies_curvatures(
#             normal: np.ndarray,
#             mesh: trimesh.Trimesh,
#             reciprocal_lattice: np.ndarray
#     ) -> Tuple[dict, dict, dict]:
#         """
#         Calculate the frequencies and curvatures for a single normal.
#         """
        
#         heights = generate_heights(reciprocal_lattice)
#         sections = calculate_sections(mesh, normal, heights)
#         areas, heights, centres = get_section_data(sections, normal, heights)

#         grouped_areas, grouped_heights, grouped_centres, grouped_indices = group_by_centre(areas, heights, centres, reciprocal_lattice)
#         fine_areas, fine_heights, fine_centres, fine_indices = split_groups_by_area(grouped_areas, grouped_heights, grouped_centres, grouped_indices, reciprocal_lattice)

#         extremal_freqs, extremal_curvs, extremal_centres, extremal_indices = calculate_extremal_orbits(fine_areas, fine_heights, fine_centres, fine_indices)

#         filtered_freqs, filtered_curvs, filtered_centres = filter_extremal_orbits(extremal_freqs, extremal_curvs, extremal_centres, reciprocal_lattice)

#         final_freqs, final_curvs, final_centres, final_counts = group_extremal_orbits(filtered_freqs, filtered_curvs, filtered_centres)

#         return final_freqs, final_curvs, final_counts