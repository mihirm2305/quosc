from sklearn.cluster import DBSCAN
import numpy as np
from typing import Optional, Tuple, List

def group_by_centre(
        areas: np.ndarray,
        heights: np.ndarray,
        centres: np.ndarray,
        reciprocal_lattice: Optional[np.ndarray] = None,
        tolerance: float = 5e-2,
        eps: Optional[float] = None,
        min_samples: int = 1
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Group areas by their 3D centre.
    """
    if reciprocal_lattice is None and eps is None:
        raise ValueError("At least one of reciprocal_lattice or eps must be provided.")
    
    if eps is None:
        eps = np.min(np.linalg.norm(reciprocal_lattice, axis=1)) * tolerance
    

    indices = np.arange(len(centres))

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(centres)
    
    grouped_areas = []
    grouped_heights = []
    grouped_centres = []
    grouped_indices = []
    
    for label in np.unique(labels):
        if label != -1:
            mask = labels == label
            grouped_areas.append(areas[mask])
            grouped_heights.append(heights[mask])
            grouped_centres.append(centres[mask])
            grouped_indices.append(indices[mask])
    
    return grouped_areas, grouped_heights, grouped_centres, grouped_indices


def split_groups_by_area(
        grouped_areas: List[np.ndarray],
        grouped_heights: List[np.ndarray],
        grouped_centres: List[np.ndarray],
        grouped_indices: List[np.ndarray],
        reciprocal_lattice: Optional[np.ndarray] = None,
        tolerance: float = 1e-1,
        eps: Optional[float] = None,
        min_samples: int = 1
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Group areas by their 3D centre.
    """

    if reciprocal_lattice is None and eps is None:
        raise ValueError("At least one of reciprocal_lattice or eps must be provided.")
    
    if eps is None:
        eps = np.min(np.linalg.norm(reciprocal_lattice, axis=1)) ** 2 * tolerance
    
    fine_grouped_areas = []
    fine_grouped_heights = []
    fine_grouped_centres = []
    fine_grouped_indices = []
    
    for areas, heights, centres, indices in zip(grouped_areas, grouped_heights, grouped_centres, grouped_indices):

        if len(np.unique(heights)) < len(heights):

            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(areas.reshape(-1, 1))

            for label in np.unique(labels):
                if label != -1:
                    mask = labels == label
                    fine_grouped_areas.append(areas[mask])
                    fine_grouped_heights.append(heights[mask])
                    fine_grouped_centres.append(centres[mask])
                    fine_grouped_indices.append(indices[mask])

        else:
            fine_grouped_areas.append(areas)
            fine_grouped_heights.append(heights)
            fine_grouped_centres.append(centres)
            fine_grouped_indices.append(indices)

    return fine_grouped_areas, fine_grouped_heights, fine_grouped_centres, fine_grouped_indices


