import numpy as np
from typing import Tuple, List
import trimesh
from warnings import warn

def generate_heights(
        reciprocal_lattice: np.ndarray,
        tiling: int | Tuple[int, int, int] = 3,
        bz_divisions: int = 100
) -> np.ndarray:
    """
    Generate a grid of heights in the reciprocal lattice.
    """
    
    if isinstance(tiling, int):
        tiling = (tiling, tiling, tiling)

    box_vectors = (reciprocal_lattice.T * tiling).T
    corner_vector = np.sum(box_vectors, axis=0)
    corner_dist = np.max(np.linalg.norm(corner_vector))
    step_size = np.min(np.linalg.norm(reciprocal_lattice, axis=1)) / bz_divisions
    heights = np.arange(-corner_dist/2, corner_dist/2, step_size)
    return heights


def calculate_sections(
        mesh: trimesh.Trimesh,
        normal: np.ndarray,
        heights: np.ndarray,
        origin: np.ndarray = np.zeros(3)
) -> List[trimesh.path.Path2D | None]:
    """
    Slice a mesh with a plane defined by a normal and a set of heights.
    """
    
    sections = mesh.section_multiplane(
        plane_origin=origin,
        plane_normal=normal,
        heights=heights
    )

    return sections


def get_section_data(
        sections: List[trimesh.path.Path2D | None],
        normal: np.ndarray,
        heights: np.ndarray,
        origin: np.ndarray = np.zeros(3)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the data from a list of sections.
    """

    areas = []
    area_heights = []
    area_3d_centres = []
    
    # Compute two orthogonal vectors on the plane from the normal
    normal = normal / np.linalg.norm(normal)  # Normalize the normal
    v1 = np.array([1, 0, 0]) if np.abs(normal[0]) < 0.9 else np.array([0, 1, 0])
    v1 = np.cross(normal, v1)
    v1 /= np.linalg.norm(v1)  # Normalize the first vector
    v2 = np.cross(normal, v1)  # Second vector orthogonal to both normal and v1

    for height, section in zip(heights, sections):
        if section is not None:
            
            if section.area == 0:
                continue

            areas_temp = []
            area_heights_temp = []
            area_3d_centres_temp = []

            section_3D = section.to_3D()

            for component in section_3D.discrete:
                area_3d_centres_temp.append(np.mean(component, axis=0))
                area_heights_temp.append(height)

                # Compute the area of the component
                component -= origin
                component_2d = np.dot(component, np.column_stack((v1, v2)))

                # use np.trapezoid to compute the area of the polygon
                areas_temp.append(np.abs(np.trapz(component_2d[:, 0], component_2d[:, 1])))

            areas.extend(areas_temp)
            area_heights.extend(area_heights_temp)
            area_3d_centres.extend(area_3d_centres_temp)

    return np.array(areas), np.array(area_heights), np.array(area_3d_centres)
            