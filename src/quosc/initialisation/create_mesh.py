import numpy as np
import trimesh
from typing import Tuple
from skimage import measure

def tile_mesh(
        egrid: np.ndarray,
        tiling: int | Tuple[int, int, int] = 3,
        endpoints_included: bool = True
) -> np.ndarray:
    """
    Tile a mesh grid to create a larger mesh grid.

    Args:
        egrid: 3D array of energy values.
        tiling: Number of times to tile the mesh grid in each direction.
        endpoint: Whether to include the edge points of the mesh grid.

    Returns:
        3D array of energy values with the mesh grid tiled.
    """
    if isinstance(tiling, int):
        tiling = (tiling, tiling, tiling)

    if endpoints_included:
        # remove the edge points (and then re-add them later)
        egrid_temp = egrid[:,0:-1, 0:-1, 0:-1]
    else:
        egrid_temp = egrid

    egrid_tiled = np.tile(egrid_temp, (1,) + tiling)

    # Add the edge points back in
    egrid_tiled = np.pad(egrid_tiled, ((0, 0), (0, 1), (0, 1), (0, 1)), mode='wrap')

    return egrid_tiled

def create_trimesh(
        band_index: int,
        energies_mesh: np.ndarray,
        reciprocal_lattice: np.ndarray,
        fermi_energy,
        tiling=3
) -> trimesh.Trimesh:
    """
    Creates a 3D mesh from energy data, applies scaling and translation to match reciprocal lattice vectors,
    and shifts the mesh by half of its shape.

    Args:
        band_index: Index of the energy band to use.
        energies_mesh: 3D array of energy values.
        reciprocal_lattice: 3x3 array representing the reciprocal lattice vectors.
        fermi_energy: Fermi energy level.

    Returns:
        trimesh.Trimesh object representing the generated mesh.
    """

    vertices, faces, _, _ = measure.marching_cubes(energies_mesh[band_index], fermi_energy)

    # Create mesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Calculate the translation vector to shift by half the shape
    shape = np.array(energies_mesh[band_index].shape, dtype=float)
    shape -= 1
    translation_vector = shape
    # translation_vector -= 1
    translation_vector = -0.5 * translation_vector

    # Construct the 4x4 homogeneous transformation matrix for translation
    homogeneous_translation = np.eye(4)
    homogeneous_translation[:3, 3] = translation_vector

    # Construct the 4x4 homogeneous transformation matrix for scaling
    tiling = np.array(tiling, dtype=float)
    shape /= tiling
    shape = shape[:, np.newaxis]
    scale = reciprocal_lattice.T / shape 
    homogeneous_scale = np.eye(4) 
    homogeneous_scale[:3, :3] = scale

    # Apply both transformations (translation first, then scaling)
    mesh.apply_transform(homogeneous_translation)
    mesh.apply_transform(homogeneous_scale)

    return mesh
