import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Iterable
from ..initialisation import load_data
from ..initialisation.band_calc import generate_kpoints

# function to plot the distribution of energies in each band
def plot_energy_distribution(
        energies: np.ndarray,
        filepath: str = 'interpolation.bt2',
        fermi_energy: Optional[float] = None,
        **kwargs
):
    """
    Plot the distribution of energies in each band
    """

    # Load the BoltzTraP2 data if Fermi energy is not provided
    if fermi_energy is None:
        data_bt = load_data.read_boltztrap_interpolation(filepath)
        fermi_energy = load_data.get_fermi_energy(data_bt)


    # flatten the non-band dimensions of the energies array
    energies = energies.reshape(energies.shape[0], -1)

    plt.figure(**kwargs)
    plt.boxplot(energies.T, whis=[0,100], tick_labels=np.arange(0, energies.shape[0]))
    plt.axhline(fermi_energy, color='r', linestyle='--')
    plt.xlabel('Band index')
    plt.ylabel('Energy (eV)')
    plt.title('Band energies')
    plt.show()

# function to plot the band structure (3D k-points coloured by energy)
def plot_band_structure(
        band_indices: Iterable[int],
        energies: np.ndarray,
        reciprocal_lattice: np.ndarray,
        kpoints: Optional[np.ndarray] = None,
        fermi_energy: Optional[float] = None,
        **kwargs
):
    """
    Plot the band structure
    """

    if kpoints is None:
        kpoints = generate_kpoints(n_kpoints=energies.shape[1:])

    kpoints = np.dot(kpoints, reciprocal_lattice)

    for band_index in band_indices:

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
        kpoints[:, 0],
        kpoints[:, 1],
        kpoints[:, 2],
        c=energies[band_index, :],
        cmap='viridis'
        )
        ax.set_xlabel('$k_x$ ($\\AA^{-1}$)')
        ax.set_ylabel('$k_y$ ($\\AA^{-1}$)')
        ax.set_zlabel('$k_z$ ($\\AA^{-1}$)')
        ax.set_title('Band structure')

        # add colour bar
        cbar = plt.colorbar(ax.collections[0], ax=ax, orientation='vertical')
        cbar.set_label('Energy (eV)')

        # add title with Fermi energy and band index
        if fermi_energy is None:
            title = f'Band {band_index}'
        else:
            title = f'Band {band_index} ($E_F$ = {fermi_energy:.2f} eV)'
        
        plt.title(title)

        plt.show()