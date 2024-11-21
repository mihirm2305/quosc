import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
from scipy.fft import rfft, rfftfreq
from joblib import Parallel, delayed
import seaborn as sns

e = 1.60217662e-19  # C
hbar = 1.0545718e-34  # J s
m_e = 9.10938356e-31  # kg
const = e**2.5 / (2**0.5 * m_e * hbar**0.5 * np.pi**2.5)


# R_T function
def R_T(X: float) -> float:
    if X == 0:
        return 1
    if np.abs(X) > 1e2:
        return 0
    else:
        return X / np.sinh(X)
    
R_T = np.vectorize(R_T)


# X_p function
def X_p(
        p: int | np.ndarray,
        m_star: float,
        T: float,
        B: float
) -> float | np.ndarray:
    
    k_B = 1.38064852e-23  # J/K
    hbar = 1.0545718e-34  # J s
    e = 1.60217662e-19  # C

    return (2 * np.pi**2 * k_B * T * m_star * p) / (e * B * hbar)


# R_D function
def R_D(
        p: int | np.ndarray,
        m_b: float,
        T_D: float,
        B: float
) -> float | np.ndarray:
    
    e = 1.60217662e-19  # C
    hbar = 1.0545718e-34  # J s
    k_B = 1.38064852e-23  # J/K

    return np.exp(- (2 * np.pi**2 * p * m_b * k_B * T_D) / (e * B * hbar))


# R_S function
def R_S(
        p: int | np.ndarray,
        m_s: float
) -> float | np.ndarray:
    
    m_e = 9.10938356e-31  # kg
    g = 2

    return np.cos( (p * np.pi * g * m_s) / (2 * m_e))


# p_sum function
def p_sum(
        m_star: float,
        m_b: float,
        m_s: float,
        phase: float,
        T: float,
        T_D: float,
        B: float,
        F: float,
        phi_B: float,
        sum_range: int = 10
) -> float | np.ndarray:
    
    p = np.arange(1, sum_range + 1)
    
    x = X_p(p, m_star, T, B)
    R_T_val = R_T(x)
    R_D_val = R_D(p, m_b, T_D, B)
    R_S_val = R_S(p, m_s)

    sum_elements = p**(-1.5) * R_T_val * R_D_val * R_S_val * np.cos(2 * np.pi * p * ((F / B) - 0.5 + (phi_B / (2 * np.pi))) + phase) # need to add +/- pi/4 later
    
    return np.sum(sum_elements)


# torque_single_angle function
def torque_single_angle_single_B(
        df: pd.DataFrame,
        angle: float,
        T: float,
        T_D: float,
        B: float,
        phi_B: float,
        sum_range_p: int = 10,
        
) -> float | np.ndarray:
    
    df['phase'] = 0.
    df.loc[(df['mass'] < 0) & (df['curvature'] < 0), 'phase'] = np.pi / 4
    df.loc[(df['mass'] < 0) & (df['curvature'] > 0), 'phase'] = - np.pi / 4
    
    # e = 1.60217662e-19  # C
    # hbar = 1.0545718e-34  # J s
    m_e = 9.10938356e-31  # kg
    # const = e**2.5 / (2**0.5 * m_e * hbar**0.5 * np.pi**2.5)

    torque = 0

    # look for angles within 0.01 degrees, i.e. 1e-4 radians of the target angle
    df_angle = df[(df['angle'] >= angle - 1e-4) & (df['angle'] <= angle + 1e-4)]

    if df_angle.empty:
        print(f'No data found for angle {angle}')
        return np.nan


    for row in df_angle.itertuples(index=False):

        if np.isnan(row.mass) or np.isnan(row.frequency) or np.isnan(row.curvature) or np.isnan(row.count):
            # print(f'Skipping row due to NaN values: {row}')
            continue

        m_star = np.abs(row.mass) * m_e
        m_b = np.abs(row.mass) * m_e
        m_s = np.abs(row.mass) * m_e
        phase = row.phase
        freq = row.frequency
        curv = row.curvature * 1e-20
        df_dtheta = row.df_dtheta
        count = row.count
        try:
            add = count * p_sum(m_star, m_b, m_s, phase, T, T_D, B, freq, phi_B, sum_range_p) * df_dtheta * (np.abs(curv)**(-0.5))# * const
        except Exception as e:
            print(f'Error calculating add for row {row}: {e}')
            continue

        if not np.isnan(add):
            torque += add
        else:
            # print(f'add is NaN for row {row}')
            continue

    return torque


# angle_sweep function
def angle_sweep(
        df: pd.DataFrame,
        T: float,
        T_D: float,
        B: float,
        phi_B: float,
        sum_range_p: int = 10,
        
) -> float | np.ndarray:
    
    # e = 1.60217662e-19  # C
    # hbar = 1.0545718e-34  # J s
    # m_e = 9.10938356e-31  # kg
    # const = e**2.5 / (2**0.5 * m_e * hbar**0.5 * np.pi**2.5)

    angles = np.unique(df['angle'])
    torques = np.zeros_like(angles)

    for i, angle in enumerate(angles):
        torques[i] = torque_single_angle_single_B(df, angle, T, T_D, B, phi_B, sum_range_p)

    return angles, torques


# B_sweep function
def B_sweep(
        df: pd.DataFrame,
        B_range: Tuple[float, float],
        angle: float,
        T: float,
        T_D: float,
        phi_B: float,
        num_B: int = 1024,
        sum_range_p: int = 10,

):
    inv_B_vals = np.linspace(1 / B_range[1], 1 / B_range[0], num_B)
    B_vals = 1 / inv_B_vals  
    torques = np.zeros_like(B_vals)

    for i, B in enumerate(B_vals):
        torques[i] = torque_single_angle_single_B(df, angle, T, T_D, B, phi_B, sum_range_p)

    return B_vals, torques


# B_angle_sweep function
def B_angle_sweep(
        df: pd.DataFrame,
        B_range: Tuple[float, float],
        T: float,
        T_D: float,
        phi_B: float,
        num_B: int = 1024,
        sum_range_p: int = 10,
):
    B_vals = np.linspace(B_range[0], B_range[1], num_B)

    # angles = np.unique(df['angle'])
    # but add a tolerance for numerical errors
    angles = np.unique(df['angle'])
    torques = np.zeros((len(angles), len(B_vals)))

    for i, angle in enumerate(angles):
        torques[i] = B_sweep(df, B_range, angle, T, T_D, phi_B, num_B, sum_range_p)[1]

    return angles, B_vals, torques


# FFT functions
from joblib import Parallel, delayed
from scipy.signal.windows import blackman
from scipy.fft import rfft, rfftfreq
from typing import Tuple

def torque_fft_single_angle(
    df: pd.DataFrame,
    B_range: Tuple[float, float],
    angle: float,
    T: float,
    T_D: float,
    phi_B: float,
    num_B: int = 2048,
    sum_range_p: int = 10,
):

    B_vals, torques = B_sweep(df, B_range, angle, T, T_D, phi_B, num_B, sum_range_p)

    x = 1/B_vals
    y = torques

    # order the x and y data by x
    torques = torques[np.argsort(x)]
    y = y[np.argsort(x)]
    x = x[np.argsort(x)]

    # apply window
    window = blackman(len(y))
    y_windowed = y * window

    # take fft
    yf = rfft(y_windowed)
    xf = rfftfreq(len(x), x[1] - x[0])

    return xf, np.abs(yf), torques


def torque_fft_multi_angle(
        df: pd.DataFrame, 
        angle_step: int = 10, 
        B_range: Tuple[int,int] = (5,45), 
        T: float = 2,
        T_D: float = 1,
        phi_B: float = np.pi,
        num_B: int = 2048,
        n_jobs: int = -1):
    """
    Calculates the FFT of torque data for multiple angles in parallel.

    Args:
        df (pd.DataFrame): DataFrame containing the data with 'angle' column.
        angle_step (int, optional): Step size for selecting angles. Defaults to 50.
        n_jobs (int, optional): Number of jobs for parallel processing. Defaults to -1 (all CPUs).

    Returns:
        tuple: A tuple containing (freqs, amps, torques) arrays.
    """

    angles = np.unique(df['angle'])[0::angle_step]

    fft_data = Parallel(n_jobs=n_jobs)(
        delayed(torque_fft_single_angle)(df, B_range, angle, T, T_D, phi_B, num_B) for angle in angles
    )

    amps = np.array([data[1] for data in fft_data])
    amps /= np.max(amps)  # Normalize amplitudes
    freqs = np.array([data[0] for data in fft_data])
    torques = np.array([data[2] for data in fft_data])

    return freqs, amps, torques, angles


def plot_fft_heatmap(freqs, amps, angles):
    """
    Generates a heatmap plot of FFT amplitudes.

    Args:
        freqs (np.ndarray): Array of frequencies.
        amps (np.ndarray): Array of normalized FFT amplitudes.
        angles (np.ndarray): Array of angles in radians.
    """

    cmap = sns.color_palette("YlOrBr", as_cmap=True)
    angles_deg = np.rad2deg(angles)

    plt.figure(figsize=(12, 10), facecolor='none')
    plt.imshow(
        amps.T, 
        aspect='auto', 
        cmap='twilight', 
        extent=[angles_deg[0], angles_deg[-1], freqs[0][-1] * 1e-3, freqs[0][0] * 1e-3] 
    )

    # Styling
    plt.xticks(fontsize=16, color='white')
    plt.yticks(fontsize=16, color='white')

    plt.xlabel('Angle (degrees)', fontsize=16, color='white')
    plt.ylabel('Frequency (T)', fontsize=16, color='white')

    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=16, colors='white')

    plt.gca().tick_params(axis='x', colors='white')
    plt.gca().tick_params(axis='y', colors='white')

    plt.gca().spines['bottom'].set_color('white')
    plt.gca().spines['top'].set_color('white')
    plt.gca().spines['right'].set_color('white')
    plt.gca().spines['left'].set_color('white')

    plt.gca().set_facecolor('none')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.show()
