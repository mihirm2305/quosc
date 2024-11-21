from scipy.interpolate import interp1d
import numpy as np
import pandas as pd

def interpolate_across_bands(
        df: pd.DataFrame,
        interpolation_fineness: int | float = 10
):
    # now we can interpolate for a finer grid
    df_interp = pd.DataFrame(columns=['angle', 'frequency', 'group', 'band', 'curvature', 'count', 'mass', 'df_dtheta'])

    for band in df['band'].unique():
        df_band1 = df[df['band'] == band]
        df_interp_band1 = pd.DataFrame(columns=['angle', 'frequency', 'group', 'curvature', 'count', 'mass', 'df_dtheta'])
            
        for group in df_band1['group'].unique():
            if np.isnan(group):
                continue
            # if group smaller than 5 points, skip
            if len(df_band1[df_band1['group'] == group]) < 4:
            
                df_group = df_band1[df_band1['group'] == group]
                angle_min = df_group['angle'].min()
                angle_max = df_group['angle'].max()
                num_spacings = (np.rad2deg(angle_max - angle_min) * interpolation_fineness) + 1
                angles_interp = np.linspace(angle_min, angle_max, int(np.round(num_spacings)))

                # interpolate frequency
                freq_interp = np.interp(angles_interp, df_group['angle'], df_group['frequency'])

                # interpolate curvature
                curv_interp = np.interp(angles_interp, df_group['angle'], df_group['curvature'])

                # all the counts are the same, so no need to interpolate - just copy the value from the original dataframe
                count = df_group['count'].values[0]
                count_interp = np.full(len(angles_interp), count)

                # interpolate mass
                mass_interp = np.interp(angles_interp, df_group['angle'], df_group['mass'])

                # calculate df_dtheta by taking the derivative of the interpolated frequency
                # if there is only one point, set df_dtheta to nan
                if len(angles_interp) == 1:
                    df_dtheta_interp = np.nan
                else:
                    df_dtheta_interp = np.gradient(freq_interp, angles_interp)

                df_group_interp = pd.DataFrame({'angle': angles_interp, 'frequency': freq_interp, 'group': group, 'curvature': curv_interp, 'count': count_interp, 'mass': mass_interp, 'df_dtheta': df_dtheta_interp})
                df_interp_band1 = pd.concat([df_interp_band1, df_group_interp])


            else:
                df_group = df_band1[df_band1['group'] == group]
                angle_min = df_group['angle'].min()
                angle_max = df_group['angle'].max()
                num_spacings = (np.rad2deg(angle_max - angle_min) * 10) + 1
                angles_interp = np.linspace(angle_min, angle_max, int(np.round(num_spacings)))

                # interpolate frequency
                f = interp1d(df_group['angle'], df_group['frequency'], kind='cubic')
                freq_interp = f(angles_interp)
                
                # interpolate curvature
                f = interp1d(df_group['angle'], df_group['curvature'], kind='cubic')
                curv_interp = f(angles_interp)

                # all the counts are the same, so no need to interpolate - just copy the value from the original dataframe
                count = df_group['count'].values[0]
                count_interp = np.full(len(angles_interp), count)

                # interpolate mass
                f = interp1d(df_group['angle'], df_group['mass'], kind='cubic')
                mass_interp = f(angles_interp)

                # calculate df_dtheta by taking the derivative of the interpolated frequency
                df_dtheta_interp = np.gradient(freq_interp, angles_interp)

                df_group_interp = pd.DataFrame({'angle': angles_interp, 'frequency': freq_interp, 'group': group, 'curvature': curv_interp, 'count': count_interp, 'mass': mass_interp, 'df_dtheta': df_dtheta_interp})
                df_interp_band1 = pd.concat([df_interp_band1, df_group_interp])
                
        # store in the original dataframe
        df_interp_band1['band'] = band
        df_interp = pd.concat([df_interp, df_interp_band1])


    df_interp.head()

    return df_interp