import numpy as np
import pandas as pd

def group_across_angles(
        df: pd.DataFrame,
        rel_freq_tolerance: float = 0.3,
        abs_freq_tolerance: float = 10,
        rel_curv_tolerance: float = 0.8,
        abs_curv_tolerance: float = 1e4,
        rel_mass_tolerance: float = 0.8,
        abs_mass_tolerance: float = 0.1,
) -> pd.DataFrame:

    df.dropna()
    df['df_dtheta'] = np.nan
    df['group'] = np.nan

    # go through angles and group similar points for later interpolation
    for band in df['band'].unique():

        df_band = df[df['band'] == band]

        angles = df_band['angle'].unique()
        angles = np.sort(angles)
        no_groups = 0

        for i, angle in enumerate(angles):

            # if i > 1:
            #     break

            condition = (df_band['angle'] == angle)
            if i == 0:
                for j, row in df_band[condition].iterrows():
                    df_band.loc[j, 'group'] = no_groups + 1
                    no_groups += 1

            # elif i < len(angles):
            else:
                condition_prev = (df_band['angle'] == angles[i-1])
                delta_angle = angle - angles[i-1]
                # print(df_band[condition_prev])
                def predict_freq(row):
                    if np.isnan(row['df_dtheta']):
                        return row['frequency']
                    else:
                        return row['frequency'] + row['df_dtheta'] * delta_angle
                        
                predicted_freqs = df_band[condition_prev].apply(predict_freq, axis=1)

                # store predicted frequencies
                for j, row in df_band[condition_prev].iterrows():
                    df_band.loc[j, 'predicted_freq'] = predicted_freqs[j]

                for j, row in df_band[condition].iterrows():

                    freq = row['frequency']
                    curv = row['curvature']
                    count = row['count']
                    mass = row['mass']

                    # print(f'angle: {angle}, freq: {freq}, curv: {curv}, count: {count}')


                    
                    # additional filter for similar frequency, curvature, and count
                    similar = (np.abs(df_band[condition_prev]['predicted_freq'] - freq) < rel_freq_tolerance * np.abs(freq) + abs_freq_tolerance) & \
                                (np.abs(df_band[condition_prev]['curvature'] - curv) < rel_curv_tolerance * np.abs(curv) + abs_curv_tolerance) & \
                                (df_band[condition_prev]['count'] == count) & \
                                (np.abs(df_band[condition_prev]['mass'] - mass) < rel_mass_tolerance * np.abs(mass) + abs_mass_tolerance)
                    
                    # print(similar)
                    
                    # if no similar points found, create new group
                    if not similar.any():
                        df_band.loc[j, 'group'] = no_groups + 1
                        no_groups += 1
                        # print(f'New group number {no_groups}')
                    # elif similar.sum() ==1
                    elif similar.sum() == 1:
                        df_band.loc[j, 'group'] = df_band[condition_prev][similar].iloc[0]['group']
                        # print(f'Group number {df_band[condition_prev][similar].iloc[0]["group"]}')
                    else:
                        # choose the group with the closest frequency
                        freq_diff = np.abs(df_band[condition_prev][similar]['predicted_freq'] - freq)
                        min_idx = freq_diff.idxmin()
                        df_band.loc[j, 'group'] = df_band[condition_prev][similar].loc[min_idx, 'group']
                
                # if there are duplicate groups for this angle, only keep the one with the closest frequency, and start a new group for the rest
                groups = df_band[condition]['group'].unique()

                for group in groups:
                    group_condition = (df_band['group'] == group) & (df_band['angle'] == angle)
                    if len(df_band[group_condition]) > 1:
                        
                        # find frequency of previous angle (same group)
                        freq_prev = df_band[(df_band['group'] == group) & (df_band['angle'] == angles[i-1])]['predicted_freq'].values[0]
                        freq_diff = np.abs(df_band[group_condition]['frequency'] - freq_prev)
                        min_idx = freq_diff.idxmin()

                        # start new group for the rest
                        for j, row in df_band[group_condition].iterrows():
                            if j != min_idx:
                                df_band.loc[j, 'group'] = no_groups + 1
                                no_groups += 1
                            


        for i, angle in enumerate(angles):

            condition = (df_band['angle'] == angle)
            groups = df_band[condition]['group'].unique()

            if i == len(angles) - 1:
                continue

            for group in groups:
                # if the group exists in the previous and next angles, store dF/dtheta
                if group in df_band[(df_band['angle'] == angles[i-1])]['group'].values and group in df_band[(df_band['angle'] == angles[i+1])]['group'].values:
                    freq_prev = df_band[(df_band['angle'] == angles[i-1]) & (df_band['group'] == group)]['frequency'].values[0]
                    freq_next = df_band[(df_band['angle'] == angles[i+1]) & (df_band['group'] == group)]['frequency'].values[0]
                    delta_freq = freq_next - freq_prev
                    delta_angle = angles[i+1] - angles[i-1]
                    df_band.loc[df_band[condition]['group'].index, 'df_dtheta'] = delta_freq / delta_angle
                else:
                    df_band.loc[df_band[condition]['group'].index, 'df_dtheta'] = np.nan
                    
                    # store in the original dataframe
                    df.loc[df['band'] == band] = df_band

    return df
