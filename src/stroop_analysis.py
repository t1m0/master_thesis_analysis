import matplotlib.pyplot as plt

from src.velocity_peaks import velocity_peaks
from src.spectral_arc_length import spectral_arc_length

def _calc_snr(df,column):
    df_copy = df.copy()
    df_copy[column+'_snr'] = (df[column+'_mean'] / df[column+'_std'])
    return df_copy

def stroop_feature_engineering(df):
    group_by_keys = ['age_group','subject','device', 'hand','uuid']
    aggregate_keys = ['x', 'y', 'z', 'mag']
    # standard deviation
    stroop_std_df = df.groupby(group_by_keys)[aggregate_keys].agg('std')
    stroop_std_df.rename(columns = {'x':'x_std', 'y':'y_std', 'z':'z_std', 'mag':'mag_std'}, inplace = True)
    # mean deviation
    stroop_mean_df = df.groupby(group_by_keys)[aggregate_keys].agg('mean')
    stroop_mean_df.rename(columns = {'x':'x_mean', 'y':'y_mean', 'z':'z_mean', 'mag':'mag_mean'}, inplace = True)
    # standard error to mean
    stroop_sem_df = df.groupby(group_by_keys)[aggregate_keys].agg('sem')
    stroop_sem_df.rename(columns = {'x':'x_sem', 'y':'y_sem', 'z':'z_sem', 'mag':'mag_sem'}, inplace = True)
    # peaks
    stroop_peak_df = df.groupby(group_by_keys)[aggregate_keys].agg(velocity_peaks)
    stroop_peak_df.rename(columns = {'x':'x_peaks', 'y':'y_peaks', 'z':'z_peaks', 'mag':'mag_peaks'}, inplace = True)
    # spectral arc length
    spectral_arc_length_df = df.groupby(group_by_keys)[aggregate_keys].agg(spectral_arc_length)
    spectral_arc_length_df.rename(columns = {'x':'x_sal', 'y':'y_sal', 'z':'z_sal', 'mag':'mag_sal'}, inplace = True)

    # total duration
    duration_df = df.groupby(group_by_keys)['duration'].agg(max)
    
    merged_df = stroop_std_df.merge(stroop_mean_df, on=group_by_keys)
    merged_df = merged_df.merge(stroop_sem_df, on=group_by_keys)
    merged_df = merged_df.merge(stroop_peak_df, on=group_by_keys)
    merged_df = merged_df.merge(spectral_arc_length_df, on=group_by_keys)
    merged_df = merged_df.merge(duration_df, on=group_by_keys)
    # snr
    for k in aggregate_keys:
        merged_df = _calc_snr(merged_df, k)
    return merged_df


def plot_stroop_stacceleration(subject_df, title):
    
    for click in subject_df['click_number'].unique():
        click_df = subject_df[subject_df['click_number']==click]
        plt.axvline(x = click_df['duration'].max(), color = 'b')
    
    plt.plot(subject_df['duration'].tolist(), subject_df['x'].tolist(), label = f"x", linestyle='solid')
    plt.plot(subject_df['duration'].tolist(), subject_df['y'].tolist(), label = f"y", linestyle='dashed')
    plt.plot(subject_df['duration'].tolist(), subject_df['z'].tolist(), label = f"z", linestyle='dotted')
    plt.plot(subject_df['duration'].tolist(), subject_df['mag'].tolist(), label = f"magnitude", linestyle='dashdot')
    
        
    plt.legend()
    plt.title(title)
    
    plt.show()