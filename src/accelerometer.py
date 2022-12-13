import numpy as np
import matplotlib.pyplot as plt

from src.velocity_peaks import velocity_peaks
from src.spectral_arc_length import spectral_arc_length

def _fourier_transformation(acceleration_df):
    values = acceleration_df['mag'].tolist()

    X = np.fft.fft(values)
    N = len(X)
    n = np.arange(N)

    duration = acceleration_df['duration'].max()
    sample_rate = (N / duration ) * 1000

    sr = 1 / sample_rate
    T = N/sr
    freq = n/T 

    # Get the one-sided specturm
    n_oneside = N//2
    # get the one side frequency
    f_oneside = list(freq[:n_oneside])

    spectrum = list(np.abs(X[:n_oneside]))
    f_oneside.pop(0)
    spectrum.pop(0)
    return f_oneside, spectrum

def _calc_snr(df,column):
    df_copy = df.copy()
    df_copy[column+'_snr'] = (df[column+'_mean'] / df[column+'_std'])
    return df_copy

def calc_magnitude(x,y,z):
    return np.sqrt(pow(x,2) + pow(y,2) + pow(z,2))

def process_accelerations(accelerations, start_time):
    if len(accelerations) <= 0: 
        return None
    x = {}
    y = {}
    z = {}
    magnitude = {}
    x_raw = {}
    y_raw = {}
    z_raw = {}
    magnitude_raw = {}
    for acceleration in accelerations:
        time_stamp = acceleration['timeStamp']
        local_x = acceleration['xAxis']
        local_y = acceleration['yAxis']
        local_z = acceleration['zAxis']
        magnitude_local = calc_magnitude(local_x,local_y,local_z)
        ms_since_start = (time_stamp - start_time)
        magnitude_raw[ms_since_start] = magnitude_local
        x_raw[ms_since_start] = local_x
        y_raw[ms_since_start] = local_y
        z_raw[ms_since_start] = local_z
    x_mean = np.mean(list(x_raw.values()))
    y_mean = np.mean(list(y_raw.values()))
    z_mean = np.mean(list(z_raw.values()))

    for timeStamp in y_raw.keys():
        x[timeStamp] = x_raw[timeStamp] - x_mean
        y[timeStamp] = y_raw[timeStamp] - y_mean
        z[timeStamp] = z_raw[timeStamp] - z_mean
        magnitude[timeStamp] = calc_magnitude(x[timeStamp],y[timeStamp],z[timeStamp])

    return {
        'x':x,
        'y':y,
        'z':z,
        'magnitude':magnitude,
        'x_raw':x_raw,
        'y_raw':y_raw,
        'z_raw':z_raw,
        'magnitude_raw':magnitude
    }



def accelerometer_feature_engineering(df):
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

def plot_fourier_transformation(acceleration_df, title=""):
    x,y = _fourier_transformation(acceleration_df)
    plt.figure(figsize = (12, 6))
    plt.plot(x, y, 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.title('FFT '+title)
    plt.show()

def plot_stacceleration(subject_df, title):
    
    plt.plot(subject_df['duration'].tolist(), subject_df['x'].tolist(), label = f"x", linestyle='solid')
    plt.plot(subject_df['duration'].tolist(), subject_df['y'].tolist(), label = f"y", linestyle='dashed')
    plt.plot(subject_df['duration'].tolist(), subject_df['z'].tolist(), label = f"z", linestyle='dotted')
    plt.plot(subject_df['duration'].tolist(), subject_df['mag'].tolist(), label = f"magnitude", linestyle='dashdot')
    
        
    plt.legend()
    plt.title(title)
#    plt.ylim([-4000, 4000])
    plt.show()
