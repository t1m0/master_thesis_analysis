import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.velocity_peaks import velocity_peaks
from src.spectral_arc_length import spectral_arc_length
from src.pandas_util import get_min_value_across_columns, get_max_value_across_columns,split_by_column
from src.plotting import box_plot_columns

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

def _plot_fourier_transformation_single(acceleration_df, title=""):
    x,y = _fourier_transformation(acceleration_df)
    plt.figure(figsize = (12, 6))
    plt.plot(x, y, 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.title('FFT '+title)
    plt.show()

def plot_fourier_transformation(df, title=""):
    if type(df) is list or type(df) is set:
        for single_df in df:
            _plot_fourier_transformation_single(single_df, title)
    else:
        _plot_fourier_transformation_single(df, title)

def _plot_acceleration_for_sub_plot(df, axis, label_suffix='', linestyle='solid'):
    x_label = 'x'+label_suffix
    y_label = 'y'+label_suffix
    z_label = 'z'+label_suffix
    axis[0].plot(df['duration'].tolist(), df['x'].tolist(), color='blue', label=x_label, linestyle=linestyle)
    axis[1].plot(df['duration'].tolist(), df['y'].tolist(), color='orange', label=y_label, linestyle=linestyle)
    axis[2].plot(df['duration'].tolist(), df['z'].tolist(), color='green', label=z_label, linestyle=linestyle)

def _add_legend_to_sub_plots(legends, label_suffix='', linestyle='solid'):
    legends[0]['x'+label_suffix] = Line2D([0], [0], color='blue', linestyle=linestyle)
    legends[1]['y'+label_suffix] = Line2D([0], [0], color='orange', linestyle=linestyle)
    legends[2]['z'+label_suffix] = Line2D([0], [0], color='green', linestyle=linestyle)

def _plot_acceleration_sub_plot(df, title,additional_plotting,ymin,ymax):

    figure, axis = plt.subplots(3, 1)
    figure.set_size_inches(20, 15)   

    legends = [{},{},{}]
    
    if title != None:
        figure.suptitle(title)

    for device_df in split_by_column(df, 'device'):
        hand = device_df['hand'].head(1).item()
        linestyle = 'solid' if hand == 'dominant' else 'dashed'
        label_suffix=' '+hand
        _add_legend_to_sub_plots(legends,label_suffix=label_suffix,linestyle=linestyle)
        for session_df in split_by_column(device_df, 'uuid'):
            _plot_acceleration_for_sub_plot(session_df, axis, linestyle=linestyle, label_suffix=label_suffix)
    
    for i in range(len(legends)):
        ax = axis[i]
        additional_plotting(df, ax=ax)
        ax.legend(list(legends[i].values()), list(legends[i].keys()))
        ax.set_ylim(ymin[i], ymax[i])

    plt.ylabel('acceleration')
    plt.xlabel('duration')
    plt.show()
    

def _plot_acceleration(df, label_suffix='', linestyle='solid'):
    x_label = 'x'+label_suffix
    y_label = 'y'+label_suffix
    z_label = 'z'+label_suffix
    plt.plot(df['duration'].tolist(), df['x'].tolist(), color='blue', label=x_label, linestyle=linestyle)
    plt.plot(df['duration'].tolist(), df['y'].tolist(), color='orange', label=y_label, linestyle=linestyle)
    plt.plot(df['duration'].tolist(), df['z'].tolist(), color='green', label=z_label, linestyle=linestyle)


def _add_legend_to_single_plots(legends, label_suffix='', linestyle='solid'):
    legends['x'+label_suffix] = Line2D([0], [0], color='blue', linestyle=linestyle)
    legends['y'+label_suffix] = Line2D([0], [0], color='orange', linestyle=linestyle)
    legends['z'+label_suffix] = Line2D([0], [0], color='green', linestyle=linestyle)

def _plot_acceleration_single_plot(df,title, additional_plotting,ymin,ymax):

    fig = plt.gcf()
    fig.set_size_inches(30, 7.5)
    plt.ylim(ymin,ymax)

    legends = {}
    for device_df in split_by_column(df, 'device'):
        hand = device_df['hand'].head(1).item()
        linestyle = 'solid' if hand == 'dominant' else 'dashed'
        label_suffix=' '+hand
        _add_legend_to_single_plots(legends,label_suffix=label_suffix,linestyle=linestyle)
        for session_df in split_by_column(device_df, 'uuid'):
            _plot_acceleration(session_df, linestyle=linestyle, label_suffix=label_suffix)
    
    plt.legend(list(legends.values()), list(legends.keys()))

    additional_plotting(df)
    
    if title != None:
        plt.title(title)
    plt.ylabel('acceleration')
    plt.xlabel('duration')
    plt.show()

def _plot_acceleration_single(df, title, subplots, additional_plotting,ymin,ymax):
    if subplots:
        _plot_acceleration_sub_plot(df, title,additional_plotting,ymin,ymax)
    else:
        _plot_acceleration_single_plot(df, title, additional_plotting,ymin,ymax)

def _extract_limit_subplot(df, columns, padding):
    min_values = [0,0,0]
    max_values = [0,0,0]
    if type(df) is list or type(df) is set:
        for single_df in df:
            for i in range(len(columns)):
                column = columns[i]
                current_min_value = get_min_value_across_columns(single_df, [column], padding)
                current_max_value = get_max_value_across_columns(single_df, [column], padding)
                if current_min_value < min_values[i]:
                    min_values[i] = current_min_value
                if current_max_value > max_values[i]:
                    max_values[i] = current_max_value
    else:
        for i in range(len(columns)):
            column = columns[i]
            min_value = get_min_value_across_columns(df, [column], padding)
            max_value = get_max_value_across_columns(df, [column], padding)
            min_values[i] = min_value
            max_values[i] = max_value
    return min_values, max_values


def _extract_limit_single(df, columns, padding):
    min_value = 0
    max_value = 0
    if type(df) is list or type(df) is set:
        for single_df in df:
            current_min_value = get_min_value_across_columns(single_df, columns, padding)
            current_max_value = get_max_value_across_columns(single_df, columns, padding)
            if current_min_value < min_value:
                min_value = current_min_value
            if current_max_value > max_value:
                max_value = current_max_value
    else:
        min_value = get_min_value_across_columns(df, columns, padding)
        max_value = get_max_value_across_columns(df, columns, padding)
    return min_value, max_value

def _extrac_limits(df, subplots):
    columns = ['x','y','z']
    padding = 0.1
    return _extract_limit_subplot(df, columns, padding) if subplots else _extract_limit_single(df, columns, padding)

def plot_acceleration(df, title=None, subplots=True, additional_plotting=lambda df, ax=None:()):
    min_value, max_value = _extrac_limits(df, subplots)
    if type(df) is list or type(df) is set:
        index = 0
        for single_df in df:
            if title != None:
                local_title = title[index]
            else:
                local_title = None
            _plot_acceleration_single(single_df,subplots=subplots, title=local_title, additional_plotting=additional_plotting,ymin=min_value,ymax=max_value)
            index+=1
    else:
        _plot_acceleration_single(df,subplots=subplots, title=title, additional_plotting=additional_plotting,ymin=min_value,ymax=max_value)
    
    
    

def plot_feature_columns(df, field, class_key='age_group'):
    features = ['x_'+field,'y_'+field,'z_'+field,'mag_'+field]
    box_plot_columns(df,class_key,features)
