import numpy as np
import matplotlib.pyplot as plt

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

def _fourier_transformation(result):
    values = list(result['magnitude'].values())

    X = np.fft.fft(values)
    N = len(X)
    n = np.arange(N)

    duration = result['magnitude'][list(result['magnitude'].keys())[-1]]
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

def plot_fourier_transformation(result):
    x,y = _fourier_transformation(result)
    plt.figure(figsize = (12, 6))
    plt.plot(x, y, 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude')
    plt.show()

def plot_acceleration(result, title,dividers = []):
    plt.plot(result['x'].keys(), result['x'].values(), label = f"x", linestyle='solid')
    plt.plot(result['y'].keys(), result['y'].values(), label = f"y", linestyle='dashed')
    plt.plot(result['z'].keys(), result['z'].values(), label = f"z", linestyle='dotted')
    plt.plot(result['magnitude'].keys(), result['magnitude'].values(), label = f"magnitude", linestyle='dashdot')
    for divider in dividers:
        plt.axvline(x = divider, color = 'b')
    plt.legend()
    plt.title(title)
    plt.ylim([-4000, 4000])
    plt.show()