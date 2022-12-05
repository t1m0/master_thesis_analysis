import numpy as np
from scipy.signal import argrelextrema

def velocity_peaks(df, column):
    sequence = df[column].values
    ilocs_min = argrelextrema(sequence, np.less_equal, order=3)[0]
    ilocs_max = argrelextrema(sequence, np.greater_equal, order=3)[0]
    return (len(ilocs_min) + len(ilocs_max))
