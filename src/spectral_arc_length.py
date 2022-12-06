import numpy as np

def spectral_arc_length(series):

    sampling_frequency = 30
    zero_padding=4
    max_cut_off_frequency=10.0
    amplitude_threshold=0.05
    
    zeros_to_be_padded = int(pow(2, np.ceil(np.log2(len(series))) + zero_padding))

    frequency = np.arange(0, sampling_frequency, sampling_frequency/zeros_to_be_padded)
    
    normalized_mag_spectrum = abs(np.fft.fft(series, zeros_to_be_padded))
    normalized_mag_spectrum = normalized_mag_spectrum/max(normalized_mag_spectrum)

    # Indices to choose only the spectrum within the given cut off frequency Fc.
    # NOTE: This is a low pass filtering operation to get rid of high frequency
    # noise from affecting the next step (amplitude threshold based cut off for
    # arc length calculation).
    fc_inx = ((frequency <= max_cut_off_frequency)*1).nonzero()
    f_sel = frequency[fc_inx]
    Mf_sel = normalized_mag_spectrum[fc_inx]

    # Choose the amplitude threshold based cut off frequency.
    # Index of the last point on the magnitude spectrum that is greater than
    # or equal to the amplitude threshold.
    inx = ((Mf_sel >= amplitude_threshold)*1).nonzero()[0]
    fc_inx = range(inx[0], inx[-1]+1)
    f_sel = f_sel[fc_inx]
    Mf_sel = Mf_sel[fc_inx]
    
    arc_length = -sum(np.sqrt(pow(np.diff(f_sel)/(f_sel[-1] - f_sel[0]), 2) +
                           pow(np.diff(Mf_sel), 2)))
    return arc_length