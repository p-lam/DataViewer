import librosa
import numpy as np
import matplotlib.pyplot as plt
from multitaper_toolbox.python.multitaper_spectrogram_python import multitaper_spectrogram, nanpow2db


# "Sleep Neurophysiological Dynamics Through the Lens of Multitaper Spectral Analysis"
# Michael J. Prerau, Ritchie E. Brown, Matt T. Bianchi, Jeffrey M. Ellenbogen, Patrick L. Purdon
# December 7, 2016 : 60-920
# DOI: 10.1152/physiol.00062.2015
def plot(mt_spectrogram, stimes, sfreqs):
    spect_data = mt_spectrogram
    clim = np.percentile(spect_data, [5, 95])  # Scale colormap from 5th percentile to 95th

    plt.figure("Multitaper Spectrogram", figsize=(10, 5))
    librosa.display.specshow(nanpow2db(mt_spectrogram), x_axis='time', y_axis='linear',
                             x_coords=stimes, y_coords=sfreqs, shading='auto', cmap="jet")
    plt.axhline(y=8, linestyle='--', linewidth=1.5, color='white')
    plt.axhline(y=12, linestyle='--', linewidth=1.5, color='white')
    plt.colorbar(label='Power (dB)')
    plt.xlabel("Time (HH:MM:SS)")
    plt.ylabel("Frequency (Hz)")


def plotSpectrogram(eeg, sr, window=[4, 1], res=1.5, cpu_cores=False, resample = True):
    # Set spectrogram params

    frequency_range = [0, 80]  # Limit frequencies from 0 to 25 Hz
    if resample:
        eeg = eeg[::int(sr / frequency_range[1]/2)]
        sr = sr / int(sr / frequency_range[1]/2)
    time_bandwidth = (window[0] * res) / 2  # Set time-half bandwidth
    num_tapers = int(time_bandwidth * 2) - 1  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    detrend_opt = 'constant'  # detrend each window by subtracting the average
    clim_scale = False  # do not auto-scale colormap

    spect, stimes, sfreqs = multitaper_spectrogram(eeg, sr, frequency_range, time_bandwidth, num_tapers, window,
                                                   detrend_opt=detrend_opt, multiprocess=True, cpus=cpu_cores,
                                                   clim_scale=clim_scale, plot_on=False)
    plot(spect, stimes, sfreqs)
