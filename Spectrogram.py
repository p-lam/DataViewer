from multitaper_toolbox.python.multitaper_spectrogram_python import multitaper_spectrogram
# "Sleep Neurophysiological Dynamics Through the Lens of Multitaper Spectral Analysis"
# Michael J. Prerau, Ritchie E. Brown, Matt T. Bianchi, Jeffrey M. Ellenbogen, Patrick L. Purdon
# December 7, 2016 : 60-920
# DOI: 10.1152/physiol.00062.2015


def plotSpectrogram(eeg, sr, window=[4, 1], res=1.5, cpu_cores=False):
    # Set spectrogram params
    frequency_range = [0, 80]  # Limit frequencies from 0 to 25 Hz
    time_bandwidth = (window[0] * res) / 2  # Set time-half bandwidth
    num_tapers = int(time_bandwidth * 2) - 1  # Set number of tapers (optimal is time_bandwidth*2 - 1)
    detrend_opt = 'constant'  # detrend each window by subtracting the average
    clim_scale = False  # do not auto-scale colormap

    spect, stimes, sfreqs = multitaper_spectrogram(eeg, sr, frequency_range, time_bandwidth, num_tapers, window,
                                                   detrend_opt=detrend_opt, multiprocess=True, cpus=cpu_cores,
                                                   clim_scale=clim_scale)
