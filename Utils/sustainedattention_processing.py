import numpy as np
import math
from datetime import time
from scipy import signal
import pandas as pd

moving_average = lambda x, w: np.convolve(x, np.ones(w)/ w, 'same')
find_nearest_index = lambda array, value: np.abs(array - value).argmin()


def SpO2_postProcessing(SpO2, times=None, window_length=None, step_size=None):
    if times is None: times = np.arange(0, len(SpO2))

    # moving average over 5 points
    SpO2 = moving_average(SpO2, 5)
    # notch filter at 0.2hz
    bnotch, anotch = signal.iirnotch(0.2, 2)
    SpO2 = signal.filtfilt(bnotch, anotch, SpO2)

    fNIRS_mean = []
    fNIRS_min = []
    fNIRS_max = []
    if window_length is None: window_length = math.ceil(times[len(times) - 1] / 10)
    if step_size is None: step_size = window_length
    # gets slices of fNIRs data window_length long in seconds in steps of step_size
    # then gets min, max and mean of the window
    for i in np.arange(0, times[len(times) - 1], int(step_size)):
        start = find_nearest_index(times, i)
        end = find_nearest_index(times, i + window_length)
        slice = SpO2[start:end]
        fNIRS_mean.append(np.mean(slice, axis=0))
        fNIRS_min.append(np.amin(slice, axis=0))
        fNIRS_max.append(np.amax(slice, axis=0))
    window_times = np.arange(0, len(fNIRS_mean) * step_size, step_size)
    window_dataframe = pd.DataFrame()
    window_dataframe["Time"] = window_times
    window_dataframe["Min"] = fNIRS_min
    window_dataframe["Mean"] = fNIRS_mean
    window_dataframe["Max"] = fNIRS_max

    return SpO2, window_dataframe


def reactionTime_postProcessing(reaction, timestamps, SpO2_times):
    reaction_mean = []
    reaction_var = []
    find_nearest_index = lambda array, value: np.abs(array - value).argmin()
    # gets slices of reaction time data to match the timestamps of the fNIRs data
    # then gets the mean and variance of the window
    for i in range(0, len(SpO2_times) - 2):
        start = find_nearest_index(timestamps, SpO2_times[i])
        end = find_nearest_index(timestamps, SpO2_times[i] + 1)
        slice = reaction[start:end]
        if len(slice) == 0:
            mean = np.nan
            var = np.nan
        else:
            mean = np.mean(slice, axis=0)
            var = np.std(slice, axis=0)
        reaction_mean.append(mean)
        reaction_var.append(var)
    reaction_mean = np.array(reaction_mean)
    reaction_var = np.array(reaction_var)
    window_dataframe = pd.DataFrame()
    window_dataframe["Time"] = timestamps
    window_dataframe["Mean"] = reaction_mean
    window_dataframe["Var"] = reaction_var
    return window_dataframe
