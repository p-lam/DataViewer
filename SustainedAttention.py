import json
import math
import os
import tkinter
from datetime import datetime

from scipy.signal import butter, filtfilt

from scipy import signal

import ChartsTabPanel

print(".", end="")

import warnings

print(".", end="")
from collections import deque

print(".", end="")
from tkinter.filedialog import askopenfilename

print(".", end="")
import matplotlib.pyplot as plt

print(".", end="")
import numpy as np

print(".", end="")
import pandas as pd

print(".", end="")
from Spectrogram import plotSpectrogram

print(".", end="")
print("Libraries loaded ▐")


def getEOH(path):
    tmp = open(path)
    eoh = 0
    for line in tmp.readlines():
        eoh += 1
        if line.find("# EndOfHeader") >= 0:
            return eoh
    return 0


def EEGTransferFunction(nparr):
    nparr = nparr / 65536
    nparr = nparr - .5
    nparr = nparr * 3
    nparr = nparr / 40000
    nparr = nparr * 1000000
    return nparr


def readBioSignalFile(path):
    print(f"Loading data from {path}")
    eoh = getEOH(path)
    eoh += 10  # skips first 10 lines
    data = pd.read_csv(path, skiprows=eoh, header=None, sep="\t", usecols=[0, 2, 3, 4],
                       names=["Time", "EEG", "fNIRS1", "fNIRS2"])
    t0 = 0
    try:
        file1 = open(path, 'r')
        jsonText = file1.readlines()[1][2:]
        headerJson = json.loads(jsonText)
        headerJson = headerJson[list(headerJson)[0]]
        fs = headerJson["sampling rate"]
        tstring = headerJson["time"]
        date_object = datetime.strptime("2000:" + tstring, "%Y:%H:%M:%S.%f")
        t0 = date_object.timestamp() * 1000
        file1.close()
    except KeyError:
        print("header missing defaulting to sampling frequency of 1000hz")
        fs = 1000
    except json.decoder.JSONDecodeError:
        print("header missing defaulting to sampling frequency of 1000hz")
        fs = 1000
    print(f"Sampling rate {fs}hz")
    # print(data.head())

    return data, fs, t0


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def FNIRsToSpO2(IR, red, window):
    IR_rolling = IR.rolling(window)
    red_rolling = red.rolling(window)
    IR_avg = IR_rolling.mean().fillna(0).to_numpy()
    red_avg = red_rolling.mean().fillna(0).to_numpy()
    IR_min = IR_rolling.min().to_numpy()
    IR_max = IR_rolling.max().fillna(0).to_numpy()
    red_min = red_rolling.min().fillna(0).to_numpy()
    red_max = red_rolling.max().fillna(0).to_numpy()
    IR_vpp = IR_max - IR_min
    red_vpp = red_max - red_min
    R = (red_vpp * IR_avg) / (red_avg * IR_vpp)
    SpO2 = 110 - 25 * R
    SpO2 = SpO2[window::]
    if np.average(SpO2) < 50:
        warnings.warn("fNIRS data is potentially bad, average SpO2<50%")
    Sp02Rev = ((SpO2 * 95) / SpO2[0])
    return SpO2[::window], Sp02Rev[::window]


def displayData(df, df_sa, sr):
    figs = deque()

    time = np.array(df["Time"]) / sr
    resolution = 16  # Resolution (number of available bits)
    signal_red_uA = (0.15 * np.array(df["fNIRS1"])) / 2 ** resolution
    signal_infrared_uA = (0.15 * np.array(df["fNIRS2"])) / 2 ** resolution

    b, a = butter(4, 15, 'highpass', fs=sr)
    signal_red_uA = pd.Series(signal_red_uA - filtfilt(b, a, signal_red_uA))
    signal_infrared_uA = pd.Series(signal_infrared_uA - filtfilt(b, a, signal_infrared_uA))

    eeg = EEGTransferFunction(df["EEG"].to_numpy())
    rawFig = plt.figure("Raw Data")
    # Plot EEG
    plt.subplot(3, 1, 1).set_title("EEG")
    plt.plot(time, eeg)
    plt.xlabel("Time (S)")
    plt.ylabel("Voltage (uV)")

    # Plot fNIRS1
    plt.subplot(3, 1, 2).set_title("fNIRs red")
    plt.plot(time, signal_red_uA)
    plt.xlabel("Time (S)")
    plt.ylabel("Amperage (uA)")

    # Plot fNIRS2
    plt.subplot(3, 1, 3).set_title("fNIRs IR")
    plt.plot(time, signal_infrared_uA)
    plt.xlabel("Time (S)")
    plt.ylabel("Amperage (uA)")

    SpO2Fig = plt.figure("SpO2 fNIRs")
    SpO2, SpO2Rev = FNIRsToSpO2(signal_infrared_uA, signal_red_uA, sr)
    rolling_avg_SpO2 = moving_average(SpO2, 5)
    bnotch, anotch = signal.iirnotch(0.2, 2)
    rolling_w_notch_filter = signal.filtfilt(bnotch, anotch, rolling_avg_SpO2)
    reaction = df_sa["Reaction"].to_numpy()
    sa_times = df_sa["Time"].to_numpy()
    fNIRS_mean = []
    fNIRS_min = []
    fNIRS_max = []
    reaction_mean = []
    reaction_var = []
    window_length = 30  # length of min max and mean window in seconds
    step_size = 2
    for i in np.arange(0, len(rolling_w_notch_filter), int(step_size)):
        start = i
        end = min(start + window_length, len(rolling_w_notch_filter))
        slice = rolling_w_notch_filter[start:end]
        if len(slice) == 0: break
        fNIRS_mean.append(np.mean(slice, axis=0))
        fNIRS_min.append(np.amin(slice, axis=0))
        fNIRS_max.append(np.amax(slice, axis=0))
        s_start = find_nearest_index(sa_times, start)
        s_end = find_nearest_index(sa_times, end)
        reaction_slice = reaction[s_start:s_end]
        mean = np.mean(reaction_slice, axis=0)
        var = np.std(reaction_slice, axis=0)
        reaction_mean.append(mean)
        reaction_var.append(var)
    reaction_mean = np.array(reaction_mean)
    reaction_var = np.array(reaction_var)
    window_times = np.arange(0, len(fNIRS_mean)*step_size, step_size)

    ax = plt.subplot(2, 1, 1)
    for t in range(window_length, int(len(time) / sr), window_length):
        ax.axvline(x=t, linestyle='--', linewidth=1, color='blue', alpha=.25)
    lspo2, = ax.plot(rolling_w_notch_filter, color='red', linewidth=1.5, alpha=.6)
    lmean, = ax.plot(window_times, fNIRS_mean, linestyle='--', linewidth=2.5, color='black')
    lmin, = ax.plot(window_times, fNIRS_min, linewidth=4, color='black', alpha=.5)
    lmax, = ax.plot(window_times, fNIRS_max, linewidth=4, color='black', alpha=.5)
    ax.legend([lspo2, lmean, lmin, lmax], ['Sp02', 'Average', 'Min', 'Max'])
    plt.xlabel("Time (S)")
    plt.ylabel("SpO2 (%)")
    ax = plt.subplot(2, 1, 2)
    for t in range(window_length, int(len(time) / sr), window_length):
        ax.axvline(x=t, linestyle='--', linewidth=1, color='blue', alpha=.25)
    lraw, = ax.plot(df_sa["Time"], reaction, linewidth=1, color='red')
    lavg, = ax.plot(window_times, reaction_mean, linewidth=2, color='purple')
    lvar, = ax.plot(window_times, reaction_var, linewidth=2, color='green')
    ax.legend([lraw, lavg, lvar], ['Reaction Time', 'Mean', 'Variance'])

    notNan = np.where(np.logical_not(np.isnan(reaction_mean)))[0]
    first_val = notNan[0]
    last_val = notNan[len(notNan) - 1]

    scatterMean = plt.figure("ScatterPlot Mean")
    ax = plt.subplot(1, 1, 1)
    x = fNIRS_mean[first_val:last_val]
    y = reaction_mean[first_val:last_val]
    ax.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--")
    correlation = np.corrcoef(x, y)[0, 1]
    ax.text(x[0], p(x)[0], "R={:.3f}, Rsq={:.3f}".format(correlation, correlation ** 2))
    scatterVar = plt.figure("ScatterPlot Variance")
    ax = plt.subplot(1, 1, 1)
    y = reaction_var[first_val:last_val]
    ax.scatter(x, y)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--")
    correlation = np.corrcoef(x, y)[0, 1]
    ax.text(x[0], p(x)[0], "R={:.3f}, Rsq={:.3f}".format(correlation, correlation ** 2))

    # scatterFig = plt.figure("ScatterPlot")
    # ax = plt.subplot(1, 1, 1)
    # resampled = np.array([find_nearest(SpO2, t) for t in df_sa["Time"]])
    # x = df_sa["Reaction"]
    # y = resampled
    # ax.scatter(x, y, s=40, alpha=.5)
    # z = np.polyfit(x, y, 1)
    # p = np.poly1d(z)
    # ax.plot(x, p(x), "r--")
    figs.append(rawFig)
    figs.append(SpO2Fig)
    figs.append(scatterMean)
    figs.append(scatterVar)
    return figs


def find_nearest_index(array, value):
    array = np.abs(array - value)
    idx = array.argmin()
    return idx


def toTime(s):
    ret = ((datetime.strptime("2000:" + s, "%Y:%H:%M:%S.%f").timestamp() * 1000 - t0) / 1000)
    return ret


def readSustainedFile(path, t0, sr):
    print(f"Loading data from {path}")
    data = pd.read_csv(path, skiprows=1, header=None, usecols=[0, 18, 19, 20],
                       names=["Time", "Input", "Correct", "Reaction"])
    # data = data[data["Reaction"] != -1]
    data["Time"] = pd.Series([toTime(s) for s in data["Time"]])
    return data[data["Reaction"] != -1]


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    running = True
    while running:
        print("running main loop")
        path = askopenfilename(title="Open file from BioSignals")
        if path == '':
            exit(0)
        df, fs, t0 = readBioSignalFile(path)
        path = askopenfilename(title="Open file from sustained attention")
        if path == '':
            exit(0)
        df_sa = readSustainedFile(path, t0, fs)
        figs = displayData(df, df_sa, fs)
        root = tkinter.Tk()
        root.title(f"BioSignalPlux Viewer: {os.path.basename(path)}")
        tabPane = ChartsTabPanel.ChartsTabPane(root, figs)
        plt.close('all')
        # running = False
