import json
import math
import os
import tkinter

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


def readFile(path):
    print(f"Loading data from {path}")
    eoh = getEOH(path)
    eoh += 10  # skips first 10 lines
    data = pd.read_csv(path, skiprows=eoh, header=None, sep="\t", usecols=[0, 2, 3, 4],
                       names=["Time", "EEG", "fNIRS1", "fNIRS2"])

    try:
        file1 = open(path, 'r')
        jsonText = file1.readlines()[1][2:]
        headerJson = json.loads(jsonText)
        headerJson = headerJson[list(headerJson)[0]]
        fs = headerJson["sampling rate"]
        file1.close()
    except KeyError:
        print("header missing defaulting to sampling frequency of 1000hz")
        fs = 1000
    except json.decoder.JSONDecodeError:
        print("header missing defaulting to sampling frequency of 1000hz")
        fs = 1000
    print(f"Sampling rate {fs}hz")
    # print(data.head())

    return data, fs


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


def displayData(df, sr):
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

    spectrogramFig, meanFig, aplha, beta, gamma = plotSpectrogram(eeg, sr, cpu_cores=1, window=[4, 1], res=1.5,
                                                                  resample=False)

    SpO2Fig = plt.figure("SpO2 fNIRs")
    SpO2, SpO2Rev = FNIRsToSpO2(signal_infrared_uA, signal_red_uA, sr)
    ax = plt.subplot(1, 1, 1)
    #ax.plot(SpO2, linewidth=0.5)
    rolling_avg_SpO2 = moving_average(SpO2, 5)
    #ax.plot(rolling_avg_SpO2, color='purple', linewidth=0.75)
    for t in range(60, int(len(time) / sr), 60):
        ax.axvline(x=t, linestyle='--', linewidth=1.5, color='blue')
    bnotch, anotch = signal.iirnotch(0.2, 2)
    rolling_w_notch_filter = signal.filtfilt(bnotch, anotch, rolling_avg_SpO2)
    fNIRS_mean = []
    fNIRS_min = []
    fNIRS_max = []
    n = math.ceil(time[len(time)-1]/10)  # length of min max and mean window in seconds
    for i in np.arange(0,int(len(rolling_w_notch_filter)/n)+1):
        start = i * n
        end = start + n
        slice = rolling_w_notch_filter[start:end]
        fNIRS_mean.append(np.mean(slice,axis=0))
        fNIRS_min.append(np.amin(slice,axis=0))
        fNIRS_max.append(np.amax(slice,axis=0))
    fNIRS_times = np.arange(0, len(fNIRS_mean)*n, n)
    lspo2, = ax.plot(rolling_w_notch_filter, color='red', linewidth=2)
    lmean, = ax.plot(fNIRS_times,fNIRS_mean, linestyle='--', linewidth=.75, color='black')
    lmin, = ax.plot(fNIRS_times, fNIRS_min, linewidth=4, color='black',alpha=.3)
    lmax, = ax.plot(fNIRS_times, fNIRS_max, linewidth=4, color='black',alpha=.3)
    ax.legend([lspo2, lmean, lmin, lmax], ['Sp02', 'Average', 'Min','Max'])
    plt.xlabel("Time (S)")
    plt.ylabel("SpO2 (%)")
    figs.append(rawFig)
    figs.append(SpO2Fig)
    figs.append(spectrogramFig)
    figs.append(meanFig)
    return figs


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    running = True
    while running:
        print("running main loop")
        path = askopenfilename()
        if path == '':
            exit(0)
        df, fs = readFile(path)
        figs = displayData(df, fs)
        root = tkinter.Tk()
        root.title(f"BioSignalPlux Viewer: {os.path.basename(path)}")
        tabPane = ChartsTabPanel.ChartsTabPane(root, figs)
        plt.close('all')
        # running = False
