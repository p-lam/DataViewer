import json
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
#import matplotlib.pyplot as plt

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

    b, a = butter(4, [.01 / (sr * 0.5), 15 / (sr * 0.5)], 'bandpass', analog=True)
    df["fNIRS1"] = pd.Series(filtfilt(b, a, df["fNIRS1"]))
    df["fNIRS2"] = pd.Series(filtfilt(b, a, df["fNIRS2"]))

    time = np.array(df["Time"]) / sr
    resolution = 16  # Resolution (number of available bits)
    signal_red_uA = (0.15 * np.array(df["fNIRS1"])) / 2 ** resolution
    signal_infrared_uA = (0.15 * np.array(df["fNIRS1"])) / 2 ** resolution

    eeg = EEGTransferFunction(df["EEG"].to_numpy())
    rawFig = plt.figure("Raw Data")
    # Plot EEG
    plt.subplot(3, 1, 1).set_title("EEG")
    plt.plot(time, eeg)

    # Plot fNIRS1
    plt.subplot(3, 1, 2).set_title("fNIRs red")
    plt.plot(time, signal_red_uA)


    # Plot fNIRS2
    plt.subplot(3, 1, 3).set_title("fNIRs IR")
    plt.plot(time, signal_infrared_uA)

    spectrogramFig, meanFig = plotSpectrogram(eeg, sr, cpu_cores=1, window=[5, 1], res=1, resample=False)

    SpO2Fig = plt.figure("SpO2 fNIRs")
    SpO2, SpO2Rev = FNIRsToSpO2(df["fNIRS2"], df["fNIRS1"], sr)
    ax = plt.subplot(1, 1, 1)
    ax.plot(SpO2, linewidth=0.75)
    rolling_avg_SpO2 = moving_average(SpO2, 5)
    ax.plot(rolling_avg_SpO2, color='purple', linewidth=0.75)
    bnotch, anotch = signal.iirnotch(0.2, 2)
    rolling_w_notch_filter = signal.filtfilt(bnotch, anotch, rolling_avg_SpO2)
    ax.plot(rolling_w_notch_filter, color='red')
    # plt.figure("FNIRS psd")
    # plt.magnitude_spectrum(rolling_avg_SpO2)
    # plt.xlim([0.05, 1])
    #plt.show()
    figs.append(rawFig)
    figs.append(SpO2Fig)
    figs.append(spectrogramFig)
    figs.append(meanFig)
    return figs


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    running = True
    while running:
        import matplotlib.pyplot as plt
        print("running main loop")
        path = askopenfilename()
        if path == '':
            exit(0)
        df, fs = readFile(path)
        figs = displayData(df, fs)
        root = tkinter.Tk()
        root.title(f"BioSignalPlux Viewer: {os.path.basename(path)}")
        tabPane = ChartsTabPanel.ChartsTabPane(root, figs)
        #running = False
