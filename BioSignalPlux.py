import json

from scipy.signal import filtfilt

print(".", end="")
from tkinter import Tk

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
print("Libraries loaded")


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
    data = pd.read_csv(path, skiprows=eoh, header=None, sep="\t", usecols=[0, 1, 2, 3],
                       names=["Time", "EEG", "fNIRS1", "fNIRS2"])

    try:
        file1 = open(path, 'r')
        jsonText = file1.readlines()[1][2:]
        headerJson = json.loads(jsonText)
        headerJson = headerJson[list(headerJson)[0]]
        fs = headerJson["sampling rate"]
        file1.close()
    except KeyError:
        print("\theader missing defaulting to sampling frequency of 1000hz")
        fs = 1000
    except json.decoder.JSONDecodeError:
        print("\theader missing defaulting to sampling frequency of 1000hz")
        fs = 1000
    print(f"\tSampling rate {fs}hz")
    print(data.head())

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
    Sp02Rev = ((SpO2 * 95) / SpO2[0])
    return SpO2[::window], Sp02Rev[::window]


def displayData(df, sr):
    # yasa.plot_spectrogram(df["EEG"].to_numpy(), win_sec=1, sf=1000, cmap='Spectral_r')
    # plt.xlabel("Time[S]")
    time = np.array(df["Time"]) / sr
    resolution = 16  # Resolution (number of available bits)
    signal_red_uA = (0.15 * np.array(df["fNIRS1"])) / 2 ** resolution
    signal_infrared_uA = (0.15 * np.array(df["fNIRS1"])) / 2 ** resolution

    eeg = EEGTransferFunction(df["EEG"])
    plt.figure(0)
    # Plot EEG
    plt.subplot(4, 1, 1)
    plt.plot(time, eeg)

    # Plot fNIRS1
    plt.subplot(4, 1, 2)
    plt.plot(time, signal_red_uA)

    # Plot fNIRS2
    plt.subplot(4, 1, 3)
    plt.plot(time, signal_infrared_uA)

    ax = plt.subplot(4, 1, 4)
    avg_red = moving_average(signal_infrared_uA, 100)[::sr]
    avg_ir = moving_average(signal_infrared_uA, 100)[::sr]
    ax.plot(avg_red[0] - avg_red)
    ax.plot(avg_ir - avg_ir[0])

    plotSpectrogram(eeg, sr)

    plt.figure()
    SpO2, Sp02Rev = FNIRsToSpO2(df["fNIRS2"], df["fNIRS1"], sr)
    ax = plt.subplot(1, 1, 1)
    ax.plot(SpO2)
    ax.plot(moving_average(SpO2, 5))
    plt.ylim([85, 99])
    plt.show()


if __name__ == '__main__':

    while True:
        Tk().withdraw()
        path = askopenfilename()
        if path == '':
            exit(0)
        df, fs = readFile(path)
        displayData(df, fs)
