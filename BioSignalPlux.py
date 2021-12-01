import json

from scipy.signal import filtfilt

print(".",end="")
from tkinter import Tk
print(".",end="")
from tkinter.filedialog import askopenfilename
print(".",end="")
import matplotlib.pyplot as plt
print(".",end="")
import numpy as np
print(".",end="")
import pandas as pd
print(".",end="")
from Spectrogram import plotSpectrogram
print(".",end="")
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
    data = pd.read_csv(path, skiprows=eoh, header=None, sep="\t", usecols=[0,2, 3, 4], names=["Time","EEG", "fNIRS1", "fNIRS2"])

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


def displayData(df, sr):
    # yasa.plot_spectrogram(df["EEG"].to_numpy(), win_sec=1, sf=1000, cmap='Spectral_r')
    # plt.xlabel("Time[S]")
    time = np.array(df["Time"])/sr
    resolution = 16  # Resolution (number of available bits)
    signal_red_uA = (0.15 * np.array(df["fNIRS1"])) / 2 ** resolution
    signal_infrared_uA = (0.15 * np.array(df["fNIRS1"])) / 2 ** resolution
    eeg = EEGTransferFunction(df["EEG"])
    plt.figure(0)
    # Plot EEG
    plt.subplot(4, 1, 1)
    plt.plot(time,eeg)

    # Plot fNIRS1
    plt.subplot(4, 1, 2)
    plt.plot(time,signal_red_uA)

    # Plot fNIRS2
    plt.subplot(4, 1, 3)
    plt.plot(time,signal_infrared_uA)

    ax = plt.subplot(4, 1, 4)
    avg_red = moving_average(signal_infrared_uA, 100)
    avg_ir = moving_average(signal_infrared_uA, 100)
    ax.plot(avg_red[0]-avg_red)
    ax.plot(avg_ir-avg_ir[0])

    plotSpectrogram(eeg, sr)

    #lp = np.hamming(35) / np.sum(np.hamming(35))
    #eeg = eeg - filtfilt(lp, 1, eeg)

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.magnitude_spectrum(eeg[10 * sr:50 * sr],Fs=sr)
    # plt.xlim([0,40])
    # plt.subplot(2, 1, 2)
    # plt.magnitude_spectrum(eeg[70 * sr:110 * sr],Fs=sr)
    # plt.xlim([0,40])

    plt.show()


if __name__ == '__main__':

    while True:
        Tk().withdraw()
        path = askopenfilename()
        if path == '':
            exit(0)
        df, fs = readFile(path)
        displayData(df, fs)
