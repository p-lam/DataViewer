from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from Spectrogram import plotSpectrogram


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
    data = pd.read_csv(path, skiprows=eoh, header=None, sep="\t", usecols=[2, 3, 4], names=["EEG", "fNIRS1", "fNIRS2"])

    try:
        file1 = open(path, 'r')
        jsonText = file1.readlines()[1][2:]
        headerJson = json.loads(jsonText)
        fs = headerJson["00:07:80:79:6F:DB"]["sampling rate"]
        file1.close()
    except json.decoder.JSONDecodeError:
        print("\theader missing defaulting to sampling frequency of 1000hz")
        fs = 1000
    print(f"\tSampling rate {fs}hz")
    print(data.head())

    return data, fs


def displayData(df, sr):
    # yasa.plot_spectrogram(df["EEG"].to_numpy(), win_sec=1, sf=1000, cmap='Spectral_r')
    # plt.xlabel("Time[S]")
    resolution = 16  # Resolution (number of available bits)
    signal_red_uA = (0.15 * np.array(df["fNIRS1"])) / 2 ** resolution
    signal_infrared_uA = (0.15 * np.array(df["fNIRS1"])) / 2 ** resolution
    eeg = EEGTransferFunction(df["EEG"])
    plt.figure(0)
    # Plot EEG
    plt.subplot(3, 1, 1)
    plt.plot(eeg)

    # Plot fNIRS1
    plt.subplot(3, 1, 2)
    plt.plot(signal_red_uA)

    # Plot fNIRS2
    plt.subplot(3, 1, 3)
    plt.plot(signal_infrared_uA)

    plotSpectrogram(eeg, sr)
    plt.show()


if __name__ == '__main__':

    while True:
        Tk().withdraw()
        path = askopenfilename()
        if path == '':
            break
        df, fs = readFile(path)
        displayData(df, fs)
