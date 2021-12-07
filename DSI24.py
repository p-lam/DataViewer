import json
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Spectrogram import plotSpectrogram
print("Libraries loaded")

def getEOH(path):
    tmp = open(path)
    eoh = 0
    for line in tmp.readlines():
        if line.find("#") < 0:
            return eoh
        eoh += 1
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
    data = pd.read_csv(path, skiprows=eoh)
    fs = 300
    print(f"\tSampling rate {fs}hz")
    print(data.head())

    return data, fs


def displayData(df, sr):
    for column in df:
        plt.suptitle(f"EEG: {column}")
        plotSpectrogram(df[column].to_numpy(), sr)
        plt.show()


if __name__ == '__main__':

    while True:
        Tk().withdraw()
        path = askopenfilename()
        if path == '':
            exit(0)
        df, fs = readFile(path)
        df = df.drop(["Time", "Trigger", "Time_Offset", "ADC_Status", "ADC_Sequence", "Event", "Comments"], axis=1)
        displayData(df, fs)
