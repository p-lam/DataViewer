import json
import tkinter
import warnings
from collections import deque
from multiprocessing import Pool, cpu_count
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ChartsTabPanel

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
    figs = deque()
    i = 0
    for col in df:
        figs.append(plotSpectrogram(df[col].to_numpy(), sr, cpu_cores=1, name=col)[0])
        i = i+1
        print(f"{i}/{len(df.columns)}")
    return figs


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    while True:
        Tk().withdraw()
        path = askopenfilename()
        if path == '':
            exit(0)
        df, fs = readFile(path)
        df = df.drop(["Time", "Trigger", "Time_Offset", "ADC_Status", "ADC_Sequence", "Event", "Comments"], axis=1)
        figs = displayData(df, fs)
        root = tkinter.Tk()
        tabPane = ChartsTabPanel.ChartsTabPane(root, figs)
