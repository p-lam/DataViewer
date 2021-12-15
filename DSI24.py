import utils.custom_Importer as importer
import os
import tkinter
import warnings
from collections import deque
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from utils import parser
from utils import charts_tab_panel
from utils.spectrogram import plotSpectrogram
importer.doneImports()
print("\nLibraries loaded")


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
        df, fs = parser.readDSI(path)
        df = df.drop(["Time", "Trigger", "Time_Offset", "ADC_Status", "ADC_Sequence", "Event", "Comments"], axis=1)
        figs = displayData(df, fs)
        root = tkinter.Tk()
        root.title(f"DSI24 Viewer: {os.path.basename(path)}")
        tabPane = charts_tab_panel.ChartsTabPane(root, figs)
