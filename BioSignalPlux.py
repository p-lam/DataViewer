import utils.custom_Importer as importer
from utils import parser
import os
import tkinter
from utils import charts_tab_panel
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from utils.biosignal_processing import *
from utils.sustainedattention_processing import *
from utils.spectrogram import plotSpectrogram
from utils import plotter
importer.doneImports()
print("\nLibraries loaded")



def displayData(df, sr):
    time = np.array(df["Time"]) / sr
    #unit conversions
    signal_red_uA,signal_infrared_uA = fNIRsTransferFunction(df["fNIRS1"], df["fNIRS2"],sr)
    eeg = EEGTransferFunction(df["EEG"].to_numpy())

    # Plot Raw Data
    rawFig, axes = plt.subplots(3, 1, num ="Raw Data")
    raw_plotter = plotter.CommonX(axes, time, "Time (S)")
    raw_plotter.plot(eeg, "Voltage (uV)", "EEG")
    raw_plotter.plot(signal_red_uA, "Amperage (uA)", "fNIRs red")
    raw_plotter.plot(signal_infrared_uA, "Amperage (uA)", "fNIRs IR")

    #spectrogram plotting
    spectrogramFig, meanFig, aplha, beta, gamma = plotSpectrogram(eeg, sr, cpu_cores=1, window=[4, 1], res=1.5,
                                                                  resample=False)
    #fNIRS plotting
    SpO2Fig = plt.figure("SpO2 fNIRs")
    SpO2, SpO2Rev, fNIRs_times = fNIRsToSpO2(signal_infrared_uA, signal_red_uA, sr)
    SpO2, wdf = SpO2_postProcessing(SpO2Rev,times=fNIRs_times)
    ax = plt.subplot(1, 1, 1)
    for t in range(60, int(len(time) / sr), 60):
        ax.axvline(x=t, linestyle='--', linewidth=1.5, color='blue')
    lspo2, = ax.plot(fNIRs_times,SpO2)
    lmean, = ax.plot(wdf["Time"], wdf["Mean"])
    lmin, = ax.plot(wdf["Time"], wdf["Min"])
    lmax, = ax.plot(wdf["Time"], wdf["Max"])
    ax.legend([lspo2, lmean, lmin, lmax], ['Sp02', 'Average', 'Min', 'Max'])
    plt.xlabel("Time (S)")
    plt.ylabel("SpO2 (%)")

    return [rawFig,SpO2Fig,spectrogramFig,meanFig]


if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    while True:
        print("running main loop")
        path = askopenfilename()
        if path == '':
            exit(0)
        df, fs = parser.readBioSignals(path)
        figs = displayData(df, fs)
        root = tkinter.Tk()
        root.title(f"BioSignalPlux Viewer: {os.path.basename(path)}")
        tabPane = charts_tab_panel.ChartsTabPane(root, figs)
        plt.close('all')
