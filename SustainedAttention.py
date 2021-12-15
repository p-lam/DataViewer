import utils.custom_Importer
import os
import tkinter
from utils import parser, plotter
from utils.biosignal_processing import *
from utils.sustainedattention_processing import *
from utils import charts_tab_panel
import warnings
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
print("\nLibraries loaded")


def displayData(df, df_sa, sr):
    # unit conversions
    signal_red_uA, signal_infrared_uA = fNIRsTransferFunction(df["fNIRS1"], df["fNIRS2"],sr)
    eeg = EEGTransferFunction(df["EEG"].to_numpy())

    # Plot Raw Data
    rawFig, axes = plt.subplots(3, 1, num ="Raw Data")
    raw_plotter = plotter.CommonX(axes, time, "Time (S)")
    raw_plotter.plot(eeg, "Voltage (uV)", "EEG")
    raw_plotter.plot(signal_red_uA, "Amperage (uA)", "fNIRs red")
    raw_plotter.plot(signal_infrared_uA, "Amperage (uA)", "fNIRs IR")

    # fNIRS and reaction-time post processing
    reaction = df_sa["Reaction"]
    reaction_timestamps = df_sa["Time"]

    SpO2, SpO2Rev, fNIRS_time = fNIRsToSpO2(signal_infrared_uA, signal_red_uA, sr)

    window_length = 30
    step_size = 2

    SpO2, df_nir_w = SpO2_postProcessing(SpO2, times=fNIRS_time, window_length=window_length, step_size=step_size)
    df_rt = reactionTime_postProcessing(reaction, reaction_timestamps)

    # fNIRS and reaction-time plotting
    drawBlueLines = lambda ax: [  # draws a dotted blue line at every window
        ax.axvline(x=t, linestyle='--', linewidth=1, color='blue', alpha=.25)
        for t in range(window_length, int(len(time) / sr), window_length)
    ]
    SpO2Fig = plt.figure("SpO2 fNIRs")
    ax = plt.subplot(2, 1, 1)
    drawBlueLines(ax)
    lspo2, = ax.plot(SpO2)
    lmean, = ax.plot(df_nir_w["Time"], df_nir_w["Mean"])
    lmin, = ax.plot(df_nir_w["Time"], df_nir_w["Min"])
    lmax, = ax.plot(df_nir_w["Time"], df_nir_w["Max"])
    ax.legend([lspo2, lmean, lmin, lmax], ['Sp02', 'Average', 'Min', 'Max'])
    plt.xlabel("Time (S)")
    plt.ylabel("SpO2 (%)")

    ax = plt.subplot(2, 1, 2)
    drawBlueLines(ax)
    lraw, = ax.plot(df_sa["Time"], reaction)
    lavg, = ax.plot(df_rt["Time"], df_rt["Mean"])
    lvar, = ax.plot(df_rt["Time"], df_rt["Var"])
    ax.legend([lraw, lavg, lvar], ['Reaction Time', 'Mean', 'Variance'])

    fNIRS_mean = df_nir_w["Mean"].to_numpy()
    reaction_mean = df_rt["Mean"].to_numpy()
    fNIRS_mean = df_nir_w["Mean"].to_numpy()
    reaction_var = df_rt["Mean"].to_numpy()
    scatter_plotter = plotter.Scatter_And_Trend(xLabel="SpO2 (%)", yLabel="Reaction Time (S)")

    notNan = np.where(np.logical_not(np.isnan(reaction_mean)))[0]
    first_val = notNan[0]
    last_val = notNan[len(notNan) - 1]
    x = fNIRS_mean[first_val:last_val]

    scatterMean = plt.figure("ScatterPlot Mean")
    ax = plt.subplot(1, 1, 1)
    y = reaction_mean[first_val:last_val]
    scatter_plotter.plot(ax, x, y)

    scatterVar = plt.figure("ScatterPlot Variance")
    ax = plt.subplot(1, 1, 1)
    y = reaction_var[first_val:last_val]
    scatter_plotter.plot(ax, x, y)

    return [rawFig, SpO2Fig, scatterMean, scatterVar]

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    while True:
        print("running main loop")
        path = askopenfilename(title="Open file from BioSignals")
        if not path.endswith(".txt"):exit(0)
        df, fs, t0 = parser.readBioSignals(path)
        path = askopenfilename(title="Open file from sustained attention")
        if not path.endswith(".csv"):exit(0)
        df_sa = parser.readSustainedFile(path, t0, fs)
        figs = displayData(df, df_sa, fs)
        root = tkinter.Tk()
        root.title(f"BioSignalPlux Viewer: {os.path.basename(path)}")
        tabPane = charts_tab_panel.ChartsTabPane(root, figs)
        plt.close('all')
