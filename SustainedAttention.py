import utils.custom_Importer as importer
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

importer.doneImports()
print("\nLibraries loaded")


def displayData(df, sa_df, sr):
    # unit conversions
    signal_red_uA, signal_infrared_uA = fNIRsTransferFunction(df["fNIRS1"], df["fNIRS2"], sr)
    eeg = EEGTransferFunction(df["EEG"].to_numpy())

    time = np.array(df["Time"]) / sr

    # Plot Raw Data
    rawFig, axes = plt.subplots(3, 1, num="Raw Data")
    raw_plotter = plotter.CommonX(axes, time, "Time (S)")
    raw_plotter.plot(eeg, "Voltage (uV)", "EEG")
    raw_plotter.plot(signal_red_uA, "Amperage (uA)", "fNIRs red")
    raw_plotter.plot(signal_infrared_uA, "Amperage (uA)", "fNIRs IR")

    # fNIRS and reaction-time post processing
    reaction = sa_df["Reaction"]
    reaction_timestamps = sa_df["Time"]

    SpO2, SpO2Rev, fNIRS_time = fNIRsToSpO2(signal_infrared_uA, signal_red_uA, sr)

    window_length = 30
    step_size = 2

    SpO2, nirs_df = SpO2_postProcessing(SpO2, times=fNIRS_time, window_length=window_length, step_size=step_size)
    rt_df = reactionTime_postProcessing(reaction, reaction_timestamps, nirs_df["Time"], window_length)

    # fNIRS and reaction-time plotting
    drawBlueLines = lambda ax: [  # draws a dotted blue line at every window
        ax.axvline(x=t, linestyle='--', linewidth=1, color='blue', alpha=.25)
        for t in range(window_length, int(len(time) / sr), window_length)
    ]
    SpO2Fig = plt.figure("SpO2 fNIRs")
    ax = plt.subplot(2, 1, 1)
    drawBlueLines(ax)
    lspo2, = ax.plot(SpO2, linewidth=1.5, color='red', alpha=.6)
    lmean, = ax.plot(nirs_df["Time"], nirs_df["Mean"], linestyle='--', linewidth=2.5, color='black')
    lmin, = ax.plot(nirs_df["Time"], nirs_df["Min"], linewidth=4, color='black', alpha=.5)
    lmax, = ax.plot(nirs_df["Time"], nirs_df["Max"], linewidth=4, color='black', alpha=.5)
    ax.legend([lspo2, lmean, lmin, lmax], ['Sp02', 'Average', 'Min', 'Max'])
    plt.xlabel("Time (S)")
    plt.ylabel("SpO2 (%)")

    ax = plt.subplot(2, 1, 2)
    drawBlueLines(ax)
    lraw, = ax.plot(sa_df["Time"], reaction, linewidth=1, color='red')
    lavg, = ax.plot(rt_df["Time"], rt_df["Mean"], linewidth=2, color='purple')
    lvar, = ax.plot(rt_df["Time"], rt_df["Var"], linewidth=2, color='green')
    ax.legend([lraw, lavg, lvar], ['Reaction Time', 'Mean', 'Variance'])

    fNIRS_mean = nirs_df["Mean"].to_numpy()
    reaction_mean = rt_df["Mean"].to_numpy()
    reaction_var = rt_df["Var"].to_numpy()
    scatter_plotter = plotter.Scatter_And_Trend(xLabel="SpO2 (%)", yLabel="Reaction Time (S)")

    notNan = np.where(np.logical_not(np.isnan(reaction_mean)))[0]

    clip_size = 10  # ammount of time to clip off the ends where the data is the noisiest in seconds
    # clip_size should preffrably be very small as the recording duration is only around 270 seconds long
    first_val = notNan[0] + int(clip_size / step_size)
    last_val = notNan[len(notNan) - 1] - int(clip_size / step_size)
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
        if not path.endswith(".txt"): exit(0)
        df, fs, t0 = parser.readBioSignals(path)
        path = askopenfilename(title="Open file from sustained attention")
        if not path.endswith(".csv"): exit(0)
        df_sa = parser.readSustainedFile(path, t0, fs)
        figs = displayData(df, df_sa, fs)
        root = tkinter.Tk()
        root.title(f"BioSignalPlux Viewer: {os.path.basename(path)}")
        tabPane = charts_tab_panel.ChartsTabPane(root, figs)
        plt.close('all')
