import pandas as pd
import numpy as np
import warnings

from scipy import signal

def EEGTransferFunction(nparr):
    # Equation from: https://biosignalsplux.com/learn/notebooks/Categories/Pre-Process/unit_conversion_eeg_rev.html
    # EEG(µV)=( ( (ADC / 2^n) - 1/2) * VCC) / Gᴇᴇɢ
    # ADC = raw value
    # n = resolution = 16
    # VCC = 3V
    # Gᴇᴇɢ = gain = 40000

    nparr = nparr / 65536
    nparr = nparr - .5
    nparr = nparr * 3
    nparr = nparr / 40000
    nparr = nparr * 1000000  # microvolts to volts
    return nparr

def fNIRsTransferFunction(red, ir, sr):
    resolution = 16  # Resolution (number of available bits)
    signal_red_uA = (0.15 * np.array(red)) / 2 ** resolution
    signal_infrared_uA = (0.15 * np.array(ir)) / 2 ** resolution
    b, a = signal.butter(4, 15, 'highpass', fs=sr)
    signal_red_uA = pd.Series(signal_red_uA - signal.filtfilt(b, a, signal_red_uA))
    signal_infrared_uA = pd.Series(signal_infrared_uA - signal.filtfilt(b, a, signal_infrared_uA))
    return signal_red_uA, signal_infrared_uA

def fNIRsToSpO2(IR, red, sr=500, window=1):
    window = window * sr  # window size defaults to 500 samples or 1 second of data at 500hz

    # Equations from https://biosignalsplux.com/downloads/docs/technical-notes/fNIRS_Sensor_apnea_TN.pdf

    # i refers to the index of the window
    # Vpp refers to the difference in amplitude between max and min over a window
    # Vavg refers to the average amplitude over a window

    # Red/Infrared Modulation Ratio Equation
    # R[i]=(Vppᴿ[i] * Vavgᴵᴿ[i]) / (Vppᴵᴿ[i] * Vavgᴿ[i])
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

    # SpO2 Equation
    # %SpO2 = 110 = 25 * R[i]
    SpO2 = 110 - 25 * R
    SpO2 = SpO2[window::window]  # removes nan values in first window-length, makes window non-rolling
    if np.average(SpO2) < 85:
        warnings.warn("fNIRS data is potentially bad, average SpO2<85%")
    # Normalizes SpO2 so that SpO2[0] = 95%
    Sp02Rev = ((SpO2 * 95) / SpO2[0])

    times = np.arange(0, len(SpO2) * window, window) / sr

    return SpO2, Sp02Rev, times






