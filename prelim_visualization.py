# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:58:00 2021

@author: mdric
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from scipy.signal import spectrogram
from scipy.fft import fftshift

from util.multitaper import multitaper_spectrogram


file1 = ['C:\\Users\\mdric\\Downloads\\eyeseopeneyesclosed_maceo_11-15-21.txt',
         'Person1', 'Fp1', 1000]

file2 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_leftear_prudence_11-22-21.txt',
         'Person2', 'left ear', 1000]

file3 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_Greg_11-15-21.txt',
         'Person3', 'Fp1', 1000]

file4 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclose_emily_11-12-21.txt',
         'Person4', 'Fp1', 1000]

file5 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed2_emily_11-12-21.txt',
         'Person4', 'Fp1', 1000]

file6 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclose_leftear_nathan_11-22-21.txt',
         'Person5', 'left ear', 1000]

file7 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_jack_11-15-21.txt',
         'Person6', 'Fp1', 1000]

file8 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_Ian_11-15-21.txt',
         'Person7', 'Fp1', 1000]

file9 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_Pz_greg_11-30-21.txt',
         'Person3', 'Pz', 1000]

file10 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_behindear_maceo_12-1-21.txt',
          'Person1', 'T3, T5', 1000]

file11 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_shane_12-1-21.txt',
          'Person8', 'T3, T5', 1000]

file12 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_fnirs_eeg_shane.txt',
          'Person8', 'T3, T5', 1000]

file13 = ['C:\\Users\\mdric\\Downloads\\sustainedattention_shane_12-2021.txt',
          'Person8', 'Fp1', 500,
          'C:\\Users\\mdric\\Downloads\\sustainedattention_shane_12-3-2021 Focus.csv']

file14 = ['C:\\Users\\mdric\\Downloads\\sustainedattention_maceo_12-3-21.txt',
          'Person1', 'T3, T5', 500]

file15 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_Ali_12-6-21.txt',
          'Person9', 'Pz, P3', 500]

file16 = ['C:\\Users\\mdric\\Downloads\\sustainedattention_ali_12-6-21.txt',
          'Person9', 'Pz, P3', 500]

file17 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclose_ali_behindEar_12-6-21.txt',
          'Person9', 'ear', 500]

file18 = ['C:\\Users\\mdric\\Downloads\\sustainedattention_ali_behindear_12-6-21.txt',
          'Person9', 'ear', 500]

file19 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclose_ali_diff_12-6-21.txt',
          'Person9', 'differential', 500]

file20 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_shane_12-6-21.txt',
          'Person8', 'differential', 500]

file21 = ['C:\\Users\\mdric\\Downloads\\sustainedattention_shane_diff_12-6-2021.txt',
          'Person8', 'differential', 500]

file22 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclosed_maceo_12-8-21.txt',
          'Person1', 'T4, F8', 500]

file23 = ['C:\\Users\\mdric\\Downloads\\sustainedattention_maceo_12-8-21.txt',
          'Person1', 'T4, F8', 500,
          'C:\\Users\\mdric\\Downloads\\sustainedattention_maceo_12-8-21_Focus.csv']

file24 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclose_Nathan_12-8-21.txt',
          'Person5', 'differential', 500]

file25 = ['C:\\Users\\mdric\\Downloads\\sustainedattention_Nathan_12-8-21.txt',
          'Person5', 'differential', 500,
          'C:\\Users\\mdric\\Downloads\\sustaainedattention_Nathan_12-8-21_focus.csv']

file26 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclose_Nathan2_12-8-21.txt',
          'Person5', 'differential', 500]

file27 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclose_NathanL_12-8-21.txt',
          'Person5', 'differential', 500]

file28 = ['C:\\Users\\mdric\\Downloads\\sustainedattention_Nathan-L 12-8-21.txt',
          'Person5', 'differential', 500,
          'C:\\Users\\mdric\\Downloads\\sustaainedattention_NathanLeft_12-8-21_focus.csv']

file29 = ['C:\\Users\\mdric\\Downloads\\eyesopeneyesclose_Shane_12-8-21.txt.txt',
          'Person8', 'differential', 500]



#holding response, average reaction time, how accurate response is held, reaction time on 3


x = file23

cols = [0,1,3,4]
data_eeg = np.transpose(np.delete(np.loadtxt(x[0]), cols, 1))

name  = x[1]
place = x[2]
Fs    = x[3]


if len(x) == 5:
    data_focus = pd.read_csv(x[4])

    plt.figure(0)
    plt.plot(data_focus['Trial'], data_focus['React'])
    plt.xlabel('Trial')
    plt.ylim(-1, 0.6)
    plt.ylabel('Reaction Time')
    plt.title('Performance')
    plt.show()    
    

min_freq = 7
max_freq = 80

spect, stimes, sfreqs = multitaper_spectrogram(data_eeg, Fs, frequency_range=[min_freq, max_freq], 
                                               time_bandwidth = 2, plot_on = False,
                                               window_params = [2, 1])

plt.figure(1)
plt.pcolormesh(stimes, sfreqs, spect, shading = 'gouraud')
plt.xlabel('Time (sec)')
plt.ylabel('Frequency (Hz)')
plt.ylim(min_freq,max_freq)
plt.title('{}, {}'.format(name, place))
plt.show()








