#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 14:42:00 2024
script for running frequency decomposition on synthetic signals

@author: asifashraf
"""

from PyEMD import EMD
from hht_tools import find_nearest
import numpy as np
from scipy import signal
from scipy.signal import chirp
import matplotlib as mtplt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from hfsims import cua_envelope
import scipy.fft
from scipy.signal import hilbert
from scipy.stats import binned_statistic_2d
import sys
import pywt


plt.close('all')
##INPUT SYNTHETIC SIGNALS PARAMETERS
#
#make time axis
dt         = 0.01
start_time = 0
end_time   = 50
t          = np.arange(start_time, end_time, dt)
#MAKE the envelop
mag     = 8
dist_km = 75
ptime   = 7
stime   = 15
#SYNTHETIC signal parameters
#assign FOUR different FREQ and CHIRP methods (a, b, c, d)
#1st one is CONSTANT for sine or cosing wave
a = ''; #no methood, because sine or cosine signal
f     = 2
ampl  = 1
#2nd, 3rd, 4th one are the freq range  for Chirp signal
b = 'linear'
f10   = 2
f11   = 6
ampl1 = 2
##
c = 'logarithmic'
f20   = 6
f21   = 11
ampl2 = 1
##
d = 'linear'
f30   = .01
f31   = 20
ampl3 = 1
#for plotting
prfr_Lwidth = 1;

figure_title = '/Users/asifashraf/Documents/Manuscripts/UO_fqDecomp/exp_result/ex_R_R_1.png'

############################

# FUNCTIONS

def fq_plot(time, frequency, amplitude):
    
    plt.pcolormesh(time, frequency, np.abs(amplitude), cmap = 'hot')
    plt.ylim(0, 15)
    plt.plot(aa, [14, 14.2], 'vg')

    plt.plot(t_cut_aroundPeakT,f1_cut, '--c', label = 'f1', linewidth = prfr_Lwidth)
    plt.plot(t_cut_aroundPeakT,f2_cut, '--c', label = 'f2', linewidth = prfr_Lwidth)
    plt.plot(t_cut_aroundPeakT,f3_cut, '--c', label = 'f3', linewidth = prfr_Lwidth)
    plt.plot(t_cut_aroundPeakT,f4_cut, '--c', label = 'f4', linewidth = prfr_Lwidth)

def spectrogram(time,data,dt,period_lims,Ntime_bins = 500, period_bins = None, 
                Nperiod_bins = 500, Nimfs = 10, return_imfs = False):
    
    # Compute IMFs with EMD
    emd = EMD(std_thr = .001, range_thr = .01, total_power_thr = 0.01, 
              savr_thr = 0.01, DTYPE=np.float16, 
              str = 'cubic', extrema_detection = 'parabol')
    emd.FIXE_H = 1000000  # Fixing a high iteration number
    imfs = emd(data, time)
    
    instant_phases = np.unwrap(np.angle(hilbert(imfs)))
    
    # Extract instantaneous phases and frequencies using Hilbert transform
    instant_freqs = abs(np.diff(instant_phases) / (2 * np.pi * dt))
    instant_periods = (1/instant_freqs)
    
    #get amplitude of each imf in absolute untis and in decibels
    amplitudes = abs(imfs)**1
    db = np.log10(amplitudes)
    
    #make dimensions of all array align
    time = time[:-1]
    data = data[:-1]
    amplitudes = amplitudes[:,:-1]
    db = db[:,:-1]
    imfs = imfs[:,:-1]
    
    #use binned statsitc to build the 2D arrayed for the spectrogram
    Nimfs = imfs.shape[0]
    
    # X variable is time
    x =  np.tile(time,(Nimfs,1))
    x = x.ravel()
    
    # Y variable is periods
    y = instant_periods.ravel()
    
    # Z variable
    z = amplitudes.ravel()
    
    # Time Bins
    x_edges = np.linspace(time.min(),time.max(),Ntime_bins)
    
    #Period bins
    if period_bins is None:
        y_edges = np.linspace(period_lims[0],period_lims[1],Nperiod_bins)
    else:
        y_edges = period_bins
        
    #Calcualte the binned statistic
    stat,binx,biny,num = binned_statistic_2d(x, y, values=z,bins = (x_edges,y_edges), statistic='sum')
    
    #Convert to output arrays
    time_out, periods_out = np.meshgrid(binx[1:],biny[1:]) 
    amplitude_out = stat
    
    if return_imfs == False:
        return time_out, periods_out, amplitude_out.T
    
    else:
        return time_out, periods_out, amplitude_out.T, imfs

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

############################

# MAKING SYNTHETIC SIGNALS

t1 = t.max()
#single sine wave
y1   = ampl*np.sin(2*np.pi*f*t)
theoretical_f1 = np.ones(len(t))*f
#1st Chirp
y2   = ampl1*chirp(t, f10, t1, f11, method = b)
theoretical_f2 = f10 + (f11 - f10) * t / t1
#2nd Chirp
y3   = ampl2*chirp(t, f20, t1, f21, method = c)
theoretical_f3 = f20 * (f21/f20) ** (t/t1)
#3rd Chirp
y4   = ampl3*chirp(t, f30, t1, f31, method = d)
theoretical_f4 = f30 + (f31-f30) * t / t1
#COMBINE all signals
y_total = y1 + y2 + y3 + y4

#NOISE (Random) addition
Noise     = np.random.randn(len(y_total))/10
y_total_N = y_total

#ENVELOPING signals
times   = np.linspace(start_time, end_time, len(y_total))
env     = cua_envelope(mag,dist_km,times,ptime, stime, Pcoeff=0,Scoeff=12)

#total Enveloped signal
y_env   = (y_total)*env
y_env_N = y_total_N*env #with NOISE

#get the time for peak accelaration
#index for the peak accelaration
timeIndex_peak = y_env.argmax()
#time of peak accelaration
peak_acc = t[timeIndex_peak]
#for plotting symbol on that time
aa = np.zeros(2)+t[timeIndex_peak]
bb = np.arange(8, 10 ,1)
#make a time range around the peak accelaration to plot
range_t_peakAcc = [peak_acc - 10 , peak_acc + 10]

cut_time = 5
peakT_t_ind      = np.where(t == (find_nearest(t, value = peak_acc)))
peakT_tPlus_ind  = np.where(t == (find_nearest(t, value = peak_acc+cut_time)))
peakT_tminus_ind = np.where(t == (find_nearest(t, value = peak_acc-cut_time)))
y_cut_aroundPeakT  = y_env_N[(peakT_tminus_ind[0][0]):(peakT_tPlus_ind[0][0])]
t_cut_aroundPeakT  = t[(peakT_tminus_ind[0][0]):(peakT_tPlus_ind[0][0])]

#dissect the theoretical frequencies
f1_cut = theoretical_f1[(peakT_tminus_ind[0][0]):(peakT_tPlus_ind[0][0])]
f2_cut = theoretical_f2[(peakT_tminus_ind[0][0]):(peakT_tPlus_ind[0][0])]
f3_cut = theoretical_f3[(peakT_tminus_ind[0][0]):(peakT_tPlus_ind[0][0])]
f4_cut = theoretical_f4[(peakT_tminus_ind[0][0]):(peakT_tPlus_ind[0][0])]


##############################

# FREQUENCY DECOMPOSITION

# general inputs
# nyquist frequency
n_fq = (len(y_cut_aroundPeakT) / (t_cut_aroundPeakT.max() - t_cut_aroundPeakT.min()))/2

# Fourier transform
print('applying FT ...')
fs = 2*n_fq + n_fq/10           # Samping frequency (more than 2X of nyquist fq)
n1 = 100; n2 = 400; n3 = 800    # Length of each segment

ftf_D1, ftt_D1, ftZ_D1 = signal.spectrogram(y_cut_aroundPeakT, fs = fs, nperseg=n1, noverlap = n1 - 10) 
ftf_D2, ftt_D2, ftZ_D2 = signal.spectrogram(y_cut_aroundPeakT, fs = fs, nperseg=n2, noverlap = n2 - 10) 
ftf_D3, ftt_D3, ftZ_D3 = signal.spectrogram(y_cut_aroundPeakT, fs = fs, nperseg=n3, noverlap = n3 - 10)


# Wavelet transform
print('applying WT ...')
wavelet = 'morl'
scales = np.arange(1, 1000)
coeff, wvfq = pywt.cwt(data = y_cut_aroundPeakT, scales = scales, wavelet = wavelet, sampling_period=1/(fs))


# Hilbert Huang transform
print('applying HHT ...')
max_period = 25
min_period = 2*dt
n_bins     = 300
period_bins = np.logspace(np.log10(min_period), np.log10(max_period), n_bins) #logarithmically spaced period bins
timeSpec, periodSpec, ampSpec = spectrogram(t_cut_aroundPeakT, y_cut_aroundPeakT, dt, [min_period, max_period], period_bins = period_bins)


#############################

# PLOTTING

# PLOT input signals

plt.figure(1)
plt.figure(figsize=(8,6))
ylims=[-4,4]
plt.subplot(7,1,1)
plt.plot(t,y1,label='sine')
plt.ylim(ylims)
#plt.legend()
#plt.grid()
#plt.title('f1')
plt.subplot(7,1,2)
plt.plot(t,y2,label=b)
plt.ylim(ylims)
#plt.legend()
#plt.grid()
#plt.title('f2')
plt.subplot(7,1,3)
plt.plot(t,y3,label=c)
plt.ylim(ylims)
#plt.legend()
#plt.grid()
#plt.title('f3')
plt.subplot(7,1,4)
plt.plot(t,y4,label=d)
plt.ylim(ylims)
#plt.legend()
#plt.grid()
#plt.title('f4')
plt.subplot(7,1,5)
plt.plot(t,Noise,label='Random')
plt.ylim(ylims)
#plt.legend()
#plt.grid()
#plt.title('Noise')
plt.subplot(7,1,6)
plt.plot(t,y_total,label='combined without Noise')
plt.ylim(ylims)
#plt.legend()
#plt.grid()
plt.subplot(7,1,7)
plt.plot(t,y_total_N,label='combined with Noise')
plt.ylim(ylims)
#plt.legend()
#plt.grid()
plt.tight_layout()


plt.figure(11)
plt.figure(figsize=(8,6))
plt.subplot(2,1,1)
plt.plot(t, env, '--r')
plt.subplot(2,1,2)
plt.plot(t, y_env, '-b')

# PLOT Freq Decomp
plt.figure(2)
plt.figure(figsize = (12,6))

plt.subplot(1,6,1)
# time axis
tt1 = np.linspace(t_cut_aroundPeakT[0], t_cut_aroundPeakT[(len(t_cut_aroundPeakT)-1)],len(ftt_D1))
fq_plot(time = tt1, frequency = ftf_D1, amplitude = ftZ_D1)
plt.title('FT (neprseg=' + str(n1) + ')' )

plt.subplot(1,6,2)
# time axis
tt2 = np.linspace(t_cut_aroundPeakT[0], t_cut_aroundPeakT[(len(t_cut_aroundPeakT)-1)],len(ftt_D2))
fq_plot(time = tt2, frequency = ftf_D2, amplitude = ftZ_D2)
plt.title('FT (neprseg=' + str(n2) + ')' )

plt.subplot(1,6,3)
# time axis
tt3 = np.linspace(t_cut_aroundPeakT[0], t_cut_aroundPeakT[(len(t_cut_aroundPeakT)-1)],len(ftt_D3))
fq_plot(time = tt3, frequency = ftf_D3, amplitude = ftZ_D3)
plt.title('FT (neprseg=' + str(n3) + ')' )

plt.subplot(1,6,4)
t_wt = np.linspace(t_cut_aroundPeakT[0], t_cut_aroundPeakT[len(t_cut_aroundPeakT)-1], coeff.shape[1])
fq_plot(time = t_wt, frequency = wvfq, amplitude = coeff)
plt.title('WT (' + wavelet + ')')

plt.subplot(1,6,5)
fq_plot(time = timeSpec, frequency = 1/periodSpec, amplitude = ampSpec)
plt.title('HHT')


plt.subplot(1,6,6)
plt.pcolormesh(timeSpec, 1/periodSpec, np.abs(ampSpec), norm = colors.LogNorm(vmin = 10, vmax = ampSpec.max()), cmap = 'hot')
plt.ylim(0, 15)
plt.plot(aa, [14, 14.2], 'vg')
plt.colorbar()
plt.plot(t_cut_aroundPeakT,f1_cut, '--c', label = 'f1', linewidth = prfr_Lwidth)
plt.plot(t_cut_aroundPeakT,f2_cut, '--c', label = 'f2', linewidth = prfr_Lwidth)
plt.plot(t_cut_aroundPeakT,f3_cut, '--c', label = 'f3', linewidth = prfr_Lwidth)
plt.plot(t_cut_aroundPeakT,f4_cut, '--c', label = 'f4', linewidth = prfr_Lwidth)


plt.savefig(figure_title, dpi =  500, format = 'png')



















