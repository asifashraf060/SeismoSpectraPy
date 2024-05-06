#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 14:42:00 2024
script for running frequency decomposition on synthetic signals

@author: asifashraf
"""

from PyEMD import EMD
import numpy as np
from scipy import signal
from scipy.signal import chirp

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.signal import hilbert
from scipy.stats import binned_statistic_2d

import pywt

plt.close('all')
##INPUT SYNTHETIC SIGNALS PARAMETERS
#
#make time axis
dt         = 0.005
start_time = 0
end_time   = 50
t          = np.arange(start_time, end_time, dt)
#MAKE the envelop
mag     = 5
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
f30   = 8
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

def cua_envelope(M,dist_in_km,times,ptime,stime,Pcoeff=0,Scoeff=12):
    '''
    Cua envelopes, modified from Ran Nof's Cua2008 module
    '''
    from numpy import where,sqrt,exp,log10,arctan,pi,zeros
    
    a = [0.719, 0.737, 0.801, 0.836, 0.950, 0.943, 0.745, 0.739, 0.821, 0.812, 0.956, 0.933,
            0.779, 0.836, 0.894, 0.960, 1.031, 1.081, 0.778, 0.751, 0.900, 0.882, 1.042, 1.034]
    b = [-3.273e-3, -2.520e-3, -8.397e-4, -5.409e-4, -1.685e-6, -5.171e-7, -4.010e-3, -4.134e-3,
                -8.543e-4, -2.652e-6, -1.975e-6, -1.090e-7, -2.555e-3, -2.324e-3, -4.286e-4, -8.328e-4,
                -1.015e-7, -1.204e-6, -2.66e-5, -2.473e-3, -1.027e-5,- 5.41e-4, -1.124e-5, -4.924e-6]
    d = [-1.195, -1.26, -1.249, -1.284, -1.275, -1.161, -1.200, -1.199, -1.362, -1.483, -1.345, -1.234,
                -1.352, -1.562, -1.440, -1.589, -1.438, -1.556, -1.385, -1.474, -1.505, -1.484, -1.367, -1.363]
    c1 = [1.600, 2.410, 0.761, 1.214, 2.162, 2.266, 1.752, 2.030, 1.148, 1.402, 1.656, 1.515,
                1.478, 2.423, 1.114, 1.982, 1.098, 1.946, 1.763, 1.593, 1.388, 1.530, 1.379, 1.549]
    c2 = [1.045, 0.955, 1.340, 0.978, 1.088, 1.016, 1.091, 1.972, 1.100, 0.995, 1.164, 1.041,
                1.105, 1.054, 1.110, 1.067, 1.133, 1.091, 1.112, 1.106, 1.096, 1.04, 1.178, 1.082]
    e = [-1.065, -1.051, -3.103, -3.135, -4.958, -5.008, -0.955, -0.775, -2.901, -2.551, -4.799, -4.749,
                -0.645, -0.338, -2.602, -2.351, -4.342, -4.101, -0.751, -0.355, -2.778, -2.537, -4.738, -4.569]
    sig_uncorr = [0.307, 0.286, 0.268, 0.263, 0.284, 0.301, 0.288, 0.317, 0.263, 0.298, 02.83, 0.312,
                0.308, 0.312, 0.279, 0.296, 0.277, 0.326, 0.300, 0.300, 0.250, 0.270, 0.253, 0.286]
    sig_corr = [0.233, 0.229, 0.211, 0.219, 0.239, 0.247, 0.243, 0.256, 0.231, 0.239, 0.254, 0.248,
                0.243, 0.248, 0.230, 0.230, 0.233, 0.236, 0.238, 0.235, 0.220, 0.221, 0.232, 0.230]
    
    # Coefficienstime for eqn: log(env_param) = alpha*M + beta*R + delta*logR + mu
    # Coefficienstime and equation for t_rise (rise time):
    
    alpha_t_rise = [0.06, 0.07, 0.06, 0.07, 0.05, 0.05, 0.06, 0.06, 0.06, 0.06, 0.08, 0.067,
                0.064, 0.055, 0.093, 0.087, 0.109, 0.12, 0.069, 0.059, 0.116, 0.11, 0.123, 0.124]  
    beta_t_rise = [5.5e-4, 1.2e-3, 1.33e-3, 4.35e-4, 1.29e-3, 1.19e-3, 7.45e-4, 5.87e-4, 7.32e-4, 1.08e-3, 1.64e-3, 1.21e-3,
                0, 1.21e-3, 0, 4.0e-4, 7.68e-4, 0, 0, 2.18e-3, 0, 1.24e-3, 1.3e-3, 0]
    delta_t_rise = [0.27, 0.24, 0.23, 0.47, 0.27, 0.47, 0.37, 0.23, 0.25, 0.22, 0.13, 0.28,
                0.48, 0.34, 0.48, 0.49, 0.38, 0.45, 0.49, 0.26, 0.503, 0.38, 0.257, 0.439]
    mu_t_rise = [-0.37, -0.38, -0.34, -0.68, -0.34, -0.58, -0.51, -0.37, -0.37, -0.36, -0.33, -0.46,
                -0.89, -0.66, -0.96, -0.98, -0.87,-0.89,-0.97, -0.66, -1.14, -0.91, -0.749, -0.82]
    
    # Coefficienstime and equation for delta_t (wave duration):
    
    alpha_delta_t = [0, 0.03, 0.054, 0.03, 0.047, 0.051, 0, 0, 0.046, 0.031, 0.058, 0.043,
                0, 0.028, 0.02, 0.028, 0.04, 0.03, 0.03, 0.03, 0.018, 0.017, 0.033, 0.023]
    beta_delta_t = [2.58e-3, 2.37e-3, 1.93e-3, 2.03e-3, 0, 1.12e-3, 2.75e-3, 1.76e-3, 2.61e-3, 1.7e-3, 2.02e-3, 9.94e-4,
                -4.87e-4, 0, 0, 0, 1.1e-3, 0, -1.4e-3, -1.78e-3, 0, -6.93e-4, 2.6e-4, -7.18e-4]
    delta_delta_t = [0.21, 0.39, 0.16, 0.289, 0.45, 0.33, 0.165, 0.36, 0, 0.26, 0, 0.19,
                0.13, 0.07, 0, 0.046, -0.15, 0.037, 0.22, 0.307, 0, 0.119, 0, 0.074]
    mu_delta_t = [-0.22, -0.59, -0.36, -0.45, -0.68, -0.59, -0.245, -0.48, -0.213, -0.52, -0.253, -0.42,
                0.0024, -0.102, 0.046, -0.083, 0.11, -0.066, -0.17, -0.66, -0.072, -0.05, -0.015, -0.005]
    
    # Coefficienstime and equation for tau (decay):
    
    alpha_tau = [0.047, 0.087, 0.054, 0.0403, 0, 0.035, 0.03, 0.057, 0.03, 0.0311, 0.05, 0.052,
                0.037, 0.0557, 0.029, 0.045, 0.029, 0.038, 0.031, 0.06, 0.04, 0.051, 0.024, 0.022]  
    beta_tau = [0, -1.89e-3, 5.37e-5, -1.26e-3, 0, -1.27e-3, 2.75e-3, -1.36e-3, 8.6e-4, -6.4e-4, 8.9e-4, 0,
                0, -8.2e-4, 8.0e-4, -5.46e-4, 0, -1.34e-3, 0, -1.45e-3, 9.4e-4, -1.41e-3, 0, -1.65e-3]
    delta_tau = [0.48, 0.58, 0.41, 0.387, 0.19, 0.19, 0.58, 0.63, 0.35, 0.44, 0.16, 0.12,
                0.39, 0.51, 0.25, 0.46, 0.36, 0.48, 0.34, 0.51, 0.25, 0.438, 0.303, 0.44]
    gamma_tau = [0.82, 0.58, 0.73, 0.58, 0, 0, 0, 0, 0, 0, 0, 0, 1.73, 1.63, 1.61, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mu_tau = [-0.75, -0.87, -0.51, -0.372, -0.07, -0.03, -0.97, -0.96, -0.62, -0.55, -0.387, -0.166,
                -0.59, -0.68, -0.31, -0.55, -0.38, -0.39, -0.44, -0.60, -0.34, -0.368, -0.22, -0.19]
    avg_gamma = 0.15

    
    # Coefficienstime and equation for gamma (decay):
    alpha_gamma = [-0.032, -0.048, -0.044, -0.0403, -0.062, -0.061, -0.027, -0.024, -0.039, -0.037, -0.052, -0.066,
                -0.014, -0.015, -0.024, -0.031, -0.025, -2.67e-2, -0.0149, -0.0197, -0.028, -0.0334, -0.015, -0.0176] #<--should be =-0.048 for i=1? not =-0.48?
    beta_gamma = [-1.81e-3, -1.42e-3, -1.65e-3, -2.0e-3, -2.3e-3, -1.9e-3, -1.75e-3, -1.6e-3, -1.88e-3, -2.23e-3, -1.67e-3, -2.5e-3,
                -5.28e-4, -5.89e-4, -1.02e-3, -4.61e-4, -4.22e-4, 2.0e-4, -4.64e-4, 0, -8.32e-4, 0, 0, 5.65e-4]
    delta_gamma = [-0.1, -0.13, -0.16, 0, 0, 0.11, -0.18, -0.24, -0.18, -0.14, -0.21, 0,
                -0.11, -0.163, -0.055, -0.162, -0.145, -0.217, -0.122, -0.242, -0.123, -0.21, -0.229, -0.25]
    tau_gamma = [0.27, 0.26, 0.33, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0.38, 0.39, 0.36, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    mu_gamma = [0.64, 0.71, 0.72, 0.578, 0.61, 0.39, 0.74, 0.84, 0.76, 0.71, 0.849, 0.63,
                0.26, 0.299, 0.207, 0.302, 0.262, 0.274, 0.255, 0.378, 0.325, 0.325, 0.309, 0.236]
    avg_gamma = 0.15
    

    stat_err = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sta_corr =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # coefficienstime
    t_rise_p = 10**(alpha_t_rise[Pcoeff] * M + beta_t_rise[Pcoeff] * dist_in_km + delta_t_rise[Pcoeff] * log10(dist_in_km) + mu_t_rise[Pcoeff])
    t_rise_s = 10**(alpha_t_rise[Scoeff] * M + beta_t_rise[Scoeff] * dist_in_km + delta_t_rise[Scoeff] * log10(dist_in_km) + mu_t_rise[Scoeff])
    delta_t_p = 10**(alpha_delta_t[Pcoeff] * M + beta_delta_t[Pcoeff] * dist_in_km + delta_delta_t[Pcoeff] * log10(dist_in_km) + mu_delta_t[Pcoeff])
    delta_t_s = 10**(alpha_delta_t[Scoeff] * M + beta_delta_t[Scoeff] * dist_in_km + delta_delta_t[Scoeff] * log10(dist_in_km) + mu_delta_t[Scoeff])
    tau_p = 10**(alpha_tau[Pcoeff] * M + beta_tau[Pcoeff] * dist_in_km + delta_tau[Pcoeff] * log10(dist_in_km) + mu_tau[Pcoeff])
    tau_s = 10**(alpha_tau[Scoeff] * M + beta_tau[Scoeff] * dist_in_km + delta_tau[Scoeff] * log10(dist_in_km) + mu_tau[Scoeff])
    gamma_p = 10**(alpha_gamma[Pcoeff] * M + beta_gamma[Pcoeff] * dist_in_km + delta_gamma[Pcoeff] * log10(dist_in_km) + mu_gamma[Pcoeff])
    gamma_s = 10**(alpha_gamma[Scoeff] * M + beta_gamma[Scoeff] * dist_in_km + delta_gamma[Scoeff] * log10(dist_in_km) + mu_gamma[Scoeff])
    
    # Other variable (turn on saturation for larger evenstime?)
    C_p = (arctan(M-5) + (pi/2))*(c1[Pcoeff]*exp(c2[Pcoeff] * (M-5)))
    C_s = (arctan(M-5) + (pi/2))*(c1[Scoeff]*exp(c2[Scoeff] * (M-5)))
    R1 = sqrt(dist_in_km**2 + 9)
    
    # Basic AMplitudes
    A_p = 10**(a[Pcoeff]*M + b[Pcoeff]*(R1 + C_p) + d[Pcoeff]*log10(R1+C_p) + e[Pcoeff]+(sta_corr[Pcoeff]) + stat_err[Pcoeff])
    A_s = 10**(a[Scoeff]*M + b[Scoeff]*(R1 + C_s) + d[Scoeff]*log10(R1+C_s) + e[Scoeff]+(sta_corr[Scoeff]) + stat_err[Scoeff])
    
    # calculate envelope (ENV)
    envelope = zeros(len(times))

    # P envelope
    indx = where((times>=ptime) & (times<ptime+t_rise_p)) # between trigger and rise time
    if len(indx): envelope[indx] = (A_p/t_rise_p*(times[indx]-ptime)) # make sure we have data in that time frame and get envelope
    indx = where((times>=ptime+t_rise_p) & (times<ptime+t_rise_p+delta_t_p)) # flat area
    if len(indx): envelope[indx] = A_p # make sure we have data in that time frame and get envelope
    indx = where(times>ptime+t_rise_p+delta_t_p) # coda
    if len(indx): envelope[indx] = (A_p/((times[indx]-ptime-t_rise_p-delta_t_p+tau_p)**gamma_p)) # make sure we have data in that time frame and get envelope
    
    # S envelope
    indx = where((times>=stime) & (times<stime+t_rise_s)) # between trigger and rise time
    if len(indx): envelope[indx] += (A_s/t_rise_s*(times[indx]-stime)) # make sure we have data in that time frame and get envelope
    indx = where((times>=stime+t_rise_s) & (times<stime+t_rise_s+delta_t_s)) # flat area
    if len(indx): envelope[indx] += A_s # make sure we have data in that time frame and get envelope
    indx = where(times>stime+t_rise_s+delta_t_s) # coda
    if len(indx): envelope[indx] += (A_s/((times[indx]-stime-t_rise_s-delta_t_s+tau_s)**gamma_s)) # make sure we have data in that time frame and get envelope
    
    return envelope

def zero_noise(t, envelop, fq_lw, fq_up, amp):
    '''
    Parameters
    ----------
    t : 
        1-D array defining the time axis of the seismogram.
    envelop : 
        Envelop array developed from cua_envelop.
    fq_lw :
        lowest frequency (Hz) of the chirp signal.
    fq_up : 
        highest frequency (Hz) of the chirp signal.
    amp : 
        amplitude of the Chirp signal.

    Returns
    -------
    1-D array
        A seismogram that has random noise before the P-wave arrival.
    '''
    chirpS     = amp*chirp(t, fq_lw, np.max(t), fq_up, method = 'linear')
    noise      = np.random.randn(len(chirpS))/200
    y_new_wN   = chirpS + noise
    env_new    = envelop + y_new_wN
    #env_new[np.where(envelop == 0)] = y_new_wN
    return (y_total * env_new)
    
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
y_env       = (y_total)*env
y_env_N     = y_total_N*env #with NOISE

## ADD noise before P-wave
y_env_zeroN  = zero_noise(t = t, envelop = env, fq_lw = 0, fq_up = 50, amp = 0.000001)

ar       = np.loadtxt('/Users/asifashraf/Documents/Freq_exp/dataN2.txt')
env0_ind = np.where(env == 0)
env0_ind = env0_ind[0]
ind1     = env0_ind[len(env0_ind)-len(ar):]

y_env_zeroN2 = y_env_zeroN
y_env_zeroN2[ind1] = ar




max_period = 25
min_period = 2*dt
n_bins     = 300
period_bins = np.logspace(np.log10(min_period), np.log10(max_period), n_bins) #logarithmically spaced period bins

timeSpec_y, periodSpec_y, ampSpec_y = spectrogram(t, y_env, dt, [min_period, max_period], period_bins = period_bins)
timeSpec_yZN, periodSpec_yZN, ampSpec_yZN = spectrogram(t, y_env_zeroN, dt, [min_period, max_period], period_bins = period_bins)


plt.figure()
plt.figure(figsize=(8,6))
plt.subplot(2, 3, 1)
plt.plot(t, y_env_N)

plt.subplot(2, 3, 2)
plt.plot(t, y_env_N)
plt.xlim([0, 10])
plt.ylim([-2, 2])

plt.subplot(2, 3, 3)
plt.pcolormesh(timeSpec_y, 1/periodSpec_y, np.abs(ampSpec_y), norm = colors.LogNorm(vmin = 0.1, vmax = ampSpec_y.max()), cmap = 'hot')

plt.subplot(2, 3, 4)
plt.plot(t, y_env_zeroN)

plt.subplot(2, 3, 5)
plt.plot(t, y_env_zeroN)
plt.xlim([0, 10])
plt.ylim([-2, 2])

plt.subplot(2, 3, 6)
plt.pcolormesh(timeSpec_yZN, 1/periodSpec_yZN, np.abs(ampSpec_yZN), norm = colors.LogNorm(vmin = 0.1, vmax = ampSpec_y.max()), cmap = 'hot')

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



















