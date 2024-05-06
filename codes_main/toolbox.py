#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:03:10 2024

Script to house different functions to use in getWaveform script

@author: asifashraf
"""

##IMPORT
from scipy.signal import hilbert
from scipy.stats import binned_statistic_2d
from PyEMD import EMD
from multitaper import mtspec
import numpy as np



# function for finding nearest value
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def mt_spectra(signal, dt, time_bandwidth, N_tapers):
    
    mt = mtspec.MTSpec(signal, nw=time_bandwidth, kspec = N_tapers, dt=dt)

    freq = mt.freq
    psd  = mt.spec
    ci_lower, ci_upper = mt.jackspec()[:,0], mt.jackspec()[:,1]
    
    return psd, freq, ci_lower, ci_upper
    
def instant_phase(imfs):
    """
    
    Extract analytical signal through Hilbert Transform and then obtain instantaneous
    phase

    Parameters
    ----------
    imfs : Tnp.array
        The 2D array with all the intrinsic mode functions

    Returns
    -------
    phase : np.array
        Instantaneous phase for all imfs

    """

    analytic_signal = hilbert(imfs)  # Apply Hilbert transform to each row
    phase = np.unwrap(np.angle(analytic_signal))  # Compute angle between img and real
    return phase
    
def spectrogram(time,data,dt,period_lims,Ntime_bins = 500, period_bins = None, 
                Nperiod_bins = 500, Nimfs = 10, return_imfs = True):
    """
    Create spectrogram using the HHT approach
    Parameters
    ----------
    time : np.array
        Time vector
    data : bnp.array
        The data
    dt : float
        Sampling interval
    period_lims : np.array or list
        Interval of periods for spectrogram
    Ntime_bins : int, optional
        Number of bins in time dimension. The default is 200.
    Nperiod_bins : int, optional
        Number of bins in period dimension The default is 100.
    Nimfs : int, optional
        Number of imfs to calcualte. The default is 10.
    return_imfs : boolean, optional
        Decides whetehr to output imfs or not. The default is False.
    Returns
    -------
    TYPE
        DESCRIPTION.
    """

    # Compute IMFs with EMD
    emd = EMD(std_thr = .001, range_thr = .01, total_power_thr = 0.01, savr_thr = 0.01, DTYPE=np.float16, str = 'cubic', extrema_detection = 'parabol')
    emd.FIXE_H = 1000000
    imfs = emd(data, time)
    
    # Extract instantaneous phases and frequencies using Hilbert transform
    instant_phases = instant_phase(imfs)
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
    
    
def softmax(x):
    """Compute softmax values for each sets of scores in x along axis 1"""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=1, keepdims=True)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    