#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:33:59 2024

@author: asifashraf
"""
## INPUT
root_dir  = '/Users/asifashraf/Documents/Fq_Decomp/example/2000_2023'
min_eq    = 15;

## IMPORT
from pathlib import Path
from scipy.signal import find_peaks

from toolbox import spectrogram
from toolbox import mt_spectra
from toolbox import softmax

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd

from obspy.core import read

import os
import glob


# make a list of directories that contain mseed files
dir_DataFolder = []

for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.mseed'):
            # If a .mseed file is found, save the directory if not already saved
            direc = Path(root).resolve()
            if direc not in dir_DataFolder:
                dir_DataFolder.append(direc)

print('Found ' + str(len(dir_DataFolder)) + ' station folders with data')

for i in range(332, len(dir_DataFolder)):
    
    ## LOAD FILES FROM THE FOLDER
    print('Station: (' + str(i+1) + '/' + str(len(dir_DataFolder)) + ')')
    folder_path = dir_DataFolder[i]
    
    # Station info
    station_text = glob.glob(str(folder_path) + '/st_info.txt')
    df = pd.read_csv(station_text[0], delim_whitespace=True)
    st_code, net_code, st_lat, st_lon = df['station'][0], df['netcode'][0], float(df['lat']), float(df['lon'])
    
    # Earthquake info
    eq_text = glob.glob(str(folder_path) + '/eq_info.txt')
    df = pd.read_csv(eq_text[0], delim_whitespace=True)

    # Data files
    mseed_files = glob.glob(str(folder_path) + '/*.mseed')
    print('    Found ' + str(len(mseed_files)) + ' mseed files for ' +str(len(df)) +' earthquakes')
    
    if len(df)>=min_eq:
    
        # Output directory for results
        try:
            os.mkdir(str(folder_path) + '/Analysis')
        except:
            pass
        
        
        ## FREQUENCY DECOMPOSITION for each earthquake
        
        plt.close('all')
        # empty arrays to store the results
        no    = [] 
        name  = [] 
        trc   = [] 
        lat   = [] 
        lon   = [] 
        dist  = [] 
        dp    = [] 
        mg    = []
        dm_fq = []
        mx_fq = []
        errs  = []
        for j in np.arange(0, len(df)):
            
            # specific eq info
            eq_no,  eq_name, eq_time = df['no'][j], df['name'][j], df['time'][j]
            eq_lat, eq_lon,  eq_dist, eq_dp, eq_mg = df['lat'][j], df['lon'][j], df['dist'][j], df['depth'][j], df['mag'][j]
            hypo_dist = round(np.sqrt(eq_dist**2 + eq_dp**2))
            
            print('        EQ (' + str(j) + '/' + str(len(df)) + ') name: ' + eq_name)
                        
            # PATH for that earthquake file
            mseed_path = glob.glob(root_dir + '/' + str(st_code) + '_' + net_code + '/_Eq' +  str(eq_no) + '_' + eq_name + '_spec_tr*' + '.mseed')
            
            # if data exists for this earthquake
            if len(mseed_path)>0:
                
                plt.close('all')
                plt.figure(1, figsize = (40,6))
                for k in np.arange(0, len(mseed_path)): 
                    # Read the mseed file
                    wv   = read(mseed_path[k], attach_response = True)
                    tr   = wv[0]
                    plt.subplot(1, len(mseed_path)+1, k+1)
                    plt.plot(tr.data)
                    plt.xlabel('s:'+str(tr.stats.starttime.hour)+'_'+str(tr.stats.starttime.minute)+'_'+str(tr.stats.starttime.minute)
                               + ' e:'+ str(tr.stats.endtime.hour)+'_'+str(tr.stats.endtime.minute)+'_'+str(tr.stats.endtime.minute))
                plt.subplot(1, len(mseed_path)+1, k+2)
                plt.plot(np.arange(1, len(mseed_path)+1), np.zeros(len(mseed_path))+2, '+g')
                plt.plot(len(mseed_path)+1, 2, '*r')
                plt.title('EQ: ' + str(eq_time))
                plt.ylim([0, 3])
                tr_clck = np.asarray(plt.ginput(n = 100, timeout = 0, show_clicks=True))
    
                # preferred trace number
                tr_prf = []
                for ii in range(len(tr_clck)):
                    tr_prf.append(round(tr_clck[ii][0]) - 1)
                
                if any(kk<len(mseed_path) for kk in tr_prf): # If the seleceted trace number is valid, continue
                                                              # otherwise, skipping the earthquake
                    
                    tr_total = np.array([])
                    tr_npts = 0
                    tr_tt = 0
                    for jj in range(len(tr_prf)):  # loop through every selected trace and add them
                    
                        tr = read(mseed_path[tr_prf[jj]], attach_response = True)
                        tr = tr[0]
                        tr_total = np.concatenate((tr_total, tr.data), axis=None)
            
                        tt   = (tr.stats.npts/tr.stats.sampling_rate)   #total time of the seismogram
                        tr_tt = tt + tr_tt
                        
                        npts = tr.stats.npts                            #number of samples in seismogram
                        tr_npts = tr_npts + npts
                        
                    dt = tr_tt/tr_npts                                  #t1-t2
                    t  = np.linspace(0,tr_tt,tr_npts)
                
                    print('            Selected trace no: ' +  str(tr_prf))
    
                    plt.close('all')
                    plt.plot(t, tr_total, '-b')
                    plt.title('Choose two points')
                    t12 = np.asarray(plt.ginput(n = 2, timeout = 0))
                    
                    x1, x2 = t12[:,0]
                    ind1   = np.where(abs(t-x1) == min(abs(t-x1)))
                    ind2   = np.where(abs(t-x2) == min(abs(t-x2)))
                    
                    tt   = t # need to use it later
                    t    = t[int(ind1[0]):int(ind2[0])]
                    data = tr_total[int(ind1[0]):int(ind2[0])]
                            
                    # MULTITAPER SPECTRAL ANALYSIS
                    print('           ' + ' Calculating Multitaper PSD ... ')
                    
                    psd, freq, ci_lower, ci_upper = mt_spectra(data, dt, 4, 7)
                    
                    ### automated picking peaks of psd
                    FqF_ind  = np.where(freq>1)                # indices to filter for fq > 1
                    Fq_filt  = freq[FqF_ind[0]]                # apply that filter for freq
                    psd_filt = psd[FqF_ind[0]]                 # and the PSD array
                    psd_smth = np.convolve(psd_filt.flatten(), np.ones(10)/10, mode='valid') # smoothen out the psd with moving average to make fewer peaks
                    fq_smth  = Fq_filt[range(len(psd_smth))]
                    peak_ind = find_peaks(psd_smth)            # find the peaks
                    psd_peak = psd_smth[peak_ind[0]]           # psd for the peaks
                    fq_peak  = fq_smth[peak_ind[0]]            # fq for the peaks
                    
                    threshold = np.percentile(psd_peak, 98)    # only choose the top 5% peaks
                    top_5     = np.where(psd_peak > threshold)
                    psd_Tpeak = psd_peak[top_5]
                    fq_Tpeak  = fq_peak[top_5]


            
                    # EMPERICAL MODE DECOMPOSITION
                    print('           ' + ' Calculating EMD ... ')
                
                    min_period  = 2*dt #Nyquist
                    max_period  = 500
                    n_bins      = 100
                    period_bins = np.logspace(np.log10(min_period), np.log10(max_period), n_bins)
                    timeSpec, periodSpec, ampSpec, IMF = spectrogram(t, data, dt, [min_period, max_period], period_bins = period_bins)
            
                    norm_ampSpec = (softmax(ampSpec)/np.max(softmax(ampSpec)))
            
                    plt.close('all')
                    plt.figure(figsize = (20,5))
            
                    plt.subplot(1,5,1)
                    plt.plot(tt, tr_total, '-b')
                    plt.plot([x1, x2], [0, 0], '+r')
                    plt.xlabel('time (sec)')
                    plt.ylabel('accelaration (m/s)')
                    plt.title('seismogram no ' + str(j) + ' (EQ:' + eq_name + ')')
                    
                    plt.subplot(1,5,2)
                    plt.semilogx(freq,psd, 'b', linewidth=3)
                    plt.semilogx(fq_smth,psd_smth, '--y', linewidth=1)
                    plt.plot(fq_Tpeak, psd_Tpeak, '+r')
                    plt.xlabel('Frequency (Hz)')
                    plt.ylabel('amplitude')
                    plt.title('PSD')
                    
                    # Freq from PSD
                    dominant_fq = fq_Tpeak[psd_Tpeak.argmax()]                  # dominant frequency
                    max_fq      = max(fq_Tpeak)                                 # maximum frequency
                    err         = np.std(fq_Tpeak * psd_Tpeak/psd_Tpeak.max())  # error of picking
                    
                    
                    plt.subplot(1,5,3)
                    plt.pcolormesh(timeSpec, 1/periodSpec, np.abs(ampSpec), cmap='hot')
                    plt.ylim([0, fq_Tpeak.max()+2])
                    plt.plot(np.zeros(2)+t[data.argmax()], [0, fq_Tpeak.max()], '-g')
                    for kk in fq_Tpeak:
                        plt.plot(t, np.zeros(len(t))+kk, '--c', linewidth = .5)
                    plt.plot(t, np.zeros(len(t))+dominant_fq, '--c', linewidth = 1)
                    plt.ylabel('Frequency (Hz)')
                    plt.xlabel('Time (sec)')
                    plt.title('EMD')
                    
                    plt.subplot(1,5,4)
                    plt.pcolormesh(timeSpec, 1/periodSpec, np.abs(ampSpec), norm = colors.LogNorm(vmin = 0.0001, vmax = ampSpec.max()), cmap='hot')
                    plt.ylim([0, fq_Tpeak.max()])
                    plt.plot(np.zeros(2)+t[data.argmax()], [0, fq_Tpeak.max()], '-g')
                    plt.ylabel('Frequency (Hz)')
                    plt.xlabel('Time (sec)')
                    plt.title('EMD (log-scale)')
                    
                    plt.subplot(1,5,5)
                    plt.pcolormesh(timeSpec, 1/periodSpec, norm_ampSpec, vmin = np.percentile(norm_ampSpec.flatten(), 99), vmax = np.max(norm_ampSpec), cmap='hot')
                    plt.ylim([0, fq_Tpeak.max()])
                    plt.plot(np.zeros(2)+t[data.argmax()], [0, fq_Tpeak.max()], '--g', linewidth = .5)
                    for kk in fq_Tpeak:
                        plt.plot(t, np.zeros(len(t))+kk, '--c', linewidth = .5)
                    plt.plot(t, np.zeros(len(t))+dominant_fq, '--c', linewidth = 1)
                    plt.ylabel('Frequency (Hz)')
                    plt.xlabel('Time (sec)')
                    plt.title('EMD (norm amp)')
                    
                    plt.tight_layout()
        
                    plt.savefig(str(folder_path) + '/Analysis/' + eq_name +'.png',format = 'png' )
                    
                    # append the results into the empty arrays
                    no.append(eq_no)
                    name.append(eq_name) 
                    dist.append(hypo_dist)
                    lat.append(eq_lat) 
                    lon.append(eq_lon) 
                    dp.append(eq_dp) 
                    mg.append(eq_mg)
                    dm_fq.append(round(dominant_fq[0]))
                    mx_fq.append(round(max_fq[0]))
                    errs.append(round(err))
                    
                else:
                    print('            skipping EQ ...')
       
        # write and save the picking
        with open(str(folder_path) + '/Analysis/EQ_fq_decomp_result.txt' , 'w') as file:            
            file.write(f"{'no':<15}{'name':<15}{'lat':<15}{'lon':<15}{'depth':<15}{'mag':<15}{'dist':<15}{'fq':<15}{'mxFq':<15}{'err':<15}\n")
            for eq_nums, eq_names, eq_lats, eq_lons, eq_depths, eq_mags, eq_dists, eq_fq, eq_fqM, eq_err in zip(no, name, lat, lon, dp, mg, dist, dm_fq, mx_fq, errs):
                file.write(f"{eq_nums:<15}{eq_names:<15}{eq_lats:<15}{eq_lons:<15}{eq_depths:<15}{eq_mags:<15}{eq_dists:<15}{eq_fq:<15}{eq_fqM:<15}{eq_err:<15}\n")

            else:
                print('             cant find mseed')
    else:
        print('    Does not exceed minimum eq requirement')
 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

