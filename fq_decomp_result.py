#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:27:23 2024

@author: asifashraf
"""


##IMPORT
from datetime import datetime
from scipy import signal
from PyEMD import EMD
from pathlib import Path
#from fq_estimation_tools import spectrogram
from hht_tools import spectrogram
from fq_estimation_tools import mt_spectra
from fq_estimation_tools import find_nearest
import pywt
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.widgets import Button, TextBox
import numpy as np
import urllib.request, json
# import libcomcat
# Local imports
# from libcomcat.dataframes import get_detail_data_frame
from libcomcat.search import search,get_event_by_id
import pandas as pd
#import obspy
from obspy.core import read
from obspy.core import UTCDateTime
from obspy.clients.fdsn.client import Client
#misc..
import scipy
import csv
import utm
import time
import os
import glob

#INPUT
root_dir  = '/Users/asifashraf/Documents/Freq_exp/Station_basis'

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

for i in np.arange(0, len(dir_DataFolder)):
    
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

    df_fq = pd.read_csv(str(folder_path) + '/Analysis/EQ_fq_decomp_result.txt', delim_whitespace=True)
    
    eq_no, eq_name, eq_lat, eq_lon = df_fq['no'], df_fq['name'], df['lat'], df['lon']
    eq_mag, eq_dist, eq_fq, eq_err = df_fq['mag'], df_fq['dist'], df_fq['fq'], df_fq['err']
    
    plt.close('all')
    plt.figure()
    plt.scatter(eq_dist, eq_mag, np.flip(eq_err*20+10), eq_fq, cmap = 'cool')
    plt.colorbar()
    plt.ylim([eq_mag.min()-.5, eq_mag.max() + .5])
    plt.title('Fq distribution for station ' + st_code)
    plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_fqDecomp_distVSmag.png',format = 'png' )
    
    # make a plot with annotation
    ann = []
    for k in (eq_name):
        ann.append(str(k))
    n = []
    for j in eq_fq:
        n.append(str(j))
    
    plt.close('all')
    plt.figure()
    plt.scatter(eq_dist, eq_mag, np.flip(eq_err*20+10), eq_fq, cmap = 'cool')
    plt.colorbar()
    plt.ylim([eq_mag.min()-.5, eq_mag.max() + .5])
    for jk in range(len(eq_dist)):
        plt.annotate(n[jk], (eq_dist[jk]+.3, eq_mag[jk]), va='top', ha='left')
    plt.title('Fq distribution for station ' + st_code)
    plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_fqDecomp_distVSmag_withAnnotation.png', dpi =  500, format = 'png')

        
    





















