#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 18:36:45 2023

@author: asifashraf
"""

##IMPORT
from datetime import datetime
from scipy import signal
from PyEMD import EMD
#from fq_estimation_tools import spectrogram
from fq_estimation_tools import mt_spectra
from fq_estimation_tools import find_nearest
import pywt
import matplotlib.pyplot as plt
import matplotlib
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
import csv
import utm
import time
import os

## INPUT
# Directory of station file
theFile = '/Users/asifashraf/Downloads/gmap-stations (3).txt'
# Radius around the station
rd = 200 # in km
# length of second to cut the seismogram in time
lsec = 200
# Minimum magnitude to search EQ for
minmagnitude  = 4
# start and end time for searching
starttime     = datetime(2019, 7, 3, 0, 0) #YY/MM/DD
endtime       = datetime(2019, 7, 7, 0, 0)
# Channel
chn           = 'HNE'
# directory to store the result
out_dir = '/Users/asifashraf/Documents/Freq_exp/Station_basis/';

## CALCULATION

# importing data from station file
df = pd.read_csv(theFile, comment='#', delimiter='|')

netcode = df.iloc[:,0]
sta     = df.iloc[:,1]
lat     = df.iloc[:,2]
lon     = df.iloc[:,3]

for i in np.arange(0, len(sta)): #range(len(netcode)):

    # assign station parameters  
    net = netcode[i]
    st  = sta[i]
    lt  = lat[i]
    ln  = lon[i]
    utm_result = utm.from_latlon(lt, ln)
    
    print('Searching station: ' + st + ' (' +str(i) + '/' + str(len(netcode))+ ') ..')
    
    # Spatial rectangle to search for EQ
    min_easting,  max_easting   = [utm_result[0] - rd*100, utm_result[0] + rd*100]
    min_northing, max_northing  = [utm_result[1] - rd*100, utm_result[1] + rd*100]
    minlatitude, minlongitude   = utm.to_latlon(min_easting, min_northing, utm_result[2], utm_result[3])
    maxlatitude, maxlongitude   = utm.to_latlon(max_easting, max_northing, utm_result[2], utm_result[3])
    
    # Search for earthquakes in USGS database
    attempt = True
    while attempt: # loop to avoid rate limiting problem from USGS server
        try:
            eq = search(starttime=starttime, endtime=endtime, minlatitude=minlatitude, maxlatitude=maxlatitude, 
                             minlongitude=minlongitude, maxlongitude=maxlongitude, minmagnitude=minmagnitude)
            attempt = False
        except Exception as e:
            print(f"Error: {e}.")
            print('   ')
            print(" Retrying in 1 minute...")
            time.sleep(60)
            
    if len(eq)>0:
        
        print('  ' + str(len(eq)) +' Earthquakes found for station: '+ st)
        
        # empty arrays to store data
        eq_names = []
        eq_lats  = []
        eq_lons  = []
        eq_depths = []
        eq_mags   = []
        eq_dist   = []
        
        for j in range(len(eq)):
            try:

                # Search for seismogram from IRIS
                EQ = eq[j]
                sTime, eTime = [UTCDateTime(EQ.time), UTCDateTime(EQ.time)+lsec]
                client = Client('IRIS')
                wv = client.get_waveforms(network = net, station = st, location = '*', 
                                          starttime = sTime, endtime = eTime, 
                                          channel = chn , attach_response = True)
                
                print('     ' + str(len(wv)) + ' traces found for eq no. ' + str(j+1))
                
                # directory for station
                st_dir      = out_dir + st+ '_' + net + '/'
                try:
                    os.mkdir(st_dir)
                except:
                    pass
                
                # write STATION information on a text file
                with open(st_dir + 'st_info.txt' , 'w') as file:
                    
                    # write header
                    file.write(f"{'no':<15}{'netcode':<15}{'station':<15}{'lat':<15}{'lon':<15}\n")
            
                    # write the data
                    file.write(f"{i:<15}{net:<15}{st:<15}{lt:<15}{ln:<15}")

                # directory for the earthquake within the station directory
                eq_spec_dir = st_dir + '_Eq' + str(j) + '_' + EQ.id + '_spec'
                     
                # Distance from eq to station
                eq_yx = utm.from_latlon(EQ.latitude, EQ.longitude)
                st_yx = utm.from_latlon(lt, ln)
                dist  = np.sqrt((eq_yx[0] - st_yx[0])**2 + (eq_yx[1] - st_yx[1])**2) / 1000
                
                #write the earthquake description 
                eq_names.append(EQ.id)
                eq_lats.append(EQ.latitude)
                eq_lons.append(EQ.longitude)
                eq_depths.append(EQ.depth)
                eq_mags.append(EQ.magnitude)
                eq_dist.append(np.round(dist))

                # write EQ information on a text file
                eq_no = np.arange(0,len(eq_names))
                tr_no = len(wv)
                with open(st_dir + 'eq_info.txt' , 'w') as file:
                    
                    # write header
                    file.write(f"{'no':<15}{'name':<15}{'lat':<15}{'lon':<15}{'depth':<15}{'mag':<15}{'dist':<15}\n")
            
                    # write the data
                    for no, names, lats, lons, depths, mags, dists in zip(eq_no, eq_names, eq_lats, eq_lons, eq_depths, eq_mags, eq_dist):
                        file.write(f"{no:<15}{names:<15}{lats:<15}{lons:<15}{depths:<15}{mags:<15}{dists:<15}\n")
                
                for k in range(len(wv)):
                    
                    tr = wv[k]
                    
                    # remove the sensitivity & apply baseline correction
                    sensitivity = tr.stats.response.instrument_sensitivity.value
                    tr.data = tr.data/sensitivity
                    tr.data = tr.data - np.mean(tr.data[0:10])
                    
                    print('       ' + 'downloading the trace ...')
                    tr.write((eq_spec_dir + '_tr' + str(k+1) + '.mseed'), format = 'MSEED')
                    
            except:
                
                print('     No trace found for eq no. ' + str(j+1))
                
                pass
        
    else:
        print('  No Earthquakes found for station: ' + st)





