#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 18:36:45 2023

@author: asifashraf
"""

##IMPORT
from datetime import datetime
import numpy as np
from libcomcat.search import search
import pandas as pd
from obspy.core import UTCDateTime
from obspy.clients.fdsn.client import Client
import utm
import time
import os

## INPUT
# Directory of station file
theFile = '/Users/asifashraf/Documents/Fq_Decomp/example/gmap-stations (4).txt'
# Radius around the station
rd = 1000 # in km
# length of second to cut the seismogram in time
lsec = 200
# Minimum magnitude to search EQ for
minmagnitude  = 4.5
# start and end time for searching
starttime     = datetime(2000, 1, 1, 0, 0) #YY/MM/DD
endtime       = datetime(2023, 12, 31, 0, 0)
# Channel
chn           = 'HNE'
# directory to store the result
out_dir = '/Users/asifashraf/Documents/Fq_Decomp/example/2000_2023/';

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
        eq_no    = []
        eq_names = []
        eq_times = []
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
                eq_no.append(j)
                eq_names.append(EQ.id)
                eq_times.append(str(UTCDateTime(EQ.time)))
                eq_lats.append(EQ.latitude)
                eq_lons.append(EQ.longitude)
                eq_depths.append(EQ.depth)
                eq_mags.append(EQ.magnitude)
                eq_dist.append(np.round(dist))

                # write EQ information on a text file
                tr_no = len(wv)
                with open(st_dir + 'eq_info.txt' , 'w') as file:
                    
                    # write header
                    file.write(f"{'no':<30}{'name':<30}{'time':<30}{'lat':<30}{'lon':<30}{'depth':<30}{'mag':<30}{'dist':<30}\n")
            
                    # write the data
                    for no, names, times, lats, lons, depths, mags, dists in zip(eq_no, eq_names, eq_times, eq_lats, eq_lons, eq_depths, eq_mags, eq_dist):
                        file.write(f"{no:<30}{names:<30}{times:<30}{lats:<30}{lons:<30}{depths:<30}{mags:<30}{dists:<30}\n")
                
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





