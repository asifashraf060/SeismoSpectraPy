#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:27:23 2024

@author: asifashraf
"""


##IMPORT
from pathlib import Path
#from fq_estimation_tools import spectrogram
import matplotlib.pyplot as plt
import numpy as np
# import libcomcat
# Local imports
# from libcomcat.dataframes import get_detail_data_frame
import pandas as pd
#misc..
import os
import glob
from pykrige.ok import OrdinaryKriging

#INPUT
root_dir  = '/Users/asifashraf/Documents/Fq_Decomp/example/2000_2023'

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
    
    try:
    
        # Station info
        station_text = glob.glob(str(folder_path) + '/st_info.txt')
        df = pd.read_csv(station_text[0], delim_whitespace=True)
        st_code, net_code, st_lat, st_lon = df['station'][0], df['netcode'][0], float(df['lat']), float(df['lon'])
        
        # Earthquake info
        eq_text = glob.glob(str(folder_path) + '/eq_info.txt')
        df = pd.read_csv(eq_text[0], delim_whitespace=True)
    
        # Frequency decomposition result
        df_fq = pd.read_csv(str(folder_path) + '/Analysis/EQ_fq_decomp_result.txt', delim_whitespace=True) # import as a dataframe
        
        eq_no, eq_name, eq_lat, eq_lon = df_fq['no'], df_fq['name'], df['lat'], df['lon']
        eq_mag, eq_dist, eq_fq, eq_err, eq_mxfq = df_fq['mag'], df_fq['dist'], df_fq['fq'], df_fq['err'], df_fq['mxFq']
        
        # Spatial interpolation
                # kriging
        dist_L    = np.linspace(min(eq_dist), max(eq_dist), 500)
        mag_L     = np.linspace(min(eq_mag), max(eq_mag), 100)
        [grid_dist, grid_mag] = np.meshgrid(dist_L, mag_L)
        
        OK_dmFQ   = OrdinaryKriging(eq_dist, eq_mag, eq_fq, variogram_model='gaussian', verbose=False, enable_plotting=False)
        OK_mxFQ   = OrdinaryKriging(eq_dist, eq_mag, eq_mxfq, variogram_model='gaussian', verbose=False, enable_plotting=False)
        dmfq_int, ss  = OK_dmFQ.execute('grid', dist_L, mag_L)
        mxfq_int, ss  = OK_mxFQ.execute('grid', dist_L, mag_L)
        
        
        print('     Plotting result ...')
        
        ## SCATTER PLOT
        
        # Plot DOMINANT FQ as scatters
        plt.close('all')
        plt.figure()
        plt.scatter(eq_dist, eq_mag, np.flip(eq_err*20+10), eq_fq, cmap = 'cool')
        plt.colorbar()
        plt.ylim([eq_mag.min()-.5, eq_mag.max() + .5])
        plt.title('Dominant Fq distribution for station ' + st_code)
        plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_fqDecomp_distVSmag.png',format = 'png' )
        
        ann = [] # with annotation
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
        plt.title('Dominant Fq distribution for station ' + st_code)
        plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_fqDecomp_distVSmag_withAnnotation.png', dpi =  500, format = 'png')
        
        # Plot MAXIMUM FQ as scatters
        plt.close('all')
        plt.figure()
        plt.scatter(eq_dist, eq_mag, np.flip(eq_err*20+10), eq_mxfq, cmap = 'cool')
        plt.colorbar()
        plt.ylim([eq_mag.min()-.5, eq_mag.max() + .5])
        plt.title('Maximum Fq distribution for station ' + st_code)
        plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_mxfqDecomp_distVSmag.png',format = 'png' )
        
        ann = [] # with annotation
        for k in (eq_name):
            ann.append(str(k))
        n = []
        for j in eq_fq:
            n.append(str(j))
        
        plt.close('all')
        plt.figure()
        plt.scatter(eq_dist, eq_mag, np.flip(eq_err*20+10), eq_mxfq, cmap = 'cool')
        plt.colorbar()
        plt.ylim([eq_mag.min()-.5, eq_mag.max() + .5])
        for jk in range(len(eq_dist)):
            plt.annotate(n[jk], (eq_dist[jk]+.3, eq_mag[jk]), va='top', ha='left')
        plt.title('Maximum Fq distribution for station ' + st_code)
        plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_mxfqDecomp_distVSmag_withAnnotation.png', dpi =  500, format = 'png')
        
        
        ## GRAPH PLOT
        
        plt.close('all')
        plt.figure()
        plt.subplot(2,1,1)
        plt.errorbar(eq_dist, eq_fq, yerr = eq_err, fmt = 'o', capsize = 5, ecolor = 'red', label = 'Dominant frequency with errorbars')
        plt.grid()
        plt.ylabel('Frequency (Hz)')
        plt.title('Fq distribution for station ' + st_code)
        plt.legend()
        plt.subplot(2,1,2)
        plt.errorbar(eq_dist, eq_mxfq, yerr = eq_err, fmt = 'o', capsize = 5, ecolor = 'blue', label = 'Maximum frequency with errorbars')
        plt.grid()
        plt.legend()
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Distance (km)')
        plt.tight_layout()
        plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_errorplot_distVSfq.png', dpi =  500, format = 'png')

        plt.close('all')
        plt.figure()
        plt.subplot(2,1,1)
        plt.errorbar(eq_mag, eq_fq, yerr = eq_err, fmt = 'o', capsize = 5, ecolor = 'red', label = 'Dominant frequency with errorbars')
        plt.grid()
        plt.ylabel('Frequency (Hz)')
        plt.title('Fq distribution for station ' + st_code)
        plt.legend()
        plt.subplot(2,1,2)
        plt.errorbar(eq_mag, eq_mxfq, yerr = eq_err, fmt = 'o', capsize = 5, ecolor = 'blue', label = 'Maximum frequency with errorbars')
        plt.grid()
        plt.legend()
        plt.ylabel('Frequency (Hz)')
        plt.xlabel('Magnitude')
        plt.tight_layout()
        plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_errorplot_magVSfq.png', dpi =  500, format = 'png')
        
        
        ## SPATIAL INTERPOLATION PLOT
        plt.close('all')
        plt.figure(figsize=(6, 4))
        contour = plt.contourf(grid_dist, grid_mag, dmfq_int, cmap='viridis', levels=15)
        plt.xlabel('Distance (km)')
        plt.ylabel('Magnitude')
        cbar = plt.colorbar(contour)
        cbar.set_label('Frequency interpolation (Hz)')
        plt.title('Ordinary Kriging of DM FQ (Station: ' + st_code+')')
        plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_interpolation_dmFQ.png', dpi =  500, format = 'png')
        
        plt.close('all')
        plt.figure(figsize=(6, 4))
        contour = plt.contourf(grid_dist, grid_mag, mxfq_int, cmap='viridis', levels=15)
        plt.xlabel('Distance (km)')
        plt.ylabel('Magnitude')
        cbar = plt.colorbar(contour)
        cbar.set_label('Frequency interpolation (Hz)')
        plt.title('Ordinary Kriging of MX FQ (Station: ' + st_code+')')
        plt.savefig(str(folder_path) + '/Analysis/' + st_code +'_interpolation_mxFQ.png', dpi =  500, format = 'png')
        

    except:
        
        print('     No Analysis Folder Found!')
        
        pass





















