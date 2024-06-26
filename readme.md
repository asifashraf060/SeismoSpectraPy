# Python-based tools for seismogram spectral decomposition (SeismoSpectraPy)

Last update: 2024-05-6
Release: 0.1

Description: SeismoSpectraPy is a Python-based software that automates the process of downloading seismogram data from specified stations within a given distance range. Following data acquisition, it offers the capability to perform various frequency decomposition analyses on the gathered seismograms.

## Usage

### To download seismogram waveforms
1. Download the station information as a text file from IRIS website https://ds.iris.edu/gmap/
2. Run 'download_waveforms.py' 
	- Input parameters:
		*theFile* (directory of the station text file from IRIS)
		*rd* (search radius in km around the stations) 
		*lsec* (length of seconds to cut the seismograms) 
		*minmagnitude* (minimum magnitude to search the earthquake for) 
		*starttime* (start time to search for earthquakes)
		*endtime* (end time to search for earthquakes)
		*chn* (channel for which to download the data) 
		*out_dir* (output directory to save the downloaded files)
	- Output files:
		*.mseed files* (downloaded obspy traces in mseed format)
		*eq_info.txt* (text file of earthquake information for the downloaded mseed files)
		*st_info.txt* (text file for station information)
### Perform frequency decompositions
1. Execute the script 'decomp_analysis.py'. This script calculates multispectral Power Spectral Density (PSD) estimates and performs Hilbert-Huang Transform (HHT) decomposition on each downloaded earthquake data set. The script requires the user to follow a series of steps to complete the frequency decomposition for each earthquake. For detailed instructions, refer to the 'analysis_steps.pdf' document located in the docs folder.
	- Input parameters:
		*root_dir* (*our_dir* from 'download_waveforms.py')
		*min_eq* (minimum number of earthquakes required to start )
	- Output:
		Figures of frequency decomposition in .png format for each earthquake and a text file containing the details of every decomposition result