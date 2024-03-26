
#This script excutes data retrieval from multiple sources using institution specific API scripts or developed scripts from the Scripts folder.

"""
This script runs all get-"metrological Data".py through the NVE API. Below some scripts are disabled through
comments. They need to be enabled depending on what has already been downloaded and what not. Through the 
get-observations.py the files are automatically split into Catchments

After All Climate data is downloaded run "splitPrecipMinMaxTemp.py" to split the dataset containing "rr", "tn" and "tx" into
three seperate ones in the folders "Precip", "maxTemp" and "minTemp"


Problem:
Some Stations can't be obtained with the NVE Api, Error 404. I dont know whats happening here.


"""

import os
import pandas as pd
import subprocess

#Retrieves current directory and sets it as working directory
Directory = os.getcwd()
os.chdir(Directory)

###############Streamflow###############
#Downloading daily streamflow data from the Norwegian Water Resources and Energy Directorate (NVE) API
#importing list of selected stations
NVE_stations = pd.read_csv('NVE_stations.csv')
stations = NVE_stations.stationId
print("ReadStations")


#Loop to run NVE API request for each station, data is downloaded into Data/Streamflow folder
for station in stations:
    item1 = station
    #print(item1)
    #This pulls all available daily stream flow to modify the time step or period read at the following link under Observations - GET-method:
    #https://hydapi.nve.no/UserDocumentation/
    script = "get-observations_csv.py -a \"CxS0G8bnxEGIOXolz/PBow==\" -s \"%s\" -p 1001 -r 1440 -t \"/P1D\""   % item1
    api_key = "CxS0G8bnxEGIOXolz/PBow=="
    script_command = ["python", "get-observations_csv.py", "-a", "CxS0G8bnxEGIOXolz/PBow==", "-s", item1, "-p", "1001", "-r", "1440", "-t", "/P1D"]
    #subprocess.run(script_command)
    #print(script_command)
    #os.system(script)
    #print(script)
print("NVE Stations Done")
################Precipitation & Temperature###############
#Downloads Nordic Gridded Climate Dataset netcdf files for daily precipitation and temperature for Norway and Sweden
script = "get-observations_PrecipTemp_old.py"
#os.system(script)
subprocess.run(["python", script])
print("Precip and Temperature Done")
###############Snow Water Equivalent###############
#Downloads seNorge Snow netcdf files for snow water equivalent for Norway
script = "get-observations_SWE.py"
#os.system(script)
subprocess.run(["python", script])
print("SWE DOne")
###############Humidity and Radiation###############
#Downloads MET observations for longwave and short wave radiation, humidity, and windspeed for Norway
script = "get-HySN_RadHum.py"
#os.system(script)
subprocess.run(["python", script])
print("Radiation and Humidity done")
# Download MET observations for pressure
script = "get-pressure.py"
subprocess.run(["python", script])
