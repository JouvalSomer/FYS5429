#This script excutes data retrieval from multiple sources using institution specific API scripts or developed scripts from the Scripts folder.

"""
This script runs all get-"metrological Data".py through the NVE API. Below some scripts are disabled through
comments. They need to be enabled depending on what has already been downloaded and what not.

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
NVE_stations = pd.read_csv('Data/Streamflow/NVE_stations.csv')
stations = NVE_stations.stationId

#Loop to run NVE API request for each station, data is downloaded into Data/Streamflow folder
for station in stations:
    item1 = station
    #This pulls all available daily stream flow to modify the time step or period read at the following link under Observations - GET-method:
    #https://hydapi.nve.no/UserDocumentation/
    script = "get-observations_csv.py -a \"CxS0G8bnxEGIOXolz/PBow==\" -s \"%s\" -p 1001 -r 1440 -t \"/P1D\""   % item1
    api_key = "CxS0G8bnxEGIOXolz/PBow=="
    script_command = ["python", "get-observations_csv.py", "-a", "CxS0G8bnxEGIOXolz/PBow==", "-s", item1, "-p", "1001", "-r", "1440", "-t", "/P1D"]
    subprocess.run(script_command)
    #os.system(script)
    #print(script)
print("1")
################Precipitation & Temperature###############
#Downloads Nordic Gridded Climate Dataset netcdf files for daily precipitation and temperature for Norway and Sweden
script = "get-observations_PrecipTemp_old.py"
#os.system(script)
#subprocess.run(["python", script])
print("2")
###############Snow Water Equivalent###############
#Downloads seNorge Snow netcdf files for snow water equivalent for Norway
#script = "get-observations_SWE.py"
#os.system(script)
#subprocess.run(["python", script])
print("3")
###############Humidity and Radiation###############
#Downloads MET observations for longwave and short wave radiation, humidity, and windspeed for Norway
#script = "get-HySN_RadHum.py"
#os.system(script)
#subprocess.run(["python", script])
print("4")