"""
This file is supposed to be the starting point of creating the final excel containing dates, hydrological discharge,
dynamic features and sstatic features.

First step is to check which stations provide hydrological data from 1980 to 2017.
Then for all these stations a Excel should be created containing the dates in the
first column and the hydrological discharge in the second column.
"""

import os
import pandas as pd
import subprocess

# Location where the excel files containing all the data are stored
Directory = os.getcwd()
StoreDirectory = os.path.join(Directory, '../DataRetrieval/Data/ExcelsForEachStation/')
print("Directory:", Directory)

# Create the directory if it doesn't exist
if not os.path.exists(StoreDirectory):
    os.makedirs(StoreDirectory)

CatchmentFolders = os.listdir("../DataRetrieval/Data/Streamflow/NVE_stations/")
#print(CatchmentFolders)
# iterate through all catchments and find the stations that offer data in the time range:
for folder in CatchmentFolders:
    stations = os.listdir(os.path.join("../DataRetrieval/Data/Streamflow/NVE_stations/",folder))
    #print(stations)
    # iterate through each station and check if it offers the correct time range
    for station in stations:
        # Read the data from the CSV file
        df = pd.read_csv(os.path.join("../DataRetrieval/Data/Streamflow/NVE_stations/"+ folder,station))
        # Remove the 'correction' and 'quality' columns
        df = df.drop(columns=['correction', 'quality'])
        # Convert the 'time' column to datetime objects
        df['time'] = pd.to_datetime(df['time'])

        # Localize the datetime column to remove timezone information
        df['time'] = df['time'].dt.tz_localize(None)
        #print(df['time'])

        # Check if the data file contains the desired time range
        start_date = pd.to_datetime('1980-01-01')
        end_date = pd.to_datetime('2018-01-01')
        #print(start_date)

        if df['time'].min() < start_date and df['time'].max() > end_date:
            print(f"processing file {station}")
            # Filter the data for dates starting from '1980-01-01' and ending at '2018-01-01'
            filtered_df = df[(df['time'] >= '1980-01-01') & (df['time'] <= '2018-01-01')]

            # Format the 'Date' column to the desired format ('yyyy-mm-dd')
            filtered_df['time'] = filtered_df['time'].dt.strftime('%Y-%m-%d')
            station = station.replace("NVEObservation", "Station")
            station = station.replace(".cvs","")
            # Write the filtered DataFrame to an Excel file
            filtered_df.to_excel(os.path.join(StoreDirectory,station + ".xlsx"), index=False)
            #print(f"Filtered data saved to {station}.xlsx")
        else:
            print("Data file does not contain the desired time range. No action taken.")
"""
Then we extract the coordinates for each of these stations and start getting the 
metrological data per day
"""


# go trough all these created excels and strip the stationId from the file name
excels = os.listdir(StoreDirectory)
print(excels)
for excel in excels:
    ID = excel.replace("Station", "")
    ID = ID.replace("_",".")
    ID = ID.replace(".xlsx", "")
    
    # find now the coordinate for the station from the NVE_stations.csv
    data = pd.read_csv("../DataRetrieval/NVE_stations.csv")
    for i in range(len(data["stationId"])):
        if data["stationId"][i] == ID:
            coordinates = (data["latitude"][i], data["longitude"][i])
            # Convert the coordinates tuple to a string because for some reason subprocess can't handle tuples?
            coordinates_str = ','.join(map(str, coordinates))
            # now that we got the coordinates we can call scripts to add the daily data (GetDailyMetrologicalDataForOneStation.py)
            fullPathToExcel = os.path.join(StoreDirectory, excel)
            print(fullPathToExcel)
            completed_process = subprocess.run(["python", "GetDailyMetrological.py", coordinates_str, excel])
            print(coordinates_str, excel)
            # Check if the subprocess has finished successfully
            if completed_process.returncode == 0:
                print("Subprocess finished successfully!")
            else:
                print("Subprocess failed with return code:", completed_process.returncode)









"""
----
averaging data Jouval
----

After that we also add the static parameters to the excel

"""





