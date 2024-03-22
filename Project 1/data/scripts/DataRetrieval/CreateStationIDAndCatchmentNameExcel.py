"""
Create an Excel from NVS_stations.csv that only contains the ID and the hierachy to have a fast check on which
catchments have what ID
"""
import pandas as pd


data = pd.read_csv("NVE_stations.csv")
meanFeatures = []
data = data[["stationId","hierarchy"]]



# Define the list of column names you want to add


# Save the modified DataFrame back to the Excel file
data.to_csv("StationsIdAndHierarchy.csv", index=False)


