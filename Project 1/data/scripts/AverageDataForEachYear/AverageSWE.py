"""
Averages all SWE .nc files in the folder SWE meaning it calculates the yearly average.
Puts the average for each year in an excel.
The coordinates used are the coordinates of NVE hydrological stations in a specific catchment.

Warning: For SWE the dataset has a smal "x" and "y" as column names, as opposed to capital ones for the
other metrological data.

"""


import xarray as xr
import pandas as pd
import numpy as np
import pyproj
import os

# Define the UTM projection for zone 33
utm_zone = 33
proj_UTM = pyproj.Proj(proj='utm', zone=utm_zone, ellps='WGS84')

# Read coordinates from coordinates.txt file
coordinates = []
with open('coordinates.txt', 'r') as file:
    for line in file:
        lon, lat = map(float, line.strip().split(','))
        coordinates.append((lon, lat))

# Define the folder containing SWE data
folder = "Data/SWE/"

# Define lists to store mean values for each location and each year
mean_swe_year = []

# Iterate through years from 1970 to 2017
for year in range(1970, 2018):
    # Construct filename based on the year
    filename = f"SWE_{year}.nc"
    file_path = os.path.join(folder, filename)
    
    # Check if the file exists
    if os.path.exists(file_path):
        print("Processing file:", filename)
        
        # Open dataset
        ds = xr.open_dataset(file_path)
        
        # Initialize list to store mean swe for each coordinate
        mean_swe_loc = []
        
        # Iterate through each coordinate
        for coord in coordinates:
            # Convert longitude and latitude to UTM coordinates
            lon, lat = coord
            utm_x, utm_y = proj_UTM(lon, lat)

            # Extract a dataset closest to the specified point
            dsloc = ds.sel(x=utm_x, y=utm_y, method='nearest').to_dataframe()
            
            # Drop unnecessary variables
            dsloc = dsloc[['swe']]

            # Calculate mean swe
            mean_swe = np.nanmean(dsloc["swe"])
            mean_swe_loc.append(mean_swe)
        
        # Calculate mean swe for the year
        mean_swe_year.append((str(year), np.nanmean(mean_swe_loc)))
        print(mean_swe_year)

# Create DataFrame from the list of tuples
dfinal = pd.DataFrame(mean_swe_year, columns=["Year", "Mean swe"])

# Write DataFrame to Excel file
output_file = "YearlyAverageswe_1970_2017.xlsx"
dfinal.to_excel(output_file, index=False)

print(f"Excel file '{output_file}' has been created successfully.")