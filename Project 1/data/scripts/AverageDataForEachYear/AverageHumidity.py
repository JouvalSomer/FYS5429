"""
Averages all humidity .nc files in the folder humidity meaning it calculates the yearly average.
Puts the average for each year in an excel.
The coordinates used are the coordinates of NVE hydrological stations in a specific catchment.
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

# Define the folder containing humidity data
folder = "Data/humidity/"

# Define lists to store mean values for each location and each year
mean_huss_year = []

# Iterate through years from 1970 to 2017
for year in range(1970, 2018):
    # Construct filename based on the year
    filename = f"humidity_{year}.nc"
    file_path = os.path.join(folder, filename)
    
    # Check if the file exists
    if os.path.exists(file_path):
        print("Processing file:", filename)
        
        # Open dataset
        ds = xr.open_dataset(file_path)
        
        # Initialize list to store mean huss for each coordinate
        mean_huss_loc = []
        
        # Iterate through each coordinate
        for coord in coordinates:
            # Convert longitude and latitude to UTM coordinates
            lon, lat = coord
            utm_x, utm_y = proj_UTM(lon, lat)

            # Extract a dataset closest to the specified point
            dsloc = ds.sel(X=utm_x, Y=utm_y, method='nearest').to_dataframe()
            
            # Drop unnecessary variables
            dsloc = dsloc[['huss']]

            # Calculate mean huss
            mean_huss = np.nanmean(dsloc["huss"])
            mean_huss_loc.append(mean_huss)
        
        # Calculate mean huss for the year
        mean_huss_year.append((str(year), np.nanmean(mean_huss_loc)))
        print(mean_huss_year)

# Create DataFrame from the list of tuples
dfinal = pd.DataFrame(mean_huss_year, columns=["Year", "Mean huss"])

# Write DataFrame to Excel file
output_file = "YearlyAveragehuss_1970_2017.xlsx"
dfinal.to_excel(output_file, index=False)

print(f"Excel file '{output_file}' has been created successfully.")