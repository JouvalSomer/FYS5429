"""
This script splits the Climate.nc files containing "rr", "tn" and "tx" values into three seperates. This reduces
computation time due to smaller files that need to be opened and closed (I believe)
"""



import os
import xarray as xr

# Input folder containing climate data files
input_folder = "Data/Climate/"

# Output folders for split climate data
output_precipitation_folder = "Data/Precip/"
output_min_temperature_folder = "Data/minTemp/"
output_max_temperature_folder = "Data/maxTemp/"
if not os.path.exists(output_precipitation_folder):
        os.makedirs(output_precipitation_folder)
if not os.path.exists(output_min_temperature_folder):
        os.makedirs(output_min_temperature_folder)
if not os.path.exists(output_max_temperature_folder):
        os.makedirs(output_max_temperature_folder)


# Iterate through all files in the Climate folder
for filename in os.listdir(input_folder):
    if filename.endswith(".nc"):
        print("Processing file:", filename)
        
        # Open the dataset
        input_file = os.path.join(input_folder, filename)
        ds = xr.open_dataset(input_file)
        
        # Extract year from filename
        year = filename.split('_')[-1].split('.')[0]  # Extract year from the filename
        
        # Extract data variables for precipitation, minimum temperature, and maximum temperature
        precipitation = ds['rr']
        min_temperature = ds['tn']
        max_temperature = ds['tx']
        
        # Output file paths
        output_precipitation_file = os.path.join(output_precipitation_folder, f"precip_{year}.nc")
        output_min_temperature_file = os.path.join(output_min_temperature_folder, f"mintemp_{year}.nc")
        output_max_temperature_file = os.path.join(output_max_temperature_folder, f"maxtemp_{year}.nc")
        
        # Save each variable to a separate NetCDF file
        precipitation.to_netcdf(output_precipitation_file)
        min_temperature.to_netcdf(output_min_temperature_file)
        max_temperature.to_netcdf(output_max_temperature_file)
        
        print(f"Files for {year} have been created successfully.")

print("All NetCDF files have been processed and split.")