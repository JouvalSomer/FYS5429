"""
Gets the metrological Data for each parameter and puts it into a .xlsx from 1980 to 2017 in a daily interval


Needs to be added:
- coordinates x and y should be given as a argument and should have the value of the station we are looking at
- it should be looped for all parameters "rr, tn, tx, huss, sp, rsds, swe" (also change to the respective 
folder). Exception for swe and change the
coordinate column to small "x" and "y" instead of capital
- add each column to the final Data excel
"""




import os
import xarray as xr
import pandas as pd

# Path to the folder containing the NetCDF files
folder_path = "Data/SWE/"
x, y = (233461, 7000123)
# Initialize an empty list to store DataFrames
dfs = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".nc"):
        print(f"Processing file: {filename}")
        # Import precipitation and temperature dataset
        path = os.path.join(folder_path, filename)
        ds = xr.open_dataset(path)
        
        # Extract a dataset closest to specified point
        dsloc = ds.sel(y=y, x=x, method='nearest')
        PT1 = dsloc.to_dataframe()
        PT1 = PT1.reset_index(0).reset_index(drop=True)
        PT1 = PT1.drop_duplicates()

        # Convert Unix timestamps to "year-month-day" format
        PT1["Date"] = pd.to_datetime(PT1["time"], unit='D').dt.strftime('%Y-%m-%d')

        # Filter data between 1980 and 2017
        PT1 = PT1[(PT1['Date'] >= '1980-01-01') & (PT1['Date'] <= '2017-12-31')]

        # Create a DataFrame with Date in the first column and values in subsequent columns
        result_df = PT1[["Date", "snow_water_equivalent"]]
        
        # Append the DataFrame to the list
        dfs.append(result_df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dfs)

# Save the combined DataFrame to an Excel file
combined_df.to_excel("OrklaDailySWE1980-2017.xlsx", index=False)