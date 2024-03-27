import xarray as xr
import geopandas as gpd
import regionmask
import pandas as pd
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_year_variable(year, variable_group, symbols, base_path, shapefile_path):
    """
    Processes datasets for a single year and variable based on a shapefile, calculates spatial means,
    and returns a list of tuples containing variable names and their corresponding DataFrames.
    """
    dataframes_list = []
    try:
        for variable, symbol in zip(variable_group, symbols):
            file_path = f'{base_path}{year}.nc'
            ds = xr.open_dataset(file_path)
            gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)

            min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
            lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
            lon_name = 'longitude' if 'longitude' in ds.coords else 'lon'
            mask = ((ds[lat_name] >= min_lat) & (ds[lat_name] <= max_lat) &
                    (ds[lon_name] >= min_lon) & (ds[lon_name] <= max_lon))
            ds = ds.where(mask, drop=True)

            polygons = [geometry for geometry in gdf.geometry]
            poly = regionmask.Regions(polygons)
            lat = ds[lat_name]
            lon = ds[lon_name]
            mask = poly.mask(lon, lat)
            ds_mask = ds.where(mask == 0)

            coord_x_name = 'X' if 'X' in ds_mask.dims else 'x'
            coord_y_name = 'Y' if 'Y' in ds_mask.dims else 'y'
            spatial_mean_var = ds_mask[symbol].mean(
                dim=[coord_y_name, coord_x_name])
            time_mean_var = spatial_mean_var.mean(dim='time')

            daily_means_df = spatial_mean_var.to_dataframe(
                name=f'daily_mean_{symbol}')
            daily_means_df[f'time_mean_{symbol}'] = time_mean_var.item()
            daily_means_df.reset_index(inplace=True)
            daily_means_df['time'] = daily_means_df['time'].dt.normalize()

            dataframes_list.append((variable, daily_means_df))
    except Exception as e:
        print(f"Error processing {variable} for {year}: {e}")
    return dataframes_list


def concatenate_dataframes(dataframes_dict, output_filename):
    """
    Concatenates the provided DataFrames vertically for each variable, horizontally across variables,
    and saves the combined DataFrame to a CSV file.
    """
    concatenated_by_variable = {}

    for variable, dfs in dataframes_dict.items():
        concatenated_by_variable[variable] = pd.concat(dfs, ignore_index=True)

    combined_df = pd.DataFrame()
    for variable, df in concatenated_by_variable.items():
        df = df.rename(columns={f'daily_mean_{variable}': f'{variable}_daily',
                       f'time_mean_{variable}': f'{variable}_yearly'})
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.merge(df, on='time', how='outer')

    combined_df.to_csv(output_filename, index=False)
    print(f"Combined data saved to '{output_filename}'.")


if __name__ == '__main__':
    start_time = time.time()

    # Define the variables information and other parameters
    variables_info = {
        ('pressure',): (('sp',), 'Data/HySN/HySN_Surface_Pressure_'),
        ('precip', 'maxTemp', 'minTemp'): (('rr', 'tx', 'tn',), 'Data/Climate/seNorge2018_'),
        ('humidity',): (('huss',), 'Data/HySN/HySN_Near_Surface_Specific_Humidity_'),
        ('Radiation',): (('rsds',), 'Data/HySN/HySN_Surface_Downwelling_Shortwave_Radiation_'),
        ('SWE',): (('snow_water_equivalent',), 'Data/SWE/swe_')
    }

    years = (2001, 2015)
    shapefile_path = 'Catchment_shapefiles/Nesbyen_Catchment_Boundary_33N.shp'

    # Prepare tasks for parallel execution
    tasks = [(year, variable_group, symbols, base_path, shapefile_path)
             for year in range(years[0], years[1] + 1)
             for variable_group, (symbols, base_path) in variables_info.items()]

    dataframes_dict = {}
    with ProcessPoolExecutor() as executor:
        # Submit tasks for parallel processing
        future_to_data = {executor.submit(
            process_year_variable, *task): task for task in tasks}

        # Process completed tasks as they become available
        for future in as_completed(future_to_data):
            results = future.result()
            for variable, df in results:
                if variable not in dataframes_dict:
                    dataframes_dict[variable] = []
                dataframes_dict[variable].append(df)

    # Concatenate and save the resulting DataFrames
    output_csv_filename = 'final_concatenated_data1234.csv'
    concatenate_dataframes(dataframes_dict, output_csv_filename)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
