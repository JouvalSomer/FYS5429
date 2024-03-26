import xarray as xr
import geopandas as gpd
import regionmask
import pandas as pd
import os
import time

def mask_data(variables_info, shapefile_path, years, output_dir='MeanData'):
    """
    Processes and masks datasets based on a shapefile and calculates spatial means,
    then stores the DataFrames in a dictionary.
    
    Parameters:
    - variables_info: Dict of variable groups, symbols, and base paths.
    - shapefile_path: Path to the shapefile used for masking.
    - years: Tuple of the start and end years to process.
    
    Returns:
    - A dictionary where keys are variable names and values are lists of DataFrames for each year.
    """
    dataframes_dict = {}

    for year in range(years[0], years[1] + 1):
        for variable_group, (symbols, base_path) in variables_info.items():
            for variable, symbol in zip(variable_group, symbols):
                try:
                    if variable not in dataframes_dict:
                        dataframes_dict[variable] = []

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
                    lat = ds['latitude' if 'latitude' in ds.coords else 'lat']
                    lon = ds['longitude' if 'longitude' in ds.coords else 'lon']
                    mask = poly.mask(lon, lat)
                    ds_mask = ds.where(mask == 0)

                    coord_x_name = 'X' if 'X' in ds_mask.dims else 'x'
                    coord_y_name = 'Y' if 'Y' in ds_mask.dims else 'y'
                    spatial_mean_var = ds_mask[symbol].mean(dim=[coord_y_name, coord_x_name])
                    time_mean_var = spatial_mean_var.mean(dim='time')

                    daily_means_df = spatial_mean_var.to_dataframe(name=f'daily_mean_{symbol}')
                    daily_means_df[f'time_mean_{symbol}'] = time_mean_var.item()
                    daily_means_df.reset_index(inplace=True)
                    # Normalize the time component to 00:00:00
                    daily_means_df['time'] = daily_means_df['time'].dt.normalize()

                    dataframes_dict[variable].append(daily_means_df)

                except Exception as e:
                    print(f"Error processing {variable} for {year}: {e}")
    
    return dataframes_dict

def concatenate_dataframes(dataframes_dict, output_filename):
    """
    Concatenates the provided DataFrames vertically for each variable, horizontally across variables,
    and saves the combined DataFrame to a CSV file.
    
    Parameters:
    - dataframes_dict: Dictionary where keys are variable names and values are lists of DataFrames.
    - output_filename: Name of the CSV file to save the combined DataFrame.
    """
    concatenated_by_variable = {}

    # Vertically concatenate DataFrames for each variable across years
    for variable, dfs in dataframes_dict.items():
        concatenated_by_variable[variable] = pd.concat(dfs, ignore_index=True)

    # Merge all vertically concatenated DataFrames horizontally to create a single DataFrame
    combined_df = pd.DataFrame()
    for variable, df in concatenated_by_variable.items():
        df = df.rename(columns={f'daily_mean_{variable}': f'{variable}_daily', f'time_mean_{variable}': f'{variable}_yearly'})
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = combined_df.merge(df, on='time', how='outer')

    # # Reorder columns to match the desired structure: first 'time', then '*_daily', then '*_yearly'
    # time_col = [col for col in combined_df.columns if 'time' in col]
    # daily_cols = [col for col in combined_df.columns if '_daily' in col]
    # yearly_cols = [col for col in combined_df.columns if '_yearly' in col]
    # combined_df = combined_df[time_col + daily_cols + yearly_cols]

    # Save the combined DataFrame to a CSV file
    combined_df.to_csv(output_filename, index=False)
    print(f"Combined data saved to '{output_filename}'.")



if __name__ == '__main__':
    start = time.time()

    variables_info = {
        ('pressure',): (('sp',), 'Data/Surface_Pressure/HySN_Surface_Pressure_'),
        ('precip', 'maxTemp', 'minTemp'): (('rr', 'tx', 'tn',), 'Data/Climate/seNorge2018_'), 
        ('humidity',): (('huss',), 'Data/Humidity/HySN_Near_Surface_Specific_Humidity_'), 
        ('Radiation',): (('rsds',), 'Data/Shortwave_Radiation/HySN_Surface_Downwelling_Shortwave_Radiation_'), 
        ('SWE',): (('snow_water_equivalent',), 'Data/Snow_Water_Equivalent/swe_')}
    
    years = (2014, 2015)
    shapefile_path = 'Catchment_shapefiles/Nesbyen_Catchment_Boundary_33N.shp'

    dataframes_dict = mask_data(variables_info, shapefile_path, years)


    output_csv_filename = 'final_concatenated_data12345.csv'
    concatenate_dataframes(dataframes_dict, output_csv_filename)

    end = time.time()
    print(f"Execution time: {end - start} seconds")
