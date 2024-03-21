
"""
takes a .xlsx with the shape 
"year" "value"
1980    v1
1981    v2
into
"Date"  "value"
1980-01-01  v1
1980-01-02  v1
...
1981-01-01  v2

Need to be added:
This should be maybe connected to the files in "AverageDataForEachYear"


"""



import pandas as pd

# Read the date range Excel file
date_range_df = pd.read_excel('date_range.xlsx')

# Convert the 'Date' column to datetime objects
date_range_df['Date'] = pd.to_datetime(date_range_df['Date'])

# Read the yearly data Excel file
yearly_data_df = pd.read_excel('OrklaMean_tx.xlsx')

# Create an empty list to store the yearly data
yearly_data = []

# Iterate through each date in the date range DataFrame
for date in date_range_df['Date']:
    # Extract the year from the date
    year = date.year
    # Find the corresponding row in the yearly data DataFrame
    yearly_row = yearly_data_df.loc[yearly_data_df['Year'] == year]
    # Get the yearly data value for the current year
    if not yearly_row.empty:
        yearly_value = yearly_row.iloc[0]['Mean tx']
    else:
        yearly_value = None
    # Append the yearly data value to the list
    yearly_data.append(yearly_value)

# Add the yearly data to the date range DataFrame as a new column
date_range_df['YearlyData'] = yearly_data

# Convert the 'Date' column back to 'year-month-day' format
date_range_df['Date'] = date_range_df['Date'].dt.strftime('%Y-%m-%d')

# Write the updated DataFrame to a new Excel file
date_range_df.to_excel('OrklaMeanValuesPerDaytx.xlsx', index=False)