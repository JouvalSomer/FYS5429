"""
This file might be redundant.


Create a Excel with days in the year range.

Needs to be added:
- filename making clear what Station in what Catchment

"""

import pandas as pd

# Create a date range from 1980-01-01 to 2017-12-31
date_range = pd.date_range(start='1980-01-01', end='2017-12-31', freq='D')

# Convert the date range to string format ('yyyy-mm-dd')
date_range_str = date_range.strftime('%Y-%m-%d')

# Create a DataFrame with the date range
df = pd.DataFrame({'Date': date_range_str})

# Write the DataFrame to an Excel file
df.to_excel('date_range.xlsx', index=False)