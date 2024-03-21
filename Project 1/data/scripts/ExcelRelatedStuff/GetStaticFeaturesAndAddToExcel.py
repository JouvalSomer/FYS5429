"""
Takes in a list of static features. Then takes in a .csv that contains the information about all 
stations in a specific catchment and averages the static features over the stations. Then adds the 
staticFeature names to the final Output excel and adds the values into each cell.

Needs to be added:
- pass the final Output file as an argument
- return the final Output file with added static features
- also look at the list of static features and think about wether they are important or not and what they mean
maybe also look in the stationsInforamtion again if there is something that should be added.

"""


import pandas as pd
import numpy as np

stationsInformation = "Hydrological/Catchment/stationsOfCatchment.csv"
finalOutput = "pass final output xlsx as argument"


listOfStaticFeatures = [
    "masl", "gradient1085", "gradientBasin", "gradientRiver", "lengthKmBasin", "lengthKmRiver", 
    "percentAgricul", "percentBog", "percentEffLake", "percentForest", 
    "percentGlacier", "percentLake", "percentMountain", "percentUrban", 
    "specificDischarge", "regulationArea", "areaReservoirs", "volumeReservoirs",
    "reservoirAreaIn", "reservoirAreaOut", "reservoirVolumeIn", "reservoirVolumeOut", 
    "remainingArea"
]
data = pd.read_csv(stationsInformation)
meanFeatures = []
for feature in listOfStaticFeatures:
    meanFeatures.append(np.mean(data[feature]))

# Load the Excel file
df = pd.read_excel(finalOutput)

# Define the list of column names you want to add

# Add the new columns to the DataFrame
for col_name in listOfStaticFeatures:
    df[col_name] = ""

for i in range(len(listOfStaticFeatures)):
    # Fill rows 1 to 10000 in column "tx" with the number 5
    df.loc[:13881, listOfStaticFeatures[i]] = meanFeatures[i]

# Save the modified DataFrame back to the Excel file
df.to_excel(finalOutput, index=False)





