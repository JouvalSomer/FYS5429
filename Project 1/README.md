# Precect description

Recent developments of Kratzert et al. (2018) suggest that the application of long short-term memory (LSTM) networks outperforms traditional hydrological models in the prediction of daily streamflows. This work was applied to catchments in the CAMELS dataset covering North America. <br>
Bernt Viggo Matheussen et al. (Å Energi) have tested the application of LSTM in combination with traditional hydrological models for catchments in southern Norway and found a significant improvement in the streamflow predictions. However, their work is not published and not publicly available. Thus, a first goal can be to repeat their study on NVE-catchments in Norway.

# Tentativ plan for project 1


| Week no. | Dates           | Description                                           |
|----------|-----------------|-------------------------------------------------------|
| 6        | Feb. 5 - 11     | Reading up on hydrology and hydrological modelling    |
| 7        | Feb. 12-18      | Acquiring data and starting to build models           |
| 8        | Feb. 19-25      | Data handling and continue to build models            |
| 9        | Feb. 26 - Mar 3 | Finishing models and starting to tune hyperparameters |
| 10       | Mar. 4 - 10     | Fine-tune hyperparameters and start writing the report|
| 11       | Mar. 11 - 17    | Writing the report                                    |
| 12       | Mar. 18 - 24    | Room for margin / leeway                              |


# Data & Tools:
- ### Meteorological forcing datasets:
    - ERA5: [ECMWF Reanalysis v5 | ECMWF](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5)
    - MERRA2: [MERRA-2 (nasa.gov)](https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/)

- ### Hydrological toolbox:
    - [Shyft Open Source GitLab](https://gitlab.com/shyft-os)

- ### catchments with shape-files:
    - NVE-catchments


## Reading/video recommendations

### Reading:
- [Å Energi presentation](https://www.hydrologiraadet.no/wp-content/uploads/2023/10/1_Matheussen.pdf)
- [Å Energi abstract](https://www.hydrologiraadet.no/wp-content/uploads/2023/09/P_Matheussen_benchmark.pdf)
- ["The LSTM model": Post-Processing the National Water Model with Long Short-Term Memory Networks for Streamflow Predictions and Model](https://onlinelibrary.wiley.com/doi/abs/10.1111/1752-1688.12964)

<br>
<br>

# References
Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018). Rainfall--runoff modelling using long short-term memory (LSTM) networks. *Hydrology and Earth System Sciences*, 22(11), 6005-6022.