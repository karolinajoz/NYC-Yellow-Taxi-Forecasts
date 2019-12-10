[Presentation](https://docs.google.com/presentation/d/1VB7F82okO_C_BSdAJD7PlY08mFHoK2Uxnn3M9ZTIvH4/edit#slide=id.ge9090756a_2_12)
    
### Project idea
    
Building a dashboard for the taxi company with statistics and forecasts of:
        
- taxi pickup demand
    
- average fare
    
- trip distance
    
- tips for the drivers (as a ratio of total price for the ride)
    
### Dataset
    
Dataset consists of over 100 milion of NYC yellow taxi pickup information from the whole year of 2018. 
Data about the rides were gathered from the [NYC Open Data](https://opendata.cityofnewyork.us/) website together with the information about the taxi zones in NYC.
Additionally weather information ([NOAA](https://www.ncdc.noaa.gov/cdo-web/search)) and CityBike demand data ([CityBike NYC](https://www.citibikenyc.com/system-data)) were added to improve the accuracy of the predictions.
    
![April-2018]('./images/April2018.png')
    
### Technologies used
    
All calculations were run using Python and its libraries such as Pandas, NumPy. Plots were created using MatPlotLib and Plotly libraries.
To handle big amount of data points I utilized Dask and PySpark for the parallelization of computing on Google Cloud. 
Forecasts were performed with the use of Facebook Prophet model. This model is a combination of three different forecasting algorithms responsible for capturing trends in time, seasonality and additional regressors impacting time series, such as holidays and one-time events.
    
For choosing best performing model time-series cross-validation approach was implemented.
    
    
### Process outline
    
After gathering all the data, they were preprocessed to get average hourly values. 
First version of models used grouped date for whole NYC area and implemeting only seasonality to the model.
Next step was to add exogenous variables to improve accuracy for each prediction. 
After getting model for NYC data was then grouped into boroughs to make prediction on more specific level. Staten Island pickups were not 
discarded as there was too little taxi pickups in total. This step resulted in 4 different forecasting model per borough - 16 models in total. Each of them consisted of a subset of exogenous variables, which had the lowest average RMSE after cross-validation.
Last iteration of building forecasting models was to predict the demand at the level of a neighborhood. For locations with lowest taxi demand closest taxi zones were grouped together. This process resulted in 58 neighborhoods for Manhattan, 21 locations at Brooklyn and 24 at Queens.
Models were fit with the demand information for each of the locations seperately to get the best prediction.
    
At the end all models were fitted to data from January to end of October 2018 - November and December data were used as a final evaluation.
    
