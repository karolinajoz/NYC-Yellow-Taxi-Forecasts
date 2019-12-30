import streamlit as st

import json

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

import calendar
import datetime

import numpy as np
import pandas as pd
import pickle

from fbprophet import Prophet
pd.plotting.register_matplotlib_converters()

@st.cache
def load_data():
    demand_df = pd.read_csv('nyc_neighborhoods_yellow_neigh_demand_hourly.csv', parse_dates = ['ds'], index_col = ['ds'])
    fare_df = pd.read_csv('nyc_neighborhoods_yellow_neigh_fare_hourly.csv', parse_dates = ['ds'], index_col = ['ds'])
    trip_df = pd.read_csv('nyc_neighborhoods_yellow_neigh_trip_hourly.csv', parse_dates = ['ds'], index_col = ['ds'])
    tips_df = pd.read_csv('nyc_neighborhoods_yellow_neigh_tips_hourly.csv', parse_dates = ['ds'], index_col = ['ds'])
    manhattan_neigh_demand = pd.read_csv("./models/manhattan_neigh_demand.csv", parse_dates = ['ds'], index_col ='ds')
    manhattan_forecasts = pd.read_csv("./models/manhattan_neigh_demand_forecast_date_12.csv", parse_dates = ['ds'], index_col = 'ds')
    queens_neigh_demand = pd.read_csv("./models/queens_neigh_demand.csv", parse_dates = ['ds'], index_col ='ds')
    queens_forecasts = pd.read_csv("./models/queens_neigh_demand_forecast_date_12.csv", parse_dates = ['ds'], index_col = 'ds')
    brooklyn_neigh_demand = pd.read_csv("./models/brooklyn_neigh_demand.csv", parse_dates = ['ds'], index_col ='ds')
    brooklyn_forecasts = pd.read_csv("./models/brooklyn_neigh_demand_forecast_date_12.csv", parse_dates = ['ds'], index_col = 'ds')
    return demand_df, fare_df, trip_df, tips_df, manhattan_neigh_demand, manhattan_forecasts, queens_neigh_demand, queens_forecasts, brooklyn_neigh_demand, brooklyn_forecasts


@st.cache
def load_demand_files():
    manhattan_demand_exog = pd.read_csv('./models/manhattan_demand_exog.csv', parse_dates = ['ds'])
    queens_demand_exog = pd.read_csv('./models/queens_demand_exog.csv', parse_dates = ['ds'])
    bronx_demand_exog = pd.read_csv('./models/bronx_demand_exog.csv', parse_dates = ['ds'])
    brooklyn_demand_exog = pd.read_csv('./models/brooklyn_demand_exog.csv', parse_dates = ['ds'])
    pkl_path = "./models/manhattan.pkl"
    with open(pkl_path, 'rb') as f:
        manhattan_model = pickle.load(f)
    pkl_path = "./models/brooklyn.pkl"
    with open(pkl_path, 'rb') as f:
        brooklyn_model = pickle.load(f)
    pkl_path = "./models/queens.pkl"
    with open(pkl_path, 'rb') as f:
        queens_model = pickle.load(f)
    pkl_path = "./models/bronx.pkl"
    with open(pkl_path, 'rb') as f:
        bronx_model = pickle.load(f)
    return manhattan_demand_exog, queens_demand_exog, bronx_demand_exog, brooklyn_demand_exog, manhattan_model, brooklyn_model, queens_model, bronx_model

@st.cache
def load_fare_files():
    manhattan_fare_exog = pd.read_csv('./models/manhattan_fare_exog.csv', parse_dates = ['ds'])
    queens_fare_exog = pd.read_csv('./models/queens_fare_exog.csv', parse_dates = ['ds'])
    bronx_fare_exog = pd.read_csv('./models/bronx_fare_exog.csv', parse_dates = ['ds'])
    brooklyn_fare_exog = pd.read_csv('./models/brooklyn_fare_exog.csv', parse_dates = ['ds'])
    pkl_path = "./models/manhattan_fare.pkl"
    with open(pkl_path, 'rb') as f:
        manhattan_model_fare = pickle.load(f)
    pkl_path = "./models/brooklyn_fare.pkl"
    with open(pkl_path, 'rb') as f:
        brooklyn_model_fare = pickle.load(f)
    pkl_path = "./models/queens_fare.pkl"
    with open(pkl_path, 'rb') as f:
        queens_model_fare = pickle.load(f)
    pkl_path = "./models/bronx_fare.pkl"
    with open(pkl_path, 'rb') as f:
        bronx_model_fare = pickle.load(f)
    return manhattan_fare_exog, queens_fare_exog, bronx_fare_exog, brooklyn_fare_exog, manhattan_model_fare, brooklyn_model_fare, queens_model_fare, bronx_model_fare

@st.cache
def load_trip_files():
    manhattan_trip_exog = pd.read_csv('./models/manhattan_trip_exog.csv', parse_dates = ['ds'])
    queens_trip_exog = pd.read_csv('./models/queens_trip_exog.csv', parse_dates = ['ds'])
    bronx_trip_exog = pd.read_csv('./models/bronx_trip_exog.csv', parse_dates = ['ds'])
    brooklyn_trip_exog = pd.read_csv('./models/brooklyn_trip_exog.csv', parse_dates = ['ds'])
    pkl_path = "./models/manhattan_trip.pkl"
    with open(pkl_path, 'rb') as f:
        manhattan_model_trip = pickle.load(f)
    pkl_path = "./models/brooklyn_trip.pkl"
    with open(pkl_path, 'rb') as f:
        brooklyn_model_trip = pickle.load(f)
    pkl_path = "./models/queens_trip.pkl"
    with open(pkl_path, 'rb') as f:
        queens_model_trip = pickle.load(f)
    pkl_path = "./models/bronx_trip.pkl"
    with open(pkl_path, 'rb') as f:
        bronx_model_trip = pickle.load(f)
    return manhattan_trip_exog, queens_trip_exog, bronx_trip_exog, brooklyn_trip_exog, manhattan_model_trip, brooklyn_model_trip, queens_model_trip, bronx_model_trip

@st.cache
def load_tips_files():
    manhattan_tips_exog = pd.read_csv('./models/manhattan_tips_exog.csv', parse_dates = ['ds'])
    queens_tips_exog = pd.read_csv('./models/queens_tips_exog.csv', parse_dates = ['ds'])
    bronx_tips_exog = pd.read_csv('./models/bronx_tips_exog.csv', parse_dates = ['ds'])
    brooklyn_tips_exog = pd.read_csv('./models/brooklyn_tips_exog.csv', parse_dates = ['ds'])
    pkl_path = "./models/manhattan_tips.pkl"
    with open(pkl_path, 'rb') as f:
        manhattan_model_tips = pickle.load(f)
    pkl_path = "./models/brooklyn_tips.pkl"
    with open(pkl_path, 'rb') as f:
        brooklyn_model_tips = pickle.load(f)
    pkl_path = "./models/queens_tips.pkl"
    with open(pkl_path, 'rb') as f:
        queens_model_tips = pickle.load(f)
    pkl_path = "./models/bronx_tips.pkl"
    with open(pkl_path, 'rb') as f:
        bronx_model_tips = pickle.load(f)
    return manhattan_tips_exog, queens_tips_exog, bronx_tips_exog, brooklyn_tips_exog, manhattan_model_tips, brooklyn_model_tips, queens_model_tips, bronx_model_tips

cal = dict(zip(range(7),calendar.day_name))

with st.spinner("Loading data.."):
    demand_df, fare_df, trip_df, tips_df, manhattan_neigh_demand, manhattan_forecasts, queens_neigh_demand, queens_forecasts, brooklyn_neigh_demand, brooklyn_forecasts = load_data()

with st.spinner("Loading demand models..."):
    manhattan_demand_exog, queens_demand_exog, bronx_demand_exog, brooklyn_demand_exog, manhattan_model, brooklyn_model, queens_model, bronx_model = load_demand_files()
with st.spinner("Loading fare models..."):
    manhattan_fare_exog, queens_fare_exog, bronx_fare_exog, brooklyn_fare_exog, manhattan_model_fare, brooklyn_model_fare, queens_model_fare, bronx_model_fare = load_fare_files()
with st.spinner("Loading trip models..."):
    manhattan_trip_exog, queens_trip_exog, bronx_trip_exog, brooklyn_trip_exog, manhattan_model_trip, brooklyn_model_trip, queens_model_trip, bronx_model_trip = load_trip_files()
with st.spinner("Loading tips models..."):
    manhattan_tips_exog, queens_tips_exog, bronx_tips_exog, brooklyn_tips_exog, manhattan_model_tips, brooklyn_model_tips, queens_model_tips, bronx_model_tips = load_tips_files()

with open('NYC Taxi Zones.geojson') as response:
    taxi_zone_3 = json.load(response)

for i in range(len(taxi_zone_3['features'])):
    taxi_zone_3['features'][i]['id'] = taxi_zone_3['features'][i]['properties']['zone']
    
print(taxi_zone_3['features'][10]['id'])

url_lookup = 'taxi+_zone_lookup.csv'

zone_lookup = pd.read_csv(url_lookup, index_col = 'LocationID')

lst = []
for i, col in enumerate(demand_df.columns):
    lst.append(zone_lookup.loc[(int(demand_df.columns[i]))]['Zone'])
    
lst_brooklyn = []
lst_queens = []
lst_manhattan = []
lst_staten = []
lst_bronx = []

for i, col in enumerate(demand_df.columns):
    bor = zone_lookup.loc[(int(demand_df.columns[i]))]['Borough']
    if bor == 'Queens':
        lst_queens.append(col)
    elif bor == 'Bronx':
        lst_bronx.append(col)
    elif bor == 'Manhattan':
        lst_manhattan.append(col)
    elif bor == 'Staten Island':
        lst_staten.append(col)
    elif bor == 'Brooklyn':
        lst_brooklyn.append(col)

def main():
    st.title("NYC Yellow Taxi Forecasts")
    what =  st.sidebar.radio("What do you want to do?", ('Read about the project','Look at the stats', 'Forecasts for boroughs', "Demand forecasts for neighborhoods"))
  
    if what == 'Read about the project':
        about()
    elif what == 'Look at the stats':
        stats()
    elif what == 'Forecasts for boroughs':
        boroughs()
    elif what == "Demand forecasts for neighborhoods":
        neighs()
    
    
def about():
    
    st.markdown(
    """ 
    [Presentation](https://docs.google.com/presentation/d/1VB7F82okO_C_BSdAJD7PlY08mFHoK2Uxnn3M9ZTIvH4/edit#slide=id.ge9090756a_2_12)
    
    ### Project idea
    
    \n Building a dashboard for the taxi company with statistics and forecasts of:
        
    - taxi pickup demand
    
    - average fare
    
    - trip distance
    
    - tips for the drivers (as a ratio of total price for the ride)
    
    ### Dataset
    
    \n Dataset consists of over 100 milion of NYC yellow taxi pickup information from the whole year of 2018. 
    Data about the rides were gathered from the [NYC Open Data](https://opendata.cityofnewyork.us/) website together with the information about the taxi zones in NYC.
    \n Additionally weather information ([NOAA](https://www.ncdc.noaa.gov/cdo-web/search)) and CityBike demand data ([CityBike NYC](https://www.citibikenyc.com/system-data))
    were added to improve the accuracy of the predictions.
    
    
    ### Technologies used
    
    \n All calculations were run using Python and its libraries such as Pandas, NumPy. Plots were created 
    using MatPlotLib and Plotly libraries.
    \n To handle big amount of data points I utilized Dask and PySpark for the parallelization of computing on Google Cloud. 
    Forecasts were performed with the use of Facebook Prophet model. This model is a combination of three different 
    forecasting algorithms resposible for capturing trends in time, seasonality and additional regressors impacting 
    time series, such as holidays and one-time events.
    
    \n For choosing best performing model time-series cross-validation approach was implemented.
    
    
    ### Process outline
    
    \n After gathering all the data, they were preprocessed to get average hourly values. 
    First version of models used grouped date for whole NYC area and implemeting only seasonality to the model.
    Next step was to add exogenous variables to improve accuracy for each prediction. 
    \n After getting model for NYC 
    data was then grouped into boroughs to make prediction on more specific level. Staten Island pickups were not 
    discarded as there was too little taxi pickups in total. This step resulted in 4 different forecasting model per borough - 
    16 models in total. Each of them consisted of a subset of exogenous variables, which had the lowest average RMSE 
    after cross-validation.
    \n Last iteration of building forecasting models was to predict the demand at the level of a neighborhood. 
    For locations with lowest taxi demand closest taxi zones were grouped together. This process resulted in 58 neighborhoods
    for Manhattan, 21 locations at Brooklyn and 24 at Queens.
    Models were fit with the demand information for each of the locations seperately to get the best prediction.
    
    \n At the end all models were fitted to data from January to end of October 2018 - November and December 
    data were used as a final evaluation.
    
    """)

def stats():
    @st.cache
    def show_sum_plot(df, time_range = 'Total', value = 0):
        if time_range == 'Weekly':
            fig = go.Figure(go.Choroplethmapbox(geojson=taxi_zone_3, 
                                            locations=lst, 
                                            z=df.groupby([df.index.weekday]).sum().loc[value].reset_index()[value],
                                            colorscale="Viridis",
                                            marker_opacity=0.5, marker_line_width=0))
        elif time_range =='Hourly':
            fig = go.Figure(go.Choroplethmapbox(geojson=taxi_zone_3, 
                                            locations=lst, 
                                            z=df.groupby([df.index.hour]).sum().loc[value].reset_index()[value],
                                            colorscale="Viridis",
                                            marker_opacity=0.5, marker_line_width=0))
        elif time_range == 'Total':
            fig = go.Figure(go.Choroplethmapbox(geojson=taxi_zone_3, 
                                            locations=lst, 
                                            z=df.sum(),
                                            colorscale="Viridis",
                                            marker_opacity=0.5, marker_line_width=0))
        fig.update_layout(mapbox_style="carto-positron",
                          mapbox_zoom=9, mapbox_center = {"lat": 40.7, "lon": -74})
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig
    
    @st.cache
    def show_avg_plot(df, time_range = 'weekday', value = 0):
        if time_range == 'Weekly':
            fig = go.Figure(go.Choroplethmapbox(geojson=taxi_zone_3, 
                                            locations=lst, 
                                            z=df.groupby([df.index.weekday]).mean().loc[value].reset_index()[value],
                                            colorscale="Viridis",
                                            marker_opacity=0.5, marker_line_width=0))
        elif time_range =='Hourly':
            fig = go.Figure(go.Choroplethmapbox(geojson=taxi_zone_3, 
                                            locations=lst, 
                                            z=df.groupby([df.index.hour]).mean().loc[value].reset_index()[value],
                                            colorscale="Viridis",
                                            marker_opacity=0.5, marker_line_width=0))
        elif time_range == 'Total':
            fig = go.Figure(go.Choroplethmapbox(geojson=taxi_zone_3, 
                                            locations=lst, 
                                            z=df.mean(),
                                            colorscale="Viridis",
                                            marker_opacity=0.5, marker_line_width=0))
        fig.update_layout(mapbox_style="carto-positron",
                          mapbox_zoom=9, mapbox_center = {"lat": 40.7, "lon": -74})
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig

    @st.cache
    def show_stats_plot(data_for_plot):
        fig2 = go.Figure(data=[go.Scatter(x = pd.Series(data_for_plot.index),
                                                     y = data_for_plot[i],
                                                     name = calendar.day_name[i],
                                                     text = i,
                                                     hovertemplate = '<b>Hour</b> %{x} <b>Value:</b>  %{y:.2f}') for i in data_for_plot.columns],
                    layout = go.Layout(title=f'Yellow Taxi {map_val} in NYC',
                                                        xaxis={'title': 'Hour',
                                                               'visible':True,
                                                               'autorange':True,
                                                               'rangemode':'nonnegative',
                                                               'ticks':'inside'},
                                                        yaxis={'title': f'{map_val}',
                                                               'visible':True}))
        return fig2

    map_val = st.selectbox("What do you want to see?", ['Pickup Demand', 'Avg Fare', 'Avg Trip Distance', 'Avg Tip Ratio'])

    if map_val == 'Avg_tips':
        df = tips_df
        
    elif map_val == 'Avg Fare':
        df = fare_df
        
    elif map_val == 'Avg Trip Distance':
        df = trip_df
        
    else:
        df = demand_df
        
    if map_val == 'Pickup Demand':
        data_for_plot = df.copy().sum(axis=1)
        data_for_plot = pd.DataFrame(data_for_plot).set_index(data_for_plot.index).groupby([data_for_plot.index.hour,data_for_plot.index.weekday]).mean()[0].unstack()
        
    else:
        data_for_plot = df.copy().mean(axis=1)
        std_df = data_for_plot.std()
        data_for_plot.index = df.index
        data_for_plot = data_for_plot[data_for_plot < 10*std_df]
        data_for_plot = pd.DataFrame(data_for_plot).set_index(data_for_plot.index).groupby([data_for_plot.index.hour,data_for_plot.index.weekday]).mean()[0].unstack()
        
    time_val = st.selectbox("What time range are you interested in?", ['Total', 'Weekly', 'Hourly'])
    if time_val == 'Total':
        range_val = 0
    elif time_val == 'Weekly':
        range_val = st.slider('Choose day of the week (0-Monday, 6-Sunday)', 0, 6, 4)
        st.write(cal[range_val])
    elif time_val == 'Hourly':
        range_val = st.slider('Choose hour of the day (0-23)', 0, 23, 15)
     
    if map_val == 'Pickup Demand':
        with st.spinner("Loading map..."):
            fig1 = show_sum_plot(df, time_range = time_val, value = range_val)
            st.plotly_chart(fig1)
        
    else:
        with st.spinner("Loading map..."):
            fig1 = show_avg_plot(df, time_range = time_val, value = range_val)  
            st.plotly_chart(fig1)
      
    with st.spinner("Showing stats..."):
        fig2 = show_stats_plot(data_for_plot)
        st.plotly_chart(fig2)
            
    st.write("Total amount of Yellow Taxi pickups in 2018:", demand_df.sum().sum())
    
    fig5 = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]], 
                    subplot_titles=("Borough distribution","Weekly distribution"))
    fig5.add_trace(go.Pie(labels = ['Manhattan', 'Queens', 'Brooklyn', 'Bronx', "Staten Island"],
                         values = [demand_df[lst_manhattan].sum().sum(),demand_df[lst_queens].sum().sum(),
                               demand_df[lst_brooklyn].sum().sum(),demand_df[lst_bronx].sum().sum(),
                               demand_df[lst_staten].sum().sum()],
                         rotation = -45),1,1)
    fig5.add_trace(go.Pie(labels=[calendar.day_name[i] for i in list(range(0,7))], 
                         values=demand_df.groupby([demand_df.index.weekday]).sum().sum(axis = 1), sort = False), 1,2)
    fig5.update_traces(
                         hole = 0.5, 
                         hoverinfo="label+value+name", 
                         name='', textinfo='label+percent', 
                         showlegend=False)
    
    fig5.update_layout(legend=dict(
                        x=0,
                        y=1.0,
                        bgcolor='rgba(255, 255, 255, 0)',
                        bordercolor='rgba(255, 255, 255, 0)'
                  ),
                  xaxis = {'title': "Day of the week", 
                          'titlefont': {'size': 14},
                          'tickfont': {'size': 14}
                          },
                  yaxis = {'title': "Weekly average", 
                          'titlefont': {'size': 14},
                          'tickfont': {'size': 14}},
                  barmode='group',
                  bargroupgap = 0.1)
    st.plotly_chart(fig5)

def boroughs():
    def fb_plot(m, fcst, xlabel='ds', ylabel='y'):
        fig = go.Figure()
        fcst_t = fcst['ds'].dt.to_pydatetime()
    
        fig.add_trace(go.Scatter(x = fcst_t, y = fcst['yhat_lower'],
        line_color='rgba(255,255,255,0)', name='', hovertemplate ='yhat_lower'))
        fig.add_trace(go.Scatter(x = fcst_t, y = fcst['yhat_upper'], fill ='tonexty',
        line_color='rgba(255,255,255,0)', hovertemplate ='yhat_upper',
        fillcolor='rgba(0,176,246,0.2)', name= ''))
    
        return fig
    
    def make_prediction_for_borough(prediction, boro_data, model,date1='2018-12-01', date2 = '2018-12-08',  show_components = False):
        num_pounts_to_pred = int((pd.to_datetime(date2)-pd.to_datetime(date1))/np.timedelta64(1, 'h'))
            
        sub_df = boro_data[(boro_data.ds < date2) & (boro_data.ds > date1)]
        
        forecast = model.predict(sub_df)
        forecast = forecast[-num_pounts_to_pred:]
        
        fig = fb_plot(model,forecast)
    
        fig.add_trace(go.Scatter(x = forecast.ds,
                                y = forecast.yhat,
                                mode = 'lines',
                                name = 'forecast',line_color='rgb(0,176,246)',
                                hovertemplate = '%{y:.3f}'))
        fig.add_trace(go.Scatter(x = sub_df.ds,
                                y = sub_df.y,
                                mode = 'lines',
                                name = 'actual',line_color='orange',
                                hovertemplate = '%{y:.3f}'))
        fig.update_layout(xaxis_range=[date1,
                                   date2])
        fig.update_layout(title = f'Forecast for {prediction}',
                      legend=dict(
                            x=0,
                            y=1.0,
                            bgcolor='rgba(255, 255, 255, 0)',
                            bordercolor='rgba(255, 255, 255, 0)'
                      ),
                      xaxis = {'title': "Time range", 
                              'titlefont': {'size': 14},
                              'tickfont': {'size': 14}
                              },
                      yaxis = {'title': "Average", 
                              'titlefont': {'size': 14},
                              'tickfont': {'size': 14}})
        return st.plotly_chart(fig)
    
    def boro_stat(df, title):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=['Manhattan', 'Queens', 'Brooklyn', 'Bronx'], 
                         y=[df[lst_manhattan].mean().mean(),df[lst_queens].mean().mean(),
                               df[lst_brooklyn].mean().mean(),df[lst_bronx].mean().mean(),
                               df[lst_staten].mean().mean()],
                         name = '',
                         hovertemplate = '%{x} %{y:.2f}',
                         marker_color='lightsalmon'))
        fig.update_layout(title = title)
        return st.plotly_chart(fig)
    
    
    boro = st.selectbox("Which borough are you interested in?", ['Brooklyn', 'Bronx', 'Manhattan', 'Queens'])
    if boro == 'Brooklyn':
        model_demand = brooklyn_model
        model_fare = brooklyn_model_fare
        data_demand = brooklyn_demand_exog
        data_fare = brooklyn_fare_exog
        model_trip = brooklyn_model_trip
        data_trip = brooklyn_trip_exog
        model_tips = brooklyn_model_tips
        data_tips = brooklyn_tips_exog
    elif boro == 'Bronx':
        model_demand = bronx_model
        model_fare = bronx_model_fare
        data_demand = bronx_demand_exog
        data_fare = bronx_fare_exog
        model_trip = bronx_model_trip
        data_trip = bronx_trip_exog
        model_tips = bronx_model_tips
        data_tips = bronx_tips_exog
    elif boro == 'Queens':
        model_demand = queens_model
        model_fare = queens_model_fare
        data_demand = queens_demand_exog
        data_fare = queens_fare_exog
        model_trip = queens_model_trip
        data_trip = queens_trip_exog
        model_tips = queens_model_tips
        data_tips = queens_tips_exog
    elif boro == 'Manhattan':
        model_demand = manhattan_model
        model_fare = manhattan_model_fare
        data_demand = manhattan_demand_exog
        data_fare = manhattan_fare_exog
        model_trip = manhattan_model_trip
        data_trip = manhattan_trip_exog
        model_tips = manhattan_model_tips
        data_tips = manhattan_tips_exog
        
    d = st.date_input('When do you want to start your prediction?', datetime.date(2018,11,1))
        
    if (d > datetime.date(2018,12,25)) or (d < datetime.date(2018,1,1)):
        st.write('Choose another date')
    else:
        dem = st.radio('What predictions do you want to see?', ['Pickup demand', 'Average Fare', 'Average Trip Distance', 'Average Tips Ratio'])
        st.write("")
        stat = st.empty()
        if dem == 'Pickup demand':    
            with st.spinner("Predicting the future demand..."):
                fig_3 = make_prediction_for_borough(dem, data_demand, model_demand, date1=d.strftime("%Y-%m-%d"), date2 = (d+datetime.timedelta(days=6)).strftime("%Y-%m-%d"))
                fig_7 = boro_stat(demand_df, title = "Average pickup demand per hour")
        if dem == 'Average Fare':
            with st.spinner("Predicting the future fare..."):
                fig_3 = make_prediction_for_borough(dem, data_fare, model_fare, date1=d.strftime("%Y-%m-%d"), date2 = (d+datetime.timedelta(days=6)).strftime("%Y-%m-%d"))
                st.write(
                fig_7 = boro_stat(fare_df, "Average fare per ride"))
        if dem == 'Average Trip Distance':
            with st.spinner("Predicting the future trip distance..."):
                fig_3 = make_prediction_for_borough(dem, data_trip, model_trip, date1=d.strftime("%Y-%m-%d"), date2 = (d+datetime.timedelta(days=6)).strftime("%Y-%m-%d"))
                st.write()
                fig_7 = boro_stat(trip_df, "Average trip distance per ride")
        if dem == 'Average Tips Ratio':
            with st.spinner("Predicting the future tips ratio..."):
                fig_3 = make_prediction_for_borough(dem, data_tips, model_tips, date1=d.strftime("%Y-%m-%d"), date2 = (d+datetime.timedelta(days=6)).strftime("%Y-%m-%d"))
                st.write()
                fig_7 = boro_stat(tips_df, "Average tip ratio per ride")
                
def neighs():

    def show_neigh_predictions(df, forecast, neigh, date_start = '2018-12-12', date_end = '2018-12-16'):
        fig = go.Figure()
        s = forecast[neigh].std()
        name = str(zone_lookup.loc[int(neigh)]['Zone'])
        
        fig.add_trace(go.Scatter(x = df[date_start:date_end].index, 
                                 y = forecast[date_start:date_end][neigh] + s, 
                                 line_color='rgba(255,255,255,0)', 
                                 hovertemplate ='yhat_upper',
                                 name = ''))
        fig.add_trace(go.Scatter(x = df[date_start:date_end].index, 
                                 y = forecast[date_start:date_end][neigh] - s, 
                                 line_color='rgba(255,255,255,0)', 
                                 hovertemplate ='yhat_lower',
                                 fill = 'tonexty',
                                 fillcolor = 'rgba(0,176,246,0.2)',
                                 name = ''))
        fig.add_trace(go.Scatter(x = df[neigh][date_start:date_end].index, 
                                 y = df[neigh][date_start:date_end], 
                                 line_color='orange', 
                                 name = 'actual',
                                 hovertemplate = '%{y:.3f}'))
        fig.add_trace(go.Scatter(x = df[date_start:date_end].index, 
                                 y = forecast[date_start:date_end][neigh], 
                                 line_color='rgb(0,176,246)', 
                                 hovertemplate = '%{y:.3f}',
                                 name = 'forecast'))
        fig.update_layout(title = f'Demand forecast for {name}',
                          legend=dict(
                            x=0,
                            y=1.0,
                            bgcolor='rgba(255, 255, 255, 0)',
                            bordercolor='rgba(255, 255, 255, 0)'),
                          xaxis_range=[date_start,
                                   date_end],
                          xaxis = {'tickfont':{'size':14}}, 
                          yaxis = {'tickfont':{'size':14}},
                          xaxis_tickformat = '%d %b %Y')
        return st.plotly_chart(fig)
    
    def show_neigh_stats(df, neigh, boro):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[calendar.day_name[i] for i in list(range(0,7))], 
                             y=df.groupby([df.index.weekday])[neigh].mean(),
                             name = str(zone_lookup.loc[int(neigh)]['Zone']),
                             hovertemplate = '%{y:.2f}'))
        fig.add_trace(go.Bar(x=[calendar.day_name[i] for i in list(range(0,7))], 
                             y=df.groupby([df.index.weekday]).mean().mean(axis = 1),
                             name = boro,
                             hovertemplate = '%{y:.2f}'))
        fig.update_layout(title = 'Comparison of the demand in neighborhood to whole borough',
                      legend=dict(
                            x=0,
                            y=1.0,
                            bgcolor='rgba(255, 255, 255, 0)',
                            bordercolor='rgba(255, 255, 255, 0)'
                      ),
                      xaxis = {'title': "Day of the week", 
                              'titlefont': {'size': 14},
                              'tickfont': {'size': 14}
                              },
                      yaxis = {'title': "Weekday average", 
                              'titlefont': {'size': 14},
                              'tickfont': {'size': 14}},
                      barmode='group',
                      bargroupgap = 0.1)
        return fig

    boro = st.selectbox("Which borough are you interested in?", ['Manhattan', 'Queens','Brooklyn'])
    if boro == 'Manhattan':
        df = manhattan_neigh_demand
        forecast = manhattan_forecasts       
    elif boro == 'Brooklyn':
        df = brooklyn_neigh_demand
        forecast = brooklyn_forecasts
    else:
        df = queens_neigh_demand
        forecast = queens_forecasts
        
    lst_zones = [str(zone_lookup.loc[int(x)]['Zone']) for x in list(df.columns)]
    selection = st.selectbox("Which neighborhood are you interested in?", lst_zones)
    neigh = str(zone_lookup[zone_lookup.Zone == selection].index[0])
        
    d = st.date_input('When do you want to start your prediction?', datetime.date(2018,11,1))
    
    if (d > datetime.date(2018,12,25)) or (d < datetime.date(2018,11,1)):
        st.write('Choose another date')
    else:    
        with st.spinner("Predicting the future..."):
            show_neigh_predictions(df, forecast, neigh, date_start = d, date_end = (d+datetime.timedelta(days=6)).strftime("%Y-%m-%d"))
    
    st.plotly_chart(show_neigh_stats(df, neigh, boro))

if __name__ == "__main__":
    main()