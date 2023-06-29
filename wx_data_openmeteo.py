import json
import requests
import pandas as pd


def get_openmeteo_geocoding_data(city):
    ''' Request geocoded data for a particular name (city name) and return a json object
    
        Example: 
            https://geocoding-api.open-meteo.com/v1/search?name=Berlin&count=1&language=en&format=json
    
        Parameters:
                city (string):  Name of the city that you want to request
                
        Returns:
                geocoding_data_json (dict):  Dict in json format with geocoded data
    '''
    
    base_uri = 'https://geocoding-api.open-meteo.com/v1/search?'
    params_uri = f'name={city}&count=1&language=en&format=json'
    geocoding_request = base_uri + params_uri
    geocoding_response = requests.get(geocoding_request)
    geocoding_data = geocoding_response.text
    geocoding_data_json = json.loads(geocoding_data)
    
    return geocoding_data_json


def get_openmeteo_hist_api_request(latitude, longitude, start_date, end_date, freq, wx_params):
    ''' Generate the request string to get historical data

        Example: 
            https://open-meteo.com/en/docs/historical-weather-api#latitude=50.85&longitude=4.35&start_date=2023-04-12&
            end_date=2023-04-26&hourly=temperature_2m,relativehumidity_2m,precipitation

        Parameters:
            latitude (float):  latitude value
            longitude (float):  longitude value
            start_date (string):  starting date for data request (format is '%Y-%m-%d')
            end_date (string):  last date for data request (format is '%Y-%m-%d')
            freq (string): weather data frequency (e.g. 'hourly')
            wx_params (list): list of strings for weather parameters
            
        Returns:
            request (string):  request string to pass to historical weather API
    '''
    
    latitude = float(latitude)
    longitude = float(longitude)
    base_uri = 'https://archive-api.open-meteo.com/v1/archive?'
    wx_params = (','.join(wx_params))
    params_uri = f"latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&{freq}={wx_params}"
    request = base_uri + params_uri
    
    return request

def get_openmeteo_forecast_api_request(latitude, longitude, start_date, end_date, freq, wx_params):
    ''' Generate the request string to get forecasted data for the next 3 days including the start and end dates

        Example: 
            https://api.open-meteo.com/v1/forecast?latitude=50.85&longitude=4.35
            &hourly=temperature_2m,apparent_temperature,precipitation&forecast_days=3&start_date=2023-06-29&end_date=2023-07-01

        Parameters:
            latitude (float):  latitude value
            longitude (float):  longitude value
            start_date (string):  starting date for data request (format is '%Y-%m-%d')
            end_date (string):  last date for data request (format is '%Y-%m-%d')
            freq (string): weather data frequency (e.g. 'hourly')
            wx_params (list): list of strings for weather parameters
            
        Returns:
            request (string):  request string to pass to historical weather API
    '''
    
    latitude = float(latitude)
    longitude = float(longitude)
    base_uri = 'https://api.open-meteo.com/v1/forecast?'
    wx_params = (','.join(wx_params))
    params_uri = f"latitude={latitude}&longitude={longitude}&start_date={start_date}&end_date={end_date}&{freq}={wx_params}&forecast_days=3"
    request = base_uri + params_uri
    
    return request

def get_openmeteo_forecast_past_api_request(latitude, longitude, past_days, freq, wx_params):
    ''' Generate the request string to get forecasted data for the next 3 days including the start and end dates

        Example: 
            https://api.open-meteo.com/v1/forecast?latitude=52.52&longitude=13.41
            &past_days=10&hourly=temperature_2m,relativehumidity_2m,windspeed_10m

        Parameters:
            latitude (float):  latitude value
            longitude (float):  longitude value
            past_days
            freq (string): weather data frequecy (e.g. 'hourly')
            wx_params (list): list of strings for weather parameters
            
        Returns:
            request (string):  request string to pass to historical weather API
    '''
    
    latitude = float(latitude)
    longitude = float(longitude)
    base_uri = 'https://api.open-meteo.com/v1/forecast?'
    wx_params = (','.join(wx_params))
    params_uri = f"latitude={latitude}&longitude={longitude}&past_days={past_days}&{freq}={wx_params}&forecast_days=3"
    request = base_uri + params_uri
    
    return request


def get_openmeteo_api_response(api_request, freq, wx_params):
    ''' Make a request to historical weather api and convert to pandas dataframe

        Parameters:
            api_request (string):  request string obtained from function 'get_openmeteo_hist_api_request'
            wx_params (list): list of strings for weather parameters
            
        Returns:
            df_wx (DataFrame):  pandas DataFrame with weather data
    '''
    response = requests.get(api_request)
    data = response.text
    data = json.loads(data)

    df_wx = pd.DataFrame()
    df_wx['datetime_utc'] = pd.to_datetime(data[freq]['time'], utc=True)
    for param in wx_params:
        df_wx[param] = data[freq][param]

    return df_wx

def get_wx_df(start_date, end_date, past_days, freq, wx_params, city, request_type='None'):
    '''Get weather dataframe (historical or forecast) for a city with renamed columns

        Parameters:
            start_date (string):  starting date for data request (format is '%Y-%m-%d')
            end_date (string):  last date for data request (format is '%Y-%m-%d')
            freq (string): weather data frequency (e.g. 'hourly')
            wx_params (list): list of strings for weather parameters
            city (string):  city name for data request
            request_type (string):  'historical' or 'forecast'
            
        Returns:
            df_wx (DataFrame):  pandas DataFrame with returned historical weather data
    
    '''
    geocoding_data = get_openmeteo_geocoding_data(city)
    latitude = geocoding_data['results'][0]['latitude']
    longitude = geocoding_data['results'][0]['longitude']

    if request_type == 'historical':
        api_request = get_openmeteo_hist_api_request(latitude, longitude, start_date, end_date, freq, wx_params)
    elif request_type == 'forecast':
        api_request = get_openmeteo_forecast_api_request(latitude, longitude, start_date, end_date, freq, wx_params)
    elif request_type == 'forecast_past':
        api_request = get_openmeteo_forecast_past_api_request(latitude, longitude, past_days, freq, wx_params)
    else:
        raise ValueError
    
    df_wx = get_openmeteo_api_response(api_request, freq, wx_params)

    # rename columns except for datetime_utc column
    columns_renamed = [df_wx.columns[0]] 
    for column in df_wx.columns[1:]:
        new_column = column + '_' + city
        columns_renamed.append(new_column)

    df_wx.columns = columns_renamed
    df_wx.set_index('datetime_utc', inplace=True)
    return df_wx