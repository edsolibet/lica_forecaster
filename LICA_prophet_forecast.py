# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:22:17 2022

@author: carlo
"""
# Essential libraries
# =============================================================================
import numpy as np
import pandas as pd
import seaborn as sns
from io import BytesIO
import openpyxl, requests
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from datetime import date, timedelta
import streamlit as st

# Modelling and Forecasting
# =============================================================================
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.utilities import regressor_coefficients
from pytrends.request import TrendReq

import itertools
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)

def ratio(a, b):
    '''
    Function for calculating ratios to avoid inf
    '''
    return a/b if b else 0

@st.experimental_memo
def get_data(platform):
    '''
    Imports data based on selected platform
    '''
    # Data download
    # =========================================================================
    if platform == "Gulong.ph":
        # https://docs.google.com/spreadsheets/d/17Yb2nYaf_0KHQ1dZBGBJkXDcpdht5aGE/edit?rtpof=true#gid=1145755332
        sheet_id = "17Yb2nYaf_0KHQ1dZBGBJkXDcpdht5aGE"
        sheet_name = 'summary'
        url = "https://docs.google.com/spreadsheets/export?exportFormat=xlsx&id=" + sheet_id
        
    elif platform=='Mechanigo.ph':
        # https://docs.google.com/spreadsheets/d/18_Kwp3izJWlO2HSjrjULl3ydhnMS0KHY/edit#gid=49034711
        sheet_id = "18_Kwp3izJWlO2HSjrjULl3ydhnMS0KHY"
        sheet_name = 'summary'
        url = "https://docs.google.com/spreadsheets/export?exportFormat=xlsx&id=" + sheet_id
        
    res = requests.get(url)
    data_ = BytesIO(res.content)
    xlsx = openpyxl.load_workbook(filename=data_)
    traffic_data_ = pd.read_excel(data_, sheet_name = sheet_name)
        
    # Data preparation
    # =========================================================================
    
    traffic_data_ = traffic_data_.dropna(axis=1)
    traffic_data_ = traffic_data_.rename(columns={'Unnamed: 0': 'date'})
    traffic_data_.loc[:, 'month'] = pd.to_datetime(traffic_data_.loc[:,'date']).dt.month
    traffic_data_.loc[:, 'year'] = traffic_data_.loc[:, 'date'].dt.year
    traffic_data_.loc[:, 'month_day'] = traffic_data_.loc[:,'date'].dt.day.values
    traffic_data_.loc[:, 'weekday'] = traffic_data_.loc[:,'date'].dt.dayofweek + 1
    traffic_data_.loc[:, 'week_of_month'] = traffic_data_.loc[:, 'date'].apply(lambda d: (d.day-1) // 7 + 1)
    traffic_data_.loc[:, 'week_number'] = traffic_data_.apply(lambda x: int(x['date'].strftime('%U')), axis=1)
    traffic_data_ = traffic_data_.set_index('date')
    traffic_data_ = traffic_data_.asfreq('1D')
    traffic_data_ = traffic_data_.sort_index()
    
    # Data engineering
    # =========================================================================

    clicks_cols = ['link_clicks_ga', 'link_clicks_fb']
    impressions_cols = ['impressions_ga', 'impressions_fb']
    purchases_backend_cols = [col for col in traffic_data_.columns if 'purchases_backend' in col]
    
    traffic_data_.loc[:, 'clicks_total'] = traffic_data_.loc[:,clicks_cols].sum(axis=1)
    traffic_data_.loc[:, 'impressions_total'] = traffic_data_.loc[:,impressions_cols].sum(axis=1)
    traffic_data_.loc[:, 'ctr_ga'] = traffic_data_.apply(lambda x: ratio(x['link_clicks_ga'], x['impressions_ga']), axis=1)
    traffic_data_.loc[:, 'ctr_fb'] = traffic_data_.apply(lambda x: ratio(x['link_clicks_fb'], x['impressions_fb']), axis=1)
    
    if platform == 'Gulong.ph':
        traffic_data_.loc[:, 'purchases_backend_total'] = traffic_data_.loc[:,purchases_backend_cols].sum(axis=1)
        traffic_data_.loc[:, 'purchases_backend_marketplace'] = traffic_data_.loc[:, 'purchases_backend_fb'] + traffic_data_.loc[:, 'purchases_backend_shopee'] + traffic_data_.loc[:, 'purchases_backend_lazada']
        traffic_data_.loc[:, 'purchases_backend_b2b'] = traffic_data_.loc[:, 'purchases_backend_b2b'] + traffic_data_.loc[:, 'purchases_backend_walk-in']
        traffic_data_.drop(labels = ['purchases_backend_shopee', 'purchases_backend_lazada', 
                                     'purchases_backend_fb', 'purchases_backend_walk-in', 
                                     'purchases_backend_nan'], axis=1, inplace=True)

    return traffic_data_.reset_index()

platform_data = {'Gulong.ph': ('sessions', 'purchases_backend_website'),
                 'Mechanigo.ph': ('sessions', 'bookings_ga')}

def make_forecast_dataframe(train_start, train_end, val_end = None, forecast_horizon = 0):
    '''
    Creates training dataframe and future dataframe
    
    Parameters
    ----------
    train: dataframe
        Training data set
    end: string
        Ending date of forecast interval
    
    Returns
    -------
    
    '''
    evals_index = pd.date_range(start=train_start.strftime('%Y-%m-%d'), end=train_end.strftime('%Y-%m-%d'), freq='D')
    if val_end == None:
        future_index = pd.date_range(start=train_start.strftime('%Y-%m-%d'), end=(train_end+timedelta(days=forecast_horizon)).strftime('%Y-%m-%d'), freq='D')

    else:
        future_index = pd.date_range(start=train_start.strftime('%Y-%m-%d'), end=val_end.strftime('%Y-%m-%d'), freq='D')
        
    evals = pd.DataFrame(index=evals_index).reset_index()
    future = pd.DataFrame(index=future_index).reset_index()
    return evals, future

def plot_forecast(data, forecast, param, end_train):
    
    dataset = data[data.set_index('date').index.isin(forecast.set_index('ds').index)].set_index('date')
    
    fig = go.Figure([
                go.Scatter(
                    name='Actual',
                    x=dataset.index,
                    y=dataset[param],
                    mode='markers',
                    marker=dict(color='rgb(255, 255, 255)',
                                size=6),
                ),
                go.Scatter(
                    name='yhat',
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    line=dict(color='rgb(30, 144, 255)'),
                ),
                go.Scatter(
                    name='yhat_upper',
                    x=forecast['ds'],
                    y=forecast['yhat_upper'],
                    mode='lines',
                    marker=dict(color='rgba(176, 225, 230, 0.3)'),
                    line=dict(width=0),
                    showlegend=False
                ),
                go.Scatter(
                    name='yhat_lower',
                    x=forecast['ds'],
                    y=forecast['yhat_lower'],
                    marker=dict(color="rgb(176, 225, 230, 0.3)"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(176, 225, 230, 0.3)',
                    fill='tonexty',
                    showlegend=False
                )
            ])
    
    fig.update_layout(
        yaxis_title=param,
        hovermode="x")
    # Change grid color and axis colors
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
    fig.show()
    st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    # DATA
    # =========================================================================
    st.sidebar.write('1. Data')
    with st.sidebar.expander('Data selection'):
        platform = st.selectbox('Select platform',
                                ('Gulong.ph', 'Mechanigo.ph'),
                                index=0)
        data = get_data(platform)
    with st.sidebar.expander('Columns'):
        date_col = st.selectbox('Date column', data.columns[data.dtypes=='datetime64[ns]'],
                                index=0)
        target_col = st.selectbox('Target column:', platform_data[platform],
                                  index=0)
    
    with st.sidebar.expander('Data Split'):
        st.write('Training dataset')
        tcol1, tcol2 = st.columns(2)
        date_series = pd.to_datetime(data.loc[:,date_col])
        with tcol1:
            train_start = st.date_input('Training data start date',
                                        value = pd.to_datetime('2022-03-01'),
                                        min_value=date_series.min().date(),
                                        max_value=date_series.max().date())
        with tcol2:
            train_end = st.date_input('Training data end date',
                                        value = pd.to_datetime('2022-07-31'),
                                        min_value= pd.to_datetime('2022-04-01'),
                                        max_value=date_series.max().date())
        if train_start >= train_end:
            st.error('Training data end should come after training data start.')
            
        st.write('Validation dataset')
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            val_start = st.date_input('Validation data start date',
                                        value = train_end + timedelta(days=1),
                                        min_value=train_end + timedelta(days=1),
                                        max_value=date_series.max().date())
        with vcol2:
            val_end = st.date_input('Validation data end date',
                                        value = date_series.max().date(),
                                        min_value= val_start + timedelta(days=1),
                                        max_value=date_series.max().date())
        if val_start >= val_end:
            st.error('Validation data end should come after validation data start.')
    
    # MODELLING
    # =========================================================================

    # set params dict
    params = {}
    
    with st.sidebar.expander('Changepoints'):
        
        n_changepoints = st.slider('Number of changepoints',
                                   min_value = 5,
                                   max_value = 100,
                                   value = 20,
                                   step = 5)
        changepoint_prior_scale = st.number_input('changepoint_prior_scale',
                                                  min_value=0.05,
                                                  max_value=50.0,
                                                  value=15.0,
                                                  step=0.05)
        params['n_changepoints'] = n_changepoints
        params['changepoint_prior_scale'] = changepoint_prior_scale
        
    with st.sidebar.expander('Other parameters'):
        changepoint_range = st.number_input('changepoint_range',
                                            min_value=0.1,
                                            max_value=1.0,
                                            value=0.8,
                                            step=0.1)
        params['changepoint_range'] = changepoint_range
        
        growth_type = st.selectbox('growth',
                                   options=['logistic', 'linear'],
                                   index = 0)
        
        params['growth'] = growth_type
    
    with st.sidebar.expander('Cap and floor'):
        use_cap = st.checkbox('Add cap value')
        if use_cap:
            cap_type = st.selectbox('Value cap type',
                                    options=['fixed', 'multiplier'])
            if cap_type == 'fixed':
                cap = st.number_input('Fixed cap value',
                                      min_value = 0,
                                      value = 100)
            elif cap_type == 'multiplier':
                cap = st.number_input('Cap multiplier',
                                      min_value = 1,
                                      value = 1)
        use_floor = st.checkbox('Add floor value')
        if use_floor:
            floor_type = st.selectbox('Value floor type',
                                    options=['fixed', 'multiplier'])
            if floor_type == 'fixed':
                floor = st.number_input('Fixed floor value',
                                      min_value = 0,
                                      value = 0)
            elif floor_type == 'multiplier':
                floor = st.number_input('Floor multiplier',
                                      min_value = 0,
                                      value = 0)
    
    # set base Prophet model
    # =========================================================================
    models = {'evals': Prophet(**params),
              'future': Prophet(**params)}
    
    # MODELLING
    # =========================================================================
    st.sidebar.write('2. Modelling')
    with st.sidebar.expander('Seasonalities'):
        # yearly
        seasonality_mode = st.selectbox('seasonality_mode',
                                        options = ['multiplicative', 'additive'],
                                        index = 0)
        models['evals'].seasonality_mode = seasonality_mode
        models['future'].seasonality_mode = seasonality_mode
        
        season_index = 0 if seasonality_mode == 'multiplicative' else 1
        
        seasonality_prior_scale = st.number_input('seasonality_prior_scale',
                                                  min_value=0.05,
                                                  max_value=50.0,
                                                  value=10.0,
                                                  step=0.05)

        models['evals'].seasonality_prior_scale = seasonality_prior_scale
        models['future'].seasonality_prior_scale = seasonality_prior_scale
        
        yearly_seasonality = st.selectbox('yearly_seasonality', 
                                          ('auto', False, 'custom'))
        if yearly_seasonality == 'custom':
                yearly_seasonality_mode = st.selectbox('Yearly seasonality mode',
                                                       options = ['multiplicative', 'additive'],
                                                       index = season_index)
                yearly_seasonality_order = st.number_input('Yearly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=5,
                                                           step=1)
                yearly_seasonality_prior_scale = st.number_input('Yearly seasonality prior scale',
                                                           min_value = 1.0,
                                                           max_value=30.0,
                                                           value=8.0,
                                                           step=1.0)
                models['evals'].yearly_seasonality = False
                models['future'].yearly_seasonality = False
                models['evals'].add_seasonality('yearly',
                                  period = 365,
                                  fourier_order = yearly_seasonality_order,
                                  prior_scale = yearly_seasonality_prior_scale)
                models['future'].add_seasonality('yearly',
                                  period = 365,
                                  fourier_order = yearly_seasonality_order,
                                  prior_scale = yearly_seasonality_prior_scale)
        else:
            models['evals'].yearly_seasonality = yearly_seasonality
            models['future'].yearly_seasonality = yearly_seasonality
        
        # monthly
        monthly_seasonality = st.selectbox('monthly_seasonality', 
                                          ('auto', False, 'custom'))
        if monthly_seasonality == 'custom':
            monthly_seasonality_mode = st.selectbox('Monthly seasonality mode',
                                                   options = ['multiplicative', 'additive'],
                                                   index = season_index)
            monthly_seasonality_order = st.number_input('Monthly seasonality order',
                                                       min_value = 1,
                                                       max_value=30,
                                                       value=5,
                                                       step=1)
            monthly_seasonality_prior_scale = st.number_input('Monthly seasonality prior scale',
                                                       min_value = 1.0,
                                                       max_value=30.0,
                                                       value=8.0,
                                                       step=1.0)
            models['evals'].monthly_seasonality = False
            models['future'].monthly_seasonality = False
            models['evals'].add_seasonality('monthly',
                              period = 30,
                              fourier_order = monthly_seasonality_order,
                              prior_scale = monthly_seasonality_prior_scale)
            models['future'].add_seasonality('monthly',
                              period = 30,
                              fourier_order = monthly_seasonality_order,
                              prior_scale = monthly_seasonality_prior_scale)
        else:
            models['evals'].monthly_seasonality = monthly_seasonality
            models['future'].monthly_seasonality = monthly_seasonality
            
        # weekly
        weekly_seasonality = st.selectbox('weekly_seasonality', 
                                          ('auto', False, 'custom'))
        if weekly_seasonality == 'custom':
            weekly_seasonality_mode = st.selectbox('Weekly seasonality mode',
                                                   options = ['multiplicative', 'additive'],
                                                   index = season_index)
            weekly_seasonality_order = st.number_input('Weekly seasonality order',
                                                       min_value = 1,
                                                       max_value=30,
                                                       value=5,
                                                       step=1)
            weekly_seasonality_prior_scale = st.number_input('Weekly seasonality prior scale',
                                                       min_value = 1.0,
                                                       max_value=30.0,
                                                       value=8.0,
                                                       step=1.0)
            models['evals'].weekly_seasonality = False
            models['evals'].add_seasonality('weekly',
                              period = 7,
                              fourier_order = weekly_seasonality_order,
                              prior_scale = weekly_seasonality_prior_scale)
            models['future'].weekly_seasonality = False
            models['future'].add_seasonality('weekly',
                              period = 7,
                              fourier_order = weekly_seasonality_order,
                              prior_scale = weekly_seasonality_prior_scale)
        else:
            models['evals'].weekly_seasonality = weekly_seasonality
            models['future'].weekly_seasonality = weekly_seasonality
            
    
    with st.sidebar.expander('Holidays'):
        add_holidays = st.checkbox('Public holidays')
        if add_holidays:
            models['evals'].add_country_holidays(country_name='PH')
            models['future'].add_country_holidays(country_name='PH')
        
        add_set_holidays = st.checkbox('Set holidays')
        holidays = []
        fathers_day = pd.DataFrame({
                        'holiday': 'fathers_day',
                        'ds': pd.to_datetime(['2022-06-19']),
                        'lower_window': -21,
                        'upper_window': 3})
        holidays_choices = {'fathers_day' : fathers_day}
        
        if add_set_holidays:
            set_holidays = st.multiselect('Set holidays',
                                          options = [holidays_choices[h].loc[0, 'holiday'] for h in holidays_choices.keys()],
                                          default = [holidays_choices[h].loc[0, 'holiday'] for h in holidays_choices.keys()])
            holidays.extend([holidays_choices[hol] for hol in set_holidays])
            
        add_custom_holidays = st.checkbox('Custom holidays')
        if add_custom_holidays:
            holiday_name = st.text_input('Holiday name')
            holiday_date = st.date_input('Holiday date')
            holiday_lower_window = st.number_input('Holiday lower window (days)',
                                                   value=-1)
            holiday_upper_window = st.number_input('Holiday upper window (days)',
                                                   value = 1)
            holiday = pd.DataFrame({'holiday' : holiday_name,
                                   'ds': pd.to_datetime([holiday_date]),
                                   'lower_window': holiday_lower_window,
                                   'upper_window': holiday_upper_window})
            holidays.append(holiday)
        if add_set_holidays or add_custom_holidays:
            models['evals'].holidays = pd.concat(tuple(holidays))
            models['future'].holidays = pd.concat(tuple(holidays))
            holiday_prior_scale = st.number_input('holiday_prior_scale',
                                                  min_value=0.05,
                                                  max_value=50.0,
                                                  value=10.0,
                                                  step=0.05)
            models['evals'].holiday_prior_scale = holiday_prior_scale
            models['future'].holiday_prior_scale = holiday_prior_scale
        
    
    with st.sidebar.expander('Regressors'):
        regressors = data.drop(columns=[date_col, target_col], axis=1).columns
        selected_regressors = st.multiselect('Select external metrics if any:',
                       options= regressors)
        
        
        
    with st.sidebar.expander('Metrics'):
        selected_metrics = st.multiselect('Select evaluation metrics',
                                          options = ['MAE', 'MSE', 'RMSE', 'MAPE'],
                                          default = ['MAE', 'MSE', 'RMSE', 'MAPE'])
    
    st.sidebar.write('4. Forecast')
    make_forecast_future = st.sidebar.checkbox('Make forecast on future dates')
    if make_forecast_future:
        with st.sidebar.expander('Horizon'):
            forecast_horizon = st.number_input('Forecast horizon in days',
                                               min_value = 1,
                                               max_value = 30,
                                               value = 15,
                                               step = 1)
            st.info('Forecast dates: \n {} to {}'.format(val_end+timedelta(days=1), 
                                                   val_end+timedelta(days=forecast_horizon)))
    else:
        forecast_horizon = 0
        # add regressors
    
    launch_forecast = st.sidebar.checkbox('Launch forecast')
    st.sidebar.write('\n\n\n')
    # start forecast results
    if launch_forecast:
        st.header('Model overview')
        models['evals'].params = params
        models['future'].params = params
        evals, future = make_forecast_dataframe(train_start, train_end, val_end, forecast_horizon)
        evals = pd.concat([evals, data.loc[evals.index, target_col]], axis=1).rename(columns={'index':'ds',
                                                                      target_col: 'y'})
        
        if use_cap and cap_type == 'fixed':
            evals['cap'] = cap
            future['cap'] = cap
        elif use_cap and cap_type == 'multiplier':
            evals['cap'] = evals['y']*cap
            future['cap'] = evals['y']*cap
        if use_floor and floor_type == 'fixed':
            evals['floor'] = floor
            future['floor'] = floor
        elif use_floor and floor_type == 'multiplier':
            evals['floor'] = evals['y']*floor
            future['floor'] = evals['y']*floor
        
        
        if make_forecast_future: 
            model = models['future']
            model.fit(evals)
            forecast = model.predict(future)
        else:
            model = models['evals']
            model.fit(evals)
            forecast = model.predict(evals)
        
        plot_forecast(data, forecast, target_col, train_end)
        
    