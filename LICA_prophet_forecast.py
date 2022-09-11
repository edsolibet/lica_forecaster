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

def extra_inputs(value, seasonality: str):
    if value == 'custom':
        additional_selectboxes = st.empty()
        with additional_selectboxes.container():
            mode = st.selectbox(seasonality + ' mode',
                                ('multiplicative', 'additive'))
            order = st.number_input(seasonality + ' order',
                                    min_value = 1,
                                    max_value=30,
                                    value=5,
                                    step=1)
            prior_scale = st.number_input(seasonality + ' prior scale',
                                    min_value = 1.0,
                                    max_value=30.0,
                                    value=8.0,
                                    step=1.0)
        return mode, order, prior_scale
    else:
        pass 


if __name__ == '__main__':
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
        
    st.sidebar.write('2. Modelling')
    with st.sidebar.expander('Prior scale'):
        changepoint_prior_scale = st.number_input('changepoint_prior_scale',
                                                  min_value=0.05,
                                                  max_value=50.0,
                                                  value=15.0,
                                                  step=0.05)
        seasonality_prior_scale = st.number_input('seasonality_prior_scale',
                                                  min_value=0.05,
                                                  max_value=50.0,
                                                  value=10.0,
                                                  step=0.05)
        holiday_prior_scale = st.number_input('holiday_prior_scale',
                                                  min_value=0.05,
                                                  max_value=50.0,
                                                  value=5.0,
                                                  step=0.05)
    
    with st.sidebar.expander('Cap and floor'):
        use_cap = st.checkbox('Add cap value')
        use_floor = st.checkbox('Add floor value')
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
    
    with st.sidebar.expander('Seasonalities'):
        # yearly
        yearly_seasonality = st.selectbox('yearly_seasonality', 
                                          ('auto', False, 'custom'))
        if yearly_seasonality == 'custom':
                yearly_seasonality_mode = st.selectbox('Yearly seasonality mode',
                                                       ('multiplicative', 'additive'))
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
        # monthly
        monthly_seasonality = st.selectbox('monthly_seasonality', 
                                          ('auto', False, 'custom'))
        if monthly_seasonality == 'custom':
            monthly_seasonality_mode = st.selectbox('Monthly seasonality mode',
                                                   ('multiplicative', 'additive'))
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
        # weekly
        weekly_seasonality = st.selectbox('weekly_seasonality', 
                                          ('auto', False, 'custom'))
        if weekly_seasonality == 'custom':
            weekly_seasonality_mode = st.selectbox('Weekly seasonality mode',
                                                   ('multiplicative', 'additive'))
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
    
    with st.sidebar.expander('Holidays'):
        add_holidays = st.checkbox('Public holidays')
    
    with st.sidebar.expander('Regressors'):
        regressors = data.drop(columns=[date_col, target_col], axis=1).columns
        selected_regressors = st.multiselect('Select external metrics if any:',
                       options= regressors)
        
    with st.sidebar.expander('Other parameters'):
        changepoint_range = st.number_input('changepoint_range',
                                            min_value=0.1,
                                            max_value=1.0,
                                            value=0.8,
                                            step=0.1)
        growth_type = st.selectbox('growth',
                                   options=['logistic', 'linear'])
    
    st.sidebar.write('3. Evaluation')
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
        
    with st.sidebar.expander('Metrics'):
        selected_metrics = st.multiselect('Select evaluation metrics',
                                          options=['MAE', 'MSE', 'RMSE', 'MAPE'],
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
                                                   val_end+timedelta(days=15)))
        
        # add regressors
    
    launch_forecast = st.sidebar.checkbox('Launch forecast')
    st.sidebar.write('\n\n\n')
    # start forecast results
    if launch_forecast:
        st.header('Model overview')

        m = Prophet()
        
    
    