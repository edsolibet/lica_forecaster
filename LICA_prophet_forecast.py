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
                                              value=10.0,
                                              step=0.05)
    seasonality_prior_scale = st.number_input('seasonality_prior_scale',
                                              min_value=0.05,
                                              max_value=50.0,
                                              value=5.0,
                                              step=0.05)
    holiday_prior_scale = st.number_input('holiday_prior_scale',
                                              min_value=0.05,
                                              max_value=50.0,
                                              value=5.0,
                                              step=0.05)

with st.sidebar.expander('Seasonalities'):
    # yearly
    yearly_seasonality = st.selectbox('yearly_seasonality', 
                                      ('auto', False, 'custom'))
    additional_selectboxes = st.empty()
    if yearly_seasonality == 'custom':
        with additional_selectboxes.container():
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

with st.expander('Holidays'):
    add_holidays = st.checkbox('Public holidays')
    
        