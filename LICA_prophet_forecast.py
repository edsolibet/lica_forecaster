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

# Google trends
# from pytrends.request import TrendReq
# save models
# from joblib import dump, load

def sunday_of_calendarweek(year, week):
    '''
    Obtain date of sunday for the given calendar week
    
    Parameters
    ----------
    year : int
    week : int
    
    Returns
    -------
    date string of sunday
    '''
    first = date(year, 1, 1)
    base = 1 if first.isocalendar()[1] == 1 else 8
    return first + timedelta(days=(base - first.isocalendar()[2] + 7 * (week - 1) - 1))

def ratio(a, b):
    '''
    Function for calculating ratios to avoid inf
    '''
    return a/b if b else 0

@st.experimental_memo
def get_data():
    
    # Data download
    # =========================================================================
    sheet_id = "17Yb2nYaf_0KHQ1dZBGBJkXDcpdht5aGE"
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
    traffic_data_.loc[:, 'week_first_day'] = traffic_data_.apply(lambda x: sunday_of_calendarweek(int(x['year']), int(x['week_number'])), axis=1)
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
    traffic_data_.loc[:, 'purchases_backend_total'] = traffic_data_.loc[:,purchases_backend_cols].sum(axis=1)
    traffic_data_.loc[:, 'purchases_backend_marketplace'] = traffic_data_.loc[:, 'purchases_backend_fb'] + traffic_data_.loc[:, 'purchases_backend_shopee'] + traffic_data_.loc[:, 'purchases_backend_lazada']
    traffic_data_.loc[:, 'purchases_backend_b2b'] = traffic_data_.loc[:, 'purchases_backend_b2b'] + traffic_data_.loc[:, 'purchases_backend_walk-in']
    traffic_data_.drop(labels = ['purchases_backend_shopee', 'purchases_backend_lazada', 
                                 'purchases_backend_fb', 'purchases_backend_walk-in', 
                                 'purchases_backend_nan'], axis=1, inplace=True)
    
    traffic_data_.loc[:, 'ctr_ga'] = traffic_data_.apply(lambda x: ratio(x['link_clicks_ga'], x['impressions_ga']), axis=1)
    traffic_data_.loc[:, 'ctr_fb'] = traffic_data_.apply(lambda x: ratio(x['link_clicks_fb'], x['impressions_fb']), axis=1)
    
    return traffic_data_

def make_forecast_dataframe(train, end, cap=None, floor=None):
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
    index = pd.date_range(start=min(train.index).strftime('%Y-%m-%d'), end=end, freq='D')
    df_train = train.reset_index()
    df_future = pd.DataFrame(index=index).reset_index()
    param = train.reset_index().columns[-1]
    df_train.rename(columns={'date': 'ds', param : 'y'}, inplace=True)
    df_future.rename(columns={'index': 'ds', param : 'y'}, inplace=True)
    if cap is not None:
        if callable(cap):
            df_train['cap'] = df_train['y'].apply(cap)
            df_future.loc[df_train.index.min():df_train.index.max(), 'cap'] = df_train['cap']
            df_future.loc[df_train.index.max():, 'cap'] = df_train['cap'].max()
            
        else:
            df_future['cap'] = cap
            df_train['cap'] = cap
    if floor is not None:
        if callable(floor):
            df_train['floor'] = df_train['y'].apply(cap)
            df_future.loc[df_train.index.min():df_train.index.max(), 'floor'] = df_train['floor']
            df_future.loc[df_train.index.max():, 'floor'] = df_train['floor'].min()
        else:
            df_future['floor'] = floor
            df_train['floor'] = floor
    return df_train, df_future

# custom holidays
# ============================================================================
fathers_day = pd.DataFrame({
    'holiday': 'fathers_day',
    'ds': pd.to_datetime(['2022-06-19']),
    'lower_window': -21,
    'upper_window': 3})

# add seasonalities
# ============================================================================
# seasonality dictionary 
seasonality_dict = {'weekly':    {'period': 7,
                                  'fourier_order': 20},
                    'bimonthly': {'period': 15.2,
                                  'fourier_order': 10},
                    'monthly':   {'period': 30.4,
                                  'fourier_order': 15}}

def add_seasonality(model, seasonality_dict):
    '''
    Helper function to add seasonality to Prophet model

    Parameters
    ----------
    model : Prophet
        Prophet model instance
    seasonality_dict : dictionary
        dictionary of seasonalities to be added to model

    Returns
    -------
    m : Prophet
        Model instance with added seasonalities

    '''
    m = model
    for season in seasonality_dict.keys():
        m.add_seasonality(name = season,
                              period = seasonality_dict[season]['period'],
                              fourier_order = seasonality_dict[season]['fourier_order'])
    return m

# add regressors and exogenous variables
# ============================================================================

def is_saturday(ds):
    date = pd.to_datetime(ds)
    return ((date.dayofweek + 1) == 6)*1

def is_sunday(ds):
    date = pd.to_datetime(ds)
    return ((date.dayofweek + 1) == 7)*1

regs = {'is_saturday': is_saturday,
        'is_sunday'  : is_sunday}

# Select exogenous variables, including those generated by one hot encoding.
exog_num_cols = {'sessions': ['ctr_fb', 'ctr_ga', 'ad_costs_fb_total', 'ad_costs_ga', 
                 'landing_page_views', 'impressions_fb', 'impressions_ga', 'pageviews'],
                 'purchases_backend_website': ['ctr_fb', 'ctr_ga', 'ad_costs_fb_total', 'ad_costs_ga', 
                 'landing_page_views', 'impressions_fb', 'impressions_ga', 'cancellations',
                 'rejections']}

def get_gtrend_data(kw_list, start_train, end_train):
    '''
    Get google trend data for specifie keywords
    '''
    pytrend = TrendReq()
    start = pd.to_datetime(start_train)
    end = pd.to_datetime(end_train)
    historicaldf = pytrend.get_historical_interest(kw_list, 
                            year_start=start.year, 
                            month_start=start.month, 
                            day_start=start.day, 
                            year_end=end.year, 
                            month_end=end.month, 
                            day_end=end.day, 
                            cat=0, 
                            geo='', 
                            gprop='', 
                            sleep=0)
    historicaldf.index = historicaldf.index.strftime('%Y-%m-%d')
    return historicaldf[kw_list].reset_index().groupby('date').mean().fillna(0)


def plot_forecast(data, forecast, param, end_train, end_pred):
    labels = ['train', 'forecast']
    color_list = ['#0000FF', '#008000', '#7F00FF', '#800020']
    color_shade_list = ['#89CFF0', '#AFE1AF', '#CF9FFF', '#FAA0A0']
    # import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    dataset = data[data.index.isin(forecast.set_index('ds').index)]
    #train = forecast.set_index('ds').loc[:end_train]
    #end_train_plus_1 = (pd.to_datetime(end_train) + timedelta(days=1)).strftime('%Y-%m-%d')
    #preds = forecast.set_index('ds').loc[end_train_plus_1:end_pred]
    ax.plot(dataset.index, dataset[param], 'o-', color='k', ms=4, label='data')
    ax.plot(forecast['ds'], forecast['yhat'], '--', color = color_list[0], label=labels[1])
    #ax.plot(data_test.index[-16:], data_test[param].iloc[-16:], linewidth=2)
    ax.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], facecolor=color_shade_list[0], alpha=0.5)
    ax.legend(prop={'size': 15})
    ax.set_xlabel('Date')
    ax.set_ylabel(param)
    ax.set_xlim([dataset.index.min(),forecast['ds'].max()])
    ax.set_ylim([0, forecast['yhat_upper'].max()])
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
def plot_forecast_(data, forecast, param, end_train, end_pred):
    
    dataset = data[data.index.isin(forecast.set_index('ds').index)]
    
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


def get_regressors(reg_list, level = None):
    '''
    Selects type of columns from forecaster dataframe
    
    Parameters
    ----------
    reg_list : list
        forecast.columns
    level : string or None
        None for regular columns
        'lower' or 'upper' for lower or upper value limits
    
    Returns
    -------
    list
        list of desired regressors
    
    '''
    col_list = list()
    remove_list = ['ds', 'cap', 'floor', 'yhat', 'yhat_lower', 'yhat_upper']
    if level == None:
        for col in reg_list:
            if '_lower' in col or '_upper' in col:
                continue
            else:
                col_list.append(col)
    else:
        for col in reg_list:
            if level in col:
                col_list.append(col)
            else:
                continue
    return [i for i in col_list if i not in remove_list]



# dictionary for to for setting prediction horizon from date today
predict_horizon_dict = {'7 days': 7,
                        '15 days': 15,
                        '30 days' : 30}

platform_data = {'Gulong.ph': ('sessions', 'purchases_backend_website'),
                 'Mechanigo.ph': ('sessions', 'website bookings')}

st.sidebar.write('1. Data')
with st.sidebar.form('Dataset'):
    platform = st.selectbox('Select platform',
                                    ('Gulong.ph', 'Mechanigo.ph'))
    data = st.selectbox('Select data:', platform_data[platform])