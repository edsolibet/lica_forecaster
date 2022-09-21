# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 01:10:34 2022

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
import plotly.express as px
import plotly.graph_objects as go
from prophet.plot import plot_plotly, plot_components_plotly
from pytrends.request import TrendReq

import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
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
    traffic_data_.loc[:, 'ctr_total'] = traffic_data_.apply(lambda x: ratio(x['clicks_total'], x['impressions_total']), axis=1)
    traffic_data_.loc[:, 'ad_costs_total'] = traffic_data_.loc[:, 'ad_costs_ga'] + traffic_data_.loc[:, 'ad_costs_fb_total']
    
    if platform == 'Gulong.ph':
        traffic_data_.loc[:, 'purchases_backend_total'] = traffic_data_.loc[:,purchases_backend_cols].sum(axis=1)
        traffic_data_.loc[:, 'purchases_backend_marketplace'] = traffic_data_.loc[:, 'purchases_backend_fb'] + traffic_data_.loc[:, 'purchases_backend_shopee'] + traffic_data_.loc[:, 'purchases_backend_lazada']
        traffic_data_.loc[:, 'purchases_backend_b2b'] = traffic_data_.loc[:, 'purchases_backend_b2b'] + traffic_data_.loc[:, 'purchases_backend_walk-in']
        traffic_data_.drop(labels = ['purchases_backend_shopee', 'purchases_backend_lazada', 
                                     'purchases_backend_fb', 'purchases_backend_walk-in', 
                                     'purchases_backend_nan'], axis=1, inplace=True)

    return traffic_data_.reset_index()


def make_forecast_dataframe(start, end):
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
    dates = pd.date_range(start=start, end=end, freq='D')
    df = pd.DataFrame(dates).rename(columns={0:'ds'})
    return df

    
def plot_forecast_(data, forecast, param):
    
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

def plot_forecast_vs_actual_scatter(evals, forecast):
    '''
    Create a plot for forecasted values vs actual data
    
    Parameters
    ----------
    evals : dataframe
        training dataframe
    forecast: dataframe
        output dataframe from model.predict
    
    Returns
    -------
    fig : figure
        plotly figure
    '''
    evals_ = evals[['ds', 'y']]
    evals_df = pd.concat([evals_, forecast['yhat']], axis=1)
    fig = px.scatter(evals_df,
                     x = 'y',
                     y = 'yhat',
                     opacity=0.5,
                     hover_data={'ds': True, 'y': ':.4f', 'yhat': ':.4f'})
    
    fig.add_trace(
        go.Scatter(
            x=evals_df['y'],
            y=evals_df['y'],
            name='optimal',
            mode='lines',
            line=dict(color='red', width=1.5)))
    
    fig.update_layout(
        xaxis_title="Truth", yaxis_title="Forecast", legend_title_text="", height=450, width=800)
    
    return fig

def make_separate_components_plot(forecast):
    pass


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


def calc_yhat(forecast, coefs, model):
    '''
    Calculate the forecasts from the regressor coefficients
    
    yhat = trend * (1 + holidays(t) + seasons(t) + regressor_comps(t) 
                    + extra_regressors_multiplicative(t))

    Parameters
    ----------
    forecast : dataframe
        Predictions made by model
    coefs : dataframe
        Regressor coefficients, center
        Input: regressor_coefficients(m).set_index('regressor')
    model : model
        fitted FB prophet model

    Returns
    -------
    yhat : list
        Calculated forecast

    '''
    
    yhat = list()
    reg_comps_list, holiday_list, seasons_list, tot = [], [], [], []
    for row in range(len(forecast)):
        reg_comps = 0
        for r in coefs.index:
            reg_comps *= (forecast.loc[row, r] - coefs.loc[r, 'center'])*coefs.loc[r, 'coef']
        reg_comps_list.append(reg_comps)
        holiday = 0
        for h in list(model.train_holiday_names):
            holiday += forecast.loc[row, h]
        holiday_list.append(holiday)
        seasons = 0
        for s in list(model.seasonalities.keys()):
            seasons += forecast.loc[row, s]
        seasons_list.append(seasons)
        tot.append(reg_comps + holiday + seasons + forecast.loc[row, 'extra_regressors_multiplicative'])
        yhat.append(forecast.loc[row,'trend']*(1 + seasons + holiday + reg_comps +  forecast.loc[row, 'extra_regressors_multiplicative']) + 
                    forecast.loc[row, 'additive_terms'])
    
    df_yhat = pd.DataFrame(list(zip(forecast.ds, reg_comps_list, holiday_list, seasons_list, forecast.extra_regressors_multiplicative, tot, yhat)), 
                           index = forecast.index, columns=['ds', 'regressors', 'holiday', 'seasons', 'extra_reg_mult', 'tot', 'yhat'])
    return df_yhat




if __name__ == '__main__':
    st.title('LICA Target Setting and Forecasting App')
    st.markdown('''
                This tool forecasts the future values of important metrics to be used
                for target setting.
                ''')
    st.sidebar.markdown('# 1. Data')
    with st.sidebar.expander('Data selection'):
        platform = st.selectbox('Select platform',
                                        ('Gulong.ph', 'Mechanigo.ph'),
                                        index=0)
        data = get_data(platform)
        # date column
        date_col = st.selectbox('Date column', data.columns[data.dtypes=='datetime64[ns]'],
                                index=0)
        
        platform_data = {'Gulong.ph': ('sessions', 'purchases_backend_website'),
                 'Mechanigo.ph': ('sessions', 'bookings_ga')}
        # target column
        param = st.selectbox('Target column:', platform_data[platform],
                                index=0)
        # select data
        st.markdown('### Training dataset')
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
            st.error('Train_end should come after train_start.')
        

        st.markdown('### Validation dataset')
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            val_start = st.date_input('val_start date',
                                        value = train_end + timedelta(days=1),
                                        min_value=train_end + timedelta(days=1),
                                        max_value=date_series.max().date())
        with vcol2:
            val_end = st.date_input('val_end date',
                                        value = date_series.max().date(),
                                        min_value= val_start + timedelta(days=1),
                                        max_value=date_series.max().date())
        if val_start >= val_end:
            st.error('Val_end should come after val_start.')
    
        # create training dataframe
        # 
        if pd.Timestamp(val_end) <= data.date.max():
            date_series = make_forecast_dataframe(train_start, val_end)
            param_series = data[data.date.isin(date_series.ds.values)][param].reset_index()
            evals = pd.concat([date_series, param_series], axis=1).rename(columns={0: 'ds',
                                                                                   param:'y'}).drop('index', axis=1)
            # check for NaNs
            if evals.y.isnull().sum() > 0.5*len(evals):
                st.warning('Evals data contains too many NaN values')
                st.dataframe(evals)
        else:
            st.error('val_end is outside available dataset.')
    
         
    with st.sidebar.expander('Forecast'):
        make_forecast_future = st.checkbox('Make forecast on future dates')
        if make_forecast_future:
            forecast_horizon = st.number_input('Forecast horizon (days)',
                                               min_value = 1,
                                               max_value = 30,
                                               value = 15,
                                               step = 1)
            st.info(f'''Forecast dates:\n 
                    {val_end+timedelta(days=1)} to 
                    {val_end+timedelta(days=forecast_horizon)}''')

            future = make_forecast_dataframe(start=train_start, 
                                             end=val_end+timedelta(days=forecast_horizon))
                             
    # MODELLING
    # ========================================================================
    st.sidebar.markdown('# 2. Modelling')
    # default parameters for target cols
    default_params = {'sessions':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 15.0,
              'n_changepoints' : 30,
              'cap' : 2500.0,
              },
              'purchases_backend_website':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 15.0,
              'n_changepoints' : 30,
              'cap' : 30.0,
              },
              'bookings_ga':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 15.0,
              'n_changepoints' : 30,
              'cap' : 30.0,
              }
              }
    
    params = {}
    with st.sidebar.expander('Model and Growth type'):
        
        growth_type = st.selectbox('growth',
                                   options=['logistic', 'linear'],
                                   index = 0)
        
        params['growth'] = growth_type
        if growth_type == 'logistic':
            # if logistic growth, cap value is required
            cap = st.number_input('Enter fixed cap value',
                                  min_value = 0.0,
                                  max_value = None,
                                  value = default_params[param]['cap'],
                                  step = 0.01)
            evals.loc[:, 'cap'] = cap
            # if forecast future, also apply cap to future df
            if make_forecast_future:
                future.loc[:,'cap'] = cap
            
            # floor is optional
            use_floor = st.checkbox('Add floor value')
            if use_floor:
                floor = st.number_input('Enter fixed floor value',
                                      min_value = 0.0,
                                      max_value = None,
                                      value = 0.0)
                evals.loc[:, 'floor'] = floor
                if make_forecast_future:
                    future.loc[:,'floor'] = floor
            
                # check viability of cap value
                if cap <= floor:
                    st.error('Cap value should be greater than floor value')
    
    # CHANGEPOINTS
    # =========================================================================
    with st.sidebar.expander('Changepoints'):

        n_changepoints = st.slider('Number of changepoints',
                                   min_value = 5,
                                   max_value = 100,
                                   value = default_params[param]['n_changepoints'],
                                   step = 5)
        changepoint_prior_scale = st.number_input('changepoint_prior_scale',
                                    min_value=0.05,
                                    max_value=50.0,
                                    value= float(default_params[param]['changepoint_prior_scale']),
                                    step=0.05)
        changepoint_range = st.number_input('changepoint_range',
                                    min_value=0.05,
                                    max_value=1.0,
                                    value=0.95,
                                    step=0.05)
        
        # add selected inputs to params dict
        params['n_changepoints'] = n_changepoints
        params['changepoint_prior_scale'] = changepoint_prior_scale
        params['changepoint_range'] = changepoint_range
        
    # apply params to model
    model = Prophet(**params)  # Input param grid

    # SEASONALITIES
    # ========================================================================
    with st.sidebar.expander('Seasonalities'):
        season_model = st.selectbox('Add seasonality', 
                            options = ['Auto', 'True', 'False'],
                            index = 1,
                            key = 'season_model')
        
        seasonality_scale_dict = {'sessions': 6,
                                  'purchases_backend_website': 3,
                                  'bookings_ga': 3}
        
        if season_model == 'True':
            model.daily_seasonality = 'auto'
            
            seasonality_mode = st.selectbox('seasonality_mode',
                                        options = ['multiplicative', 'additive'],
                                        index = 1)
            model.seasonality_mode = seasonality_mode
            
            set_seasonality_prior_scale = st.checkbox('Set seasonality_prior_scale')
            # add info
            if set_seasonality_prior_scale:
                seasonality_prior_scale = st.number_input('overall_seasonality_prior_scale',
                                                      min_value= 1.0,
                                                      max_value= 30.0,
                                                      value=float(seasonality_scale_dict[param]),
                                                      step = 1.0)
            else:
                seasonality_prior_scale = 1
                
            model.seasonality_prior_scale = seasonality_prior_scale
            
            yearly_seasonality = st.selectbox('yearly_seasonality', 
                                          ('auto', False, 'custom'))
            if yearly_seasonality == 'custom':
                model.yearly_seasonality = False
                yearly_seasonality_order = st.number_input('Yearly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=5,
                                                           step=1)
                if set_seasonality_prior_scale is False:
                    yearly_prior_scale = st.number_input('Yearly seasonality prior scale',
                                                           min_value = 1.0,
                                                           max_value=30.0,
                                                           value=8.0,
                                                           step=1.0)
                # add yearly seasonality to model
                model.add_seasonality(name='yearly', 
                                      period = 365,
                                      fourier_order = yearly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else yearly_prior_scale) # add seasonality
            
            monthly_seasonality = st.selectbox('monthly_seasonality', 
                                          options = ('Auto', 'False', 'Custom'),
                                          index = 2)
            if monthly_seasonality == 'Custom':
                model.monthly_seasonality = False
                monthly_seasonality_order = st.number_input('Monthly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=9,
                                                           step=1)
                if set_seasonality_prior_scale is False:
                    monthly_prior_scale = st.number_input('Monthly seasonality prior scale',
                                                           min_value = 1.0,
                                                           max_value=30.0,
                                                           value=8.0,
                                                           step=1.0)
                # add monthly seasonality to model
                model.add_seasonality(name='monthly', 
                                      period = 30.4,
                                      fourier_order = monthly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else monthly_prior_scale) # add seasonality
            
            weekly_seasonality = st.selectbox('weekly_seasonality', 
                                          ('Auto', 'False', 'Custom'))
            if weekly_seasonality == 'Custom':
                model.weekly_seasonality = False
                weekly_seasonality_order = st.number_input('Weekly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=6,
                                                           step=1)
                if set_seasonality_prior_scale is False:
                    weekly_prior_scale = st.number_input('Weekly seasonality prior scale',
                                                           min_value = 1.0,
                                                           max_value=30.0,
                                                           value=8.0,
                                                           step=1.0)
                # add weekly seasonality to model
                model.add_seasonality(name='weekly', 
                                      period = 7,
                                      fourier_order = weekly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else weekly_prior_scale) # add seasonality
            
        elif season_model == 'auto':
            # selected 'auto' 
            model.yearly_seasonality = 'auto'
            model.monthly_seasonality = 'auto'
            model.weekly_seasonality = 'auto'
            model.daily_seasonality = 'auto'
        
        else:
            # selected False - no seasonality
            model.yearly_seasonality = False
            model.monthly_seasonality = False
            model.weekly_seasonality = False
            model.daily_seasonality = False
    
    # HOLIDAYS
    # =========================================================================
    with st.sidebar.expander('Holidays'):
        add_holiday = st.checkbox('Add holidays', 
                            value = True,
                            key = 'holiday_model')
        if add_holiday:
            # add public holidays
            add_public_holidays = st.checkbox('Public holidays')
            if add_public_holidays:
                model.add_country_holidays(country_name='PH')
            
            # add set_holidays
            add_set_holidays = st.checkbox('Saved holidays')
            if add_set_holidays:
                fathers_day = pd.DataFrame({
                    'holiday': 'fathers_day',
                    'ds': pd.to_datetime(['2022-06-19']),
                    'lower_window': -21,
                    'upper_window': 3})
                
                holidays_set = {'fathers_day': fathers_day}
                
                selected_holidays = st.multiselect('Select saved holidays',
                                                   options=list(holidays_set.keys()),
                                                   default = list(holidays_set.keys()))
                
                model.holidays = pd.concat(selected_holidays)
            
            holiday_scale_dict = {'sessions': 3,
                                  'purchases_backend_website': 3,
                                  'bookings_ga': 3}
            
            holiday_scale = st.number_input('holiday_prior_scale',
                                            min_value = 1.0,
                                            max_value = 30.0,
                                            value = float(holiday_scale_dict[param]),
                                            step = 1.0)
            # set holiday prior scale
            model.holiday_prior_scale = holiday_scale
            
        else:
            # no holiday effects
            model.holidays = None
            model.holiday_prior_scale = 0
    
    # REGRESSORS
    # =========================================================================
    with st.sidebar.expander('Regressors'):
        
        def is_saturday(ds):
            # check if saturday
            date = pd.to_datetime(ds)
            return ((date.dayofweek + 1) == 6)*1
        
        def is_sunday(ds):
            # check if sunday
            date = pd.to_datetime(ds)
            return ((date.dayofweek + 1) == 7)*1
        
        def get_gtrend_data(kw_list, df):
            '''
            Get google trend data for specifie keywords
            '''
            pytrend = TrendReq()
            start = pd.to_datetime(df.ds.min())
            end = pd.to_datetime(df.ds.max())
            historicaldf = pytrend.get_historical_interest(list(kw_list), 
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
            #historicaldf.index = historicaldf.index.strftime('%Y-%m-%d')
            return historicaldf[list(kw_list)].groupby(historicaldf.index.date).mean().fillna(0)
        
        # add data metrics option
        add_metrics = st.checkbox('Add data metrics',
                                  value = True)
        
        exog_num_cols = {'sessions': ['ctr_fb', 'ctr_ga', 'ctr_total', 'ad_costs_fb_total', 'ad_costs_ga', 'ad_costs_total',
                             'landing_page_views', 'impressions_fb', 'impressions_ga', 'pageviews'],
                         'purchases_backend_website': ['ctr_fb', 'ctr_ga', 'ctr_total', 'ad_costs_fb_total', 'ad_costs_ga', 'ad_costs_total',
                             'landing_page_views', 'impressions_fb', 'impressions_ga', 'cancellations',
                             'rejections'],
                         'bookings_ga': ['ctr_ga', 'ad_costs_ga', 'impressions_ga']}
        if add_metrics:
            # Select traffic metrics available from data.
            
            exogs = st.multiselect('Select data metrics',
                           options = exog_num_cols[param],
                           default = exog_num_cols[param])
            
            for exog in exogs:
                evals.loc[:, exog] = data[data.date.isin(evals.ds)][exog].values
                model.add_regressor(exog)
                
                # if forecast future
                if make_forecast_future:
                    future.loc[future.ds.isin(evals.ds), exog] = data[data.date.isin(evals.ds)][exog].values
            
        
        add_gtrends = st.checkbox('Add Google trends',
                                  value = False)
        if add_gtrends:
            # keywords
            kw_list = ['gulong.ph', 'gogulong']
            gtrends_st = st.text_area('Enter google trends keywords',
                                        value = ' '.join(kw_list))
            # selected keywords
            kw_list = gtrends_st.split(' ')
            # cannot generate data for dates in forecast horizon
            gtrends = get_gtrend_data(kw_list, evals)
            for g, gtrend in enumerate(gtrends.columns):
                evals.loc[:,kw_list[g]] = gtrends[gtrend].values
                future.loc[future.ds.isin(evals.ds), gtrend] = gtrends[gtrends.date.isin(evals.ds)][gtrend].values
                model.add_regressor(kw_list[g])
                
        # custom regressors (functions applied to dates)
        add_custom_reg = st.checkbox('Add custom regressors',
                                     value = True)
        if add_custom_reg:
            regs = {'is_saturday': evals.ds.apply(is_saturday),
                    'is_sunday'  : evals.ds.apply(is_sunday)}
            
            if make_forecast_future:
                regs_future = {'is_saturday': future.ds.apply(is_saturday),
                               'is_sunday'  : future.ds.apply(is_sunday)}
            
            # regressor multiselect
            regs_list = st.multiselect('Select custom regs',
                           options = list(regs.keys()),
                           default = list(regs.keys()))
            
            for reg in regs_list:
                evals.loc[:, reg] = regs[reg].values
                model.add_regressor(reg)
            
                if make_forecast_future:
                    future.loc[:, reg] = regs_future[reg].values
                
        
        if make_forecast_future:
            # input regressor data for future/forecast dates
            # if selected metrics is not None
            regressor_input = st.empty()
            if add_metrics and len(exogs) > 0:
                # provide input field
                with regressor_input.container():
                    for exog in exogs:
                        exog_data = data[data.date.isin(date_series.ds.values)][exog]
                        
                        data_input = st.selectbox('Data input type:',
                                             options=['total', 'average'],
                                             index=0)
                        if data_input == 'total':
                            # if data input is total
                            total = st.number_input('Select {} total over forecast period'.format(exog),
                                                   min_value = 0.0, 
                                                   value = exog_data.tail(forecast_horizon).sum(),
                                                   step = 0.01)
                            future.loc[future.index[-forecast_horizon:],exog] = np.full((forecast_horizon,), round(total/forecast_horizon, 3))
                        else:
                            # if data input is average
                            average = st.number_input('Select {} average over forecast period'.format(exog),
                                                   min_value = 0.0, 
                                                   value = exog_data.tail(forecast_horizon).mean(),
                                                   step = 0.01)
                            future.loc[future.index[-forecast_horizon:],exog] = np.full((forecast_horizon,), round(average, 3))
            else:
                # delete unused fields
                regressor_input.empty()
                
    with st.sidebar.expander('Cleaning'):
        st.write('Missing values')
        nan_err_container = st.empty()
        nonan_container = st.empty()
        if make_forecast_future:
            if any(evals.isnull().sum() > 0) or any(future.isnull().sum() > 0):
                # remove no NaN info text
                nonan_container.empty()
                with nan_err_container.container():
                    # find columns with NaN values
                    col_NaN = evals.columns[evals.isnull().sum() > 0]
                    war = st.error(f'Found NaN values in {col_NaN}')
                    
                    clean_method = st.selectbox('Select method to remove NaNs',
                             options = ['None', 'fill with zero', 'fill with adjcent mean'],
                             index = 0)
                    if clean_method == 'fill with zero':
                        evals.fillna(0, inplace=True)
                        future.fillna(0, inplace=True)
                        
                    elif clean_method == 'fill with adjacent mean':
                        for col in col_NaN:
                            evals[col].fillna(0.5*(evals[col].shift() + evals[col].shift(-1)), inplace=True)
                            future[col].fillna(0.5*(future[col].shift() + future[col].shift(-1)), inplace=True)
            else:
                # remove NaN error text
                nan_err_container.empty()
                with nonan_container.container():
                    st.info('Data contains no NaN values.')

        else:
            # no future forecast
            if any(evals.isnull().sum() > 0):
                with nan_err_container.container():
                    # find columns with NaN values
                    col_NaN = evals.columns[evals.isnull().sum() > 0]
                    st.error(f'Found NaN values in {col_NaN}')
                    
                    clean_method = st.selectbox('Select method to remove NaNs',
                             options = ['None', 'fill with zero', 'fill with adjcent mean'],
                             index = 0)
                    if clean_method == 'fill with zero':
                        evals.fillna(0, inplace=True)
                        
                    elif clean_method == 'fill with adjacent mean':
                        for col in col_NaN:
                            evals[col].fillna(0.5*(evals[col].shift() + evals[col].shift(-1)), inplace=True)
            else: 
                # remove NaN error text
                nan_err_container.empty()
                with nonan_container.container():
                    st.info('Data contains no NaN values.')
                
        st.write('Outliers')
        remove_outliers = st.checkbox('Remove outliers', value = False)
        
    
    start_forecast = st.sidebar.checkbox('Launch forecast',
                                 value = False)     
    
    if start_forecast:
        model.fit(evals)
        if make_forecast_future:
            st.dataframe(future)
            forecast = model.predict(future)
        else:
            forecast = model.predict(evals)
        
        
        # plot
        st.header('1. Overview')
        st.plotly_chart(plot_plotly(model, forecast,
                                    uncertainty=True,
                                    changepoints=True
                                    ))
        
        if make_forecast_future:
            df_preds = forecast.set_index('ds').tail(forecast_horizon)
            st.dataframe(forecast.tail(forecast_horizon)['yhat'])
            view_setting = st.selectbox('View sum or mean',
                         options=['sum', 'mean'],
                         index = 0)
            if view_setting =='sum':
                st.markdown('**SUM**: {}'.format(round(sum(df_preds['yhat']), 3)))
            elif view_setting == 'mean':
                st.markdown('**MEAN**: {}'.format(round(np.mean(df_preds['yhat']), 3)))
        
        #st.expander('Plot info'):
        st.header('2. Evaluation and Error analysis')
        
        st.subheader('Global performance')
        mae = round(mean_absolute_error(evals.y, forecast.loc[evals.index,'yhat']), 3)
        mape = round(mean_absolute_percentage_error(evals.y, forecast.loc[evals.index,'yhat']), 3)
        rmse = round(np.sqrt(mean_squared_error(evals.y, forecast.loc[evals.index,'yhat'])), 3)
        
        err1, err2, err3 = st.columns(3)
        with err1:
            st.markdown('**MAE**')
            st.write(mae)
        
        with err2:
            st.markdown('**RMSE**')
            st.write(rmse)
        
        with err3:
            st.markdown('**MAPE**')
            st.write(mape)
            
        st.subheader('Forecast vs Actual')
        truth_vs_forecast = plot_forecast_vs_actual_scatter(evals, forecast)
        st.plotly_chart(truth_vs_forecast)
        
        r2 = round(r2_score(evals.y, forecast.loc[evals.index,'yhat']), 3)
        st.markdown('***R2 error***: {}'.format(r2))
        
        
        st.header('3. Impact of components')
        st.plotly_chart(plot_components_plotly(
            model,
            forecast,
            uncertainty=True))
        
   
    