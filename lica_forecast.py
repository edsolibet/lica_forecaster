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

# custom holidays
# ============================================================================


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



def add_regressors(model, temp_df, future, exogs=None, time_diff=1, regs=None):
    
    m = model
    if regs is not None:
        for reg in regs.keys():
            temp_df.loc[:, reg] = temp_df['ds'].apply(regs[reg])
            future.loc[:, reg] = future['ds'].apply(regs[reg])
            m = m.add_regressor(reg)
            
    if exogs is not None:
        new_end = (pd.to_datetime(train_end) - timedelta(days=time_diff)).strftime('%Y-%m-%d')
        for exog in exogs.columns:
            temp_df.loc[time_diff:, exog] =  exogs.loc[train_start:new_end][exog].values
            #future.loc[time_diff-1:, exog] = traffic_data_.loc[start_train:][exog].values
            future.loc[time_diff:, exog] = exogs.reset_index().iloc[-len(future.loc[time_diff:]):][exog].values
            m = m.add_regressor(exog)
    return m, temp_df.loc[time_diff:], future.loc[time_diff:]

def add_regressors_(model, df, regressors):
    
    m = model
    for reg in regressors.keys():
        df.loc[:, reg] = regressors[reg]
        m = m.add_regressor(reg)
            
    return m, df


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


platform_data = {'Gulong.ph': ('sessions', 'purchases_backend_website'),
                 'Mechanigo.ph': ('sessions', 'bookings_ga')}


if __name__ == '__main__':
    st.title('LICA Target Setting and Forecasting App')
    st.markdown('''
                This tool forecasts the future values of important metrics to be used
                for target setting.
                ''')
    st.sidebar.write('1. Data')
    with st.sidebar.expander('Data selection'):
        platform = st.selectbox('Select platform',
                                        ('Gulong.ph', 'Mechanigo.ph'),
                                        index=0)
        data = get_data(platform)
        # date column
        date_col = st.selectbox('Date column', data.columns[data.dtypes=='datetime64[ns]'],
                                index=0)
        # target column
        param = st.selectbox('Target column:', platform_data[platform],
                                index=0)
        # select data
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
            st.error('Train_end should come after train_start.')
            
        st.write('Validation dataset')
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
    
        # create forecast dataframe
        if pd.Timestamp(val_end) <= data.date.max():
            date_series = make_forecast_dataframe(train_start, val_end)
            param_series = data[data.date.isin(date_series.ds.values)][param].reset_index()
            evals = pd.concat([date_series, param_series], axis=1).rename(columns={0: 'ds',
                                                                                   param:'y'}).drop('index', axis=1)
        else:
            st.error('Train_end is outside available dataset.')
    
        if evals.y.isnull().sum() > 0.5*len(evals):
                st.warning('Evals data contains too many NaN values')
                st.write(evals)
    
    with st.sidebar.expander('Forecast:'):
        make_forecast_future = st.checkbox('Make forecast on future dates')
        if make_forecast_future:
            forecast_horizon = st.number_input('Forecast horizon in days',
                                               min_value = 1,
                                               max_value = 30,
                                               value = 15,
                                               step = 1)
            st.info('Forecast dates: \n {} to {}'.format(val_end+timedelta(days=1), 
                                                   val_end+timedelta(days=forecast_horizon)))

            future = make_forecast_dataframe(start=train_start, end=val_end+timedelta(days=forecast_horizon))
                             
    # MODELLING
    # ========================================================================
    st.sidebar.write('2. Modelling')
    # default parameters for target cols
    default_params = {'sessions':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 15.0,
              'n_changepoints' : 30,
              },
              'purchases_backend_website':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 15.0,
              'n_changepoints' : 30,
              },
              'bookings_ga':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 15.0,
              'n_changepoints' : 30.0,
              }
              }
    
    params = {}
    with st.sidebar.expander('Model and Growth type'):
        '''
        Select type of growth, cap and floor
    
        '''
        
        growth_type = st.selectbox('growth',
                                   options=['logistic', 'linear'],
                                   index = 0)
        
        params['growth'] = growth_type
        if growth_type == 'logistic':
            use_cap = st.checkbox('Add cap value')
            if use_cap:
                cap_type = st.selectbox('Value cap type',
                                        options=['fixed', 'multiplier'])
                if cap_type == 'fixed':
                    cap = st.number_input('Fixed cap value',
                                          min_value = 0,
                                          value = 1000)
                    evals.loc[:, 'cap'] = cap
                    if make_forecast_future:
                        future.loc[:,'cap'] = cap
                elif cap_type == 'multiplier':
                    cap = st.number_input('Cap multiplier',
                                          min_value = 1,
                                          value = 1)
                    evals.loc[:,'cap'] = evals['y']*cap
                    if make_forecast_future:
                        future['cap'] = evals['y']*cap
                
            use_floor = st.checkbox('Add floor value')
            if use_floor:
                floor_type = st.selectbox('Value floor type',
                                        options=['fixed', 'multiplier'])
                if floor_type == 'fixed':
                    floor = st.number_input('Fixed floor value',
                                          min_value = 0,
                                          value = 0)
                    evals.loc[:, 'floor'] = floor
                    if make_forecast_future:
                        future.loc[:,'floor'] = floor
                elif floor_type == 'multiplier':
                    floor = st.number_input('Floor multiplier',
                                          min_value = 0,
                                          value = 0)
                    evals.loc[:,'floor'] = evals['y']*floor
                    if make_forecast_future:
                        future.loc[:,'floor'] = floor
                        
                if evals.y.isnull().sum() > 0.5*len(evals):
                    st.warning('Evals data contains too many NaN values')
                    st.write(evals)
    
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
        changepoint_range = st.number_input('changepoint_prior_scale',
                                    min_value=0.05,
                                    max_value=1.0,
                                    value=0.95,
                                    step=0.05)
        
        
        params['n_changepoints'] = n_changepoints
        params['changepoint_prior_scale'] = changepoint_prior_scale
        params['changepoint_range'] = changepoint_range
        
    
    model = Prophet(**params)  # Input param grid

    # SEASONALITIES
    # ========================================================================
    with st.sidebar.expander('Seasonalities'):
        season_model = st.selectbox('Add Seasonality', 
                            options = ['auto', 'True', 'False'],
                            key = 'season_model')
        
        seasonality_scale_dict = {'sessions': 10,
                                  'purchases_backend_website': 5,
                                  'bookings_ga': 5}
        
        if season_model == 'True':
            model.daily_seasonality = 'auto'
            
            seasonality_mode = st.selectbox('seasonality_mode',
                                        options = ['multiplicative', 'additive'],
                                        index = 1)
            model.seasonality_mode = seasonality_mode
            
            set_seasonality_prior_scale = st.checkbox('Set seasonality_prior_scale')
            if set_seasonality_prior_scale:
                seasonality_prior_scale = st.number_input('overall_seasonality_prior_scale',
                                                      min_value= 1.0,
                                                      max_value= 30.0,
                                                      value=float(seasonality_scale_dict[param]),
                                                      step = 1.0)
            else:
                seasonality_prior_scale = 0
                
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
                model.add_seasonality(name='yearly', 
                                      period = 365,
                                      fourier_order = yearly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else yearly_prior_scale) # add seasonality
            
            monthly_seasonality = st.selectbox('monthly_seasonality', 
                                          ('auto', False, 'custom'))
            if monthly_seasonality == 'custom':
                model.monthly_seasonality = False
                monthly_seasonality_order = st.number_input('monthly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=9,
                                                           step=1)
                if set_seasonality_prior_scale is False:
                    monthly_prior_scale = st.number_input('monthly seasonality prior scale',
                                                           min_value = 1.0,
                                                           max_value=30.0,
                                                           value=8.0,
                                                           step=1.0)
                model.add_seasonality(name='monthly', 
                                      period = 30.4,
                                      fourier_order = monthly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else monthly_prior_scale) # add seasonality
            
            weekly_seasonality = st.selectbox('weekly_seasonality', 
                                          ('auto', False, 'custom'))
            if weekly_seasonality == 'custom':
                model.weekly_seasonality = False
                weekly_seasonality_order = st.number_input('weekly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=3,
                                                           step=1)
                if set_seasonality_prior_scale is False:
                    weekly_prior_scale = st.number_input('weekly seasonality prior scale',
                                                           min_value = 1.0,
                                                           max_value=30.0,
                                                           value=8.0,
                                                           step=1.0)
                model.add_seasonality(name='weekly', 
                                      period = 7,
                                      fourier_order = weekly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else weekly_prior_scale) # add seasonality
            
        elif season_model == 'auto':
            model.yearly_seasonality = 'auto'
            model.monthly_seasonality = 'auto'
            model.weekly_seasonality = 'auto'
            model.daily_seasonality = 'auto'
        
        else:
            model.yearly_seasonality = False
            model.monthly_seasonality = False
            model.weekly_seasonality = False
            model.daily_seasonality = False
    
    # HOLIDAYS
    # =========================================================================
    with st.sidebar.expander('Holidays'):
        holiday_model = st.checkbox('Add Holidays', 
                            value = True,
                            key = 'holiday_model')
        if holiday_model:
            # add holidays
            add_public_holidays = st.checkbox('Public holidays')
            if add_public_holidays:
                model.add_country_holidays(country_name='PH')
            add_set_holidays = st.checkbox('Saved holidays')
            if add_set_holidays:
                fathers_day = pd.DataFrame({
                    'holiday': 'fathers_day',
                    'ds': pd.to_datetime(['2022-06-19']),
                    'lower_window': -21,
                    'upper_window': 3})
                holidays = fathers_day
                model.holidays = holidays
            
            holiday_scale_dict = {'sessions': 3,
                                  'purchases_backend_website': 5,
                                  'bookings_ga': 5}
            
            holiday_scale = st.number_input('holiday_prior_scale',
                                            min_value = 1.0,
                                            max_value = 30.0,
                                            value = float(holiday_scale_dict[param]),
                                            step = 1.0)
            model.holiday_prior_scale = holiday_scale
            
        else:
            model.holidays = None
            model.holiday_prior_scale = 0
    
    # REGRESSORS
    # =========================================================================
    with st.sidebar.expander('Regressors'):
        def is_saturday(ds):
            date = pd.to_datetime(ds)
            return ((date.dayofweek + 1) == 6)*1
        
        def is_sunday(ds):
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
        
        
        add_metrics = st.checkbox('Add data metrics',
                                  value = False)
        exog_num_cols = {'sessions': ['ctr_fb', 'ctr_ga', 'ad_costs_fb_total', 'ad_costs_ga', 
                             'landing_page_views', 'impressions_fb', 'impressions_ga', 'pageviews'],
                             'purchases_backend_website': ['ctr_fb', 'ctr_ga', 'ad_costs_fb_total', 'ad_costs_ga', 
                             'landing_page_views', 'impressions_fb', 'impressions_ga', 'cancellations',
                             'rejections']}
        if add_metrics:
            # Select exogenous variables, including those generated by one hot encoding.
            
            exogs = st.multiselect('Select data metrics',
                           options = exog_num_cols[param],
                           default = exog_num_cols[param])
            
            for exog in exogs:
                evals.loc[:, exog] = data[data.date.isin(evals.ds)][exog].values
                model.add_regressor(exog)
            
            if evals.y.isnull().sum() > 0.5*len(evals):
                st.warning('Evals data contains too many NaN values')
                st.write(evals)
        
        add_gtrends = st.checkbox('google trends',
                                  value = False)
        if add_gtrends:
            kw_list = ['gulong.ph', 'gogulong']
            gtrends_st = st.text_area('Enter google trends keywords',
                                        value = ' '.join(kw_list))
            kw_list = gtrends_st.split(' ')
            gtrends = get_gtrend_data(kw_list, evals)
            for g, gtrend in enumerate(gtrends.columns):
                evals.loc[:,kw_list[g]] = gtrends[gtrend].values
                model.add_regressor(kw_list[g])
            
            if evals.y.isnull().sum() > 0.5*len(evals):
                st.warning('Evals data contains too many NaN values')
                st.write(evals)
        
        add_custom_reg = st.checkbox('Add custom regressors',
                                     value = True)
        if add_custom_reg:
            regs = {'is_saturday': evals.ds.apply(is_saturday),
                        'is_sunday'  : evals.ds.apply(is_sunday)}
            
            regs_list = st.multiselect('Select custom regs',
                           options = list(regs.keys()),
                           default = list(regs.keys()))
            
            for reg in regs_list:
                evals.loc[:, reg] = regs[reg].values
                model.add_regressor(reg)
            
            if make_forecast_future:
                regs_future = {'is_saturday': future.ds.apply(is_saturday),
                               'is_sunday'  : future.ds.apply(is_sunday)}
                
                for reg in regs_list:
                    future.loc[:, reg] = regs_future[reg].values
                
                
            if evals.y.isnull().sum() > 0.5*len(evals):
                st.warning('Evals data contains too many NaN values')
                st.write(evals)
        
        if make_forecast_future:
            # input regressor data for future/forecast dates
            # if selected metrics is not None
            if add_metrics and len(exogs) > 0:
                # provide input field
                for exog in exogs:
                    exog_data = data[data.date.isin(date_series.ds.values)][exog]
                    total = st.number_input('Select metric total over forecast period',
                                           min_value = 0.0, 
                                           max_value = max(exog_data)*1.5,
                                           value = exog_data.tail(forecast_horizon).mean(),
                                           step = 0.01)
                    future.loc[:,exog].iloc[-forecast_horizon:] = np.full((forecast_horizon,), round(total/forecast_horizon, 3))
        
    start_forecast = st.sidebar.checkbox('Launch forecast',
                                 value = False)     
    
    if start_forecast:
        model.fit(evals)
        if make_forecast_future:
            forecast = model.predict(future)
        else:
            forecast = model.predict(evals)
        
        
        # plot
        st.header('Overview')
        st.plotly_chart(plot_plotly(model, forecast,
                                    uncertainty=True,
                                    trend=True,
                                    changepoints=True
                                    ))
        
        #st.expander('Plot info'):
        st.header('Evaluation and Error analysis')
        st.write('Forecast vs Actual')
        truth_vs_forecast = plot_forecast_vs_actual_scatter(evals, forecast)
        st.plotly_chart(truth_vs_forecast)
        
        st.write('Components analysis')
        st.plotly_chart(plot_components_plotly(
            model,
            forecast,
            uncertainty=True))
        
   
    