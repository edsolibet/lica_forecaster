# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 01:10:34 2022

@author: carlo
"""

# Essential libraries
# =============================================================================
import numpy as np
import pandas as pd
from io import BytesIO
import openpyxl, requests
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
from lica_forecast_tooltips import tooltips_text

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
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


def plot_regressors(evals, selected_regressors):
    '''
    Create plotly chart for selected regressors to preview
    
    Parameters
    ----------
    evals: dataframe
        cleaned dataframe containing date and regressors
    selected_regressors: list
        list of selected regressors (strings) to preview
        
    Returns:
    --------
    None. Creates plotly plots on streamlit for large-magnitude and small-magnitude regressors
    
    '''
    def go_scat(name, x, y):
        go_fig = go.Scatter(name = name,
                            x = x,
                            y = y,
                            mode = 'lines+markers',
                            marker = dict(size=6))
        return go_fig
    
    small_mag = [sel for sel in selected_regressors if np.max(evals[sel]) <= 10]
    large_mag = [sel for sel in selected_regressors if sel not in small_mag]
    
    if len(large_mag) > 0:
        with st.expander('Large-magnitude regressors'):
            go_fig_l = [go_scat(lm, x=evals.ds, y=evals[lm]) for lm in large_mag]
            
            fig_l = go.Figure(go_fig_l)
            
            if len(small_mag) == 1:
                ylabel = large_mag[0]
            else:
                ylabel = "y"
            
            fig_l.update_layout(
                yaxis_title= ylabel,
                hovermode="x",
                height = 600,
                width = 1200,
                legend=dict(orientation='h',
                            yanchor='bottom',
                            y=-0.15,
                            xanchor='left',
                            x=0))
            # Change grid color and axis colors
            fig_l.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
            fig_l.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
            st.plotly_chart(fig_l, use_container_width = True)
        
    if len(small_mag) > 0:
        with st.expander('Small-magnitude regressors'):
            go_fig_s = [go_scat(sm, x=evals.ds, y=evals[sm]) for sm in small_mag]
            
            fig_s = go.Figure(go_fig_s)
            
            if len(small_mag) == 1:
                ylabel = small_mag[0]
            else:
                ylabel = "y"
            
            fig_s.update_layout(
                yaxis_title = ylabel,
                hovermode="x",
                height = 600,
                width = 1200,
                legend=dict(orientation='h',
                            yanchor='bottom',
                            y=-0.15,
                            xanchor='left',
                            x=0))
            # Change grid color and axis colors
            fig_s.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
            fig_s.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='#696969')
            st.plotly_chart(fig_s, use_container_width=True)

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

def convert_csv(df):
    # IMPORTANT: Cache the conversion to prevent recomputation on every rerun.
    return df.to_csv().encode('utf-8')



# PROGRAM MAIN FLOW
if __name__ == '__main__':
    st.title('LICA Auto Forecaster')
    st.markdown('''
                This tool creates a trained model that can predict future values and trends of various metrics.
                ''')
    st.sidebar.markdown('# 1. Data')
    with st.sidebar.expander('Data selection'):
        platform = st.selectbox('Select platform',
                                        ('Gulong.ph', 'Mechanigo.ph'),
                                        index=0,
                                        help = tooltips_text['platform_select'])
        data = get_data(platform)
        # date column
        date_col = st.selectbox('Date column', data.columns[data.dtypes=='datetime64[ns]'],
                                index=0,
                                help = tooltips_text['date_column'])
        
        platform_data = {'Gulong.ph': ('sessions', 'purchases_backend_website'),
                 'Mechanigo.ph': ('sessions', 'bookings_ga')}
        # target column
        param = st.selectbox('Target column:', platform_data[platform],
                                index=0,
                                help = tooltips_text['target_column'])
        # select data
        st.markdown('### Training dataset')
        tcol1, tcol2 = st.columns(2)
        date_series = pd.to_datetime(data.loc[:,date_col])
        with tcol1:
            train_start = st.date_input('Training data start date',
                                        value = pd.to_datetime('2022-03-01'),
                                        min_value=date_series.min().date(),
                                        max_value=date_series.max().date(),
                                        help = tooltips_text['training_start'])
        with tcol2:
            train_end = st.date_input('Training data end date',
                                        value = date_series.max().date(),
                                        min_value= train_start + timedelta(days=30),
                                        max_value=date_series.max().date(),
                                        help = tooltips_text['training_end'])
        if train_start >= train_end:
            st.error('Train_end should come after train_start.')
        
        
        # st.markdown('### Validation dataset')
        # vcol1, vcol2 = st.columns(2)
        # with vcol1:
        #     val_start = st.date_input('val_start date',
        #                                 value = train_end + timedelta(days=1),
        #                                 min_value=train_end + timedelta(days=1),
        #                                 max_value=date_series.max().date())
        # with vcol2:
        #     val_end = st.date_input('val_end date',
        #                                 value = date_series.max().date(),
        #                                 min_value= val_start + timedelta(days=1),
        #                                 max_value=date_series.max().date())
        # if val_start >= val_end:
        #     st.error('Val_end should come after val_start.')
    
        # create training dataframe
        # 
        if pd.Timestamp(train_end) <= data.date.max():
            date_series = make_forecast_dataframe(train_start, train_end)
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
        make_forecast_future = st.checkbox('Make forecast on future dates',
                                           value = True,
                                           help = tooltips_text['forecast_checkbox'])
        if make_forecast_future:
            forecast_horizon = st.number_input('Forecast horizon (days)',
                                               min_value = 1,
                                               max_value = 30,
                                               value = 15,
                                               step = 1,
                                               help = tooltips_text['forecast_horizon'])
            
            future = make_forecast_dataframe(start=train_start, 
                                             end=train_end+timedelta(days=forecast_horizon))
            st.info(f'''Forecast dates:\n 
                    {train_end+timedelta(days=1)} to {future.ds.dt.date.max()}''')
                             
    # MODELLING
    # ========================================================================
    st.sidebar.markdown('# 2. Modelling')
    # default parameters for target cols
    default_params = {'sessions':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 8.0,
              'n_changepoints' : 30,
              'cap' : 2500.0,
              },
              'purchases_backend_website':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 8.0,
              'n_changepoints' : 30,
              'cap' : 30.0,
              },
              'bookings_ga':{'growth': 'logistic',
              'seasonality_mode': 'multiplicative',
              'changepoint_prior_scale': 8.0,
              'n_changepoints' : 30,
              'cap' : 30.0,
              }
              }
    
    params = {}
    with st.sidebar.expander('Model and Growth type'):
        
        growth_type = st.selectbox('growth',
                                   options=['logistic', 'linear'],
                                   index = 0,
                                   help = tooltips_text['growth'])
        
        params['growth'] = growth_type
        if growth_type == 'logistic':
            # if logistic growth, cap value is required
            # cap value is round up to nearest 500
            cap = st.number_input('Fixed cap value',
                                  min_value = 0.0,
                                  max_value = None,
                                  value = np.ceil(max(data[param])/500)*500,
                                  step = 0.01,
                                  help = tooltips_text['cap_value'])
            evals.loc[:, 'cap'] = cap
            # if forecast future, also apply cap to future df
            if make_forecast_future:
                future.loc[:,'cap'] = cap
            
            # floor is optional
            use_floor = st.checkbox('Add floor value')
            if use_floor:
                floor = st.number_input('Fixed floor value',
                                      min_value = 0.0,
                                      max_value = None,
                                      value = 0.0,
                                      step = 0.01,
                                      help = tooltips_text['floor_value'])
                evals.loc[:, 'floor'] = floor
                if make_forecast_future:
                    future.loc[:,'floor'] = floor
            
                # check viability of cap value
                if cap <= floor:
                    st.error('Cap value should be greater than floor value')
    
    # CHANGEPOINTS
    # =========================================================================
    with st.sidebar.expander('Changepoints'):
        
        changepoint_select = st.selectbox('Changepoint selection',
                                          options=['Auto', 'Manual'],
                                          index=0,
                                          help = tooltips_text['changepoint_select'])
        
        if changepoint_select == 'Auto':
            n_changepoints = st.slider('Number of changepoints',
                                       min_value = 5,
                                       max_value = 100,
                                       value = default_params[param]['n_changepoints'],
                                       step = 5,
                                       help = tooltips_text['n_changepoints'])
            
            params['n_changepoints'] = n_changepoints
            params.pop('changepoints', None)
            
        elif changepoint_select == 'Manual':
            changepoints = st.multiselect('Select dates to place changepoints',
                                          options = evals.ds.dt.date.tolist(),
                                          default = [evals.ds.dt.date.min(), evals.ds.dt.date.max()],
                                          help = tooltips_text['changepoints'])
            params['changepoints'] = changepoints
            params.pop('n_changepoints', None)
            
        changepoint_prior_scale = st.number_input('changepoint_prior_scale',
                                    min_value=0.05,
                                    max_value=50.0,
                                    value= float(default_params[param]['changepoint_prior_scale']),
                                    step=0.05,
                                    help = tooltips_text['changepoint_prior_scale'])
        changepoint_range = st.number_input('changepoint_range',
                                    min_value=0.05,
                                    max_value=1.0,
                                    value=0.9,
                                    step=0.05,
                                    help = tooltips_text['changepoint_range'])
        
        # add selected inputs to params dict
        
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
                            key = 'season_model',
                            help = tooltips_text['add_seasonality'])
        
        seasonality_scale_dict = {'sessions': 6,
                                  'purchases_backend_website': 3,
                                  'bookings_ga': 3}
        
        if season_model == 'True':
            model.daily_seasonality = 'auto'
            
            seasonality_mode = st.selectbox('seasonality_mode',
                                        options = ['multiplicative', 'additive'],
                                        index = 1,
                                        help = tooltips_text['seasonality_mode'])
            model.seasonality_mode = seasonality_mode
            
            set_seasonality_prior_scale = st.checkbox('Set seasonality_prior_scale',
                                                      value = True,
                                                      help=tooltips_text['set_overall_seasonality_prior_scale'])
            # add info
            if set_seasonality_prior_scale:
                seasonality_prior_scale = st.number_input('overall_seasonality_prior_scale',
                                                      min_value= 0.01,
                                                      max_value= 50.0,
                                                      value=float(seasonality_scale_dict[param]),
                                                      step = 0.1,
                                                      help = tooltips_text['overall_seasonality_prior_scale'])
            else:
                seasonality_prior_scale = 1
                
            model.seasonality_prior_scale = seasonality_prior_scale
            
            yearly_seasonality = st.selectbox('yearly_seasonality', 
                                          ('auto', False, 'Custom'),
                                          help = tooltips_text['add_yearly_seasonality'])
            if yearly_seasonality == 'Custom':
                model.yearly_seasonality = False
                yearly_seasonality_order = st.number_input('Yearly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=5,
                                                           step=1,
                                                           help = tooltips_text['yearly_order'])
                if set_seasonality_prior_scale is False:
                    yearly_prior_scale = st.number_input('Yearly seasonality prior scale',
                                                           min_value = 0.05,
                                                           max_value=50.0,
                                                           value=6.0,
                                                           step= 0.1,
                                                           help = tooltips_text['yearly_prior_scale'])
                # add yearly seasonality to model
                model.add_seasonality(name='yearly', 
                                      period = 365,
                                      fourier_order = yearly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else yearly_prior_scale) # add seasonality
            
            monthly_seasonality = st.selectbox('monthly_seasonality', 
                                          options = ('Auto', False, 'Custom'),
                                          index = 2,
                                          help = tooltips_text['add_monthly_seasonality'])
            if monthly_seasonality == 'Custom':
                model.monthly_seasonality = False
                monthly_seasonality_order = st.number_input('Monthly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=9,
                                                           step=1,
                                                           help = tooltips_text['monthly_order'])
                if set_seasonality_prior_scale is False:
                    monthly_prior_scale = st.number_input('Monthly seasonality prior scale',
                                                           min_value = 0.05,
                                                           max_value=50.0,
                                                           value=6.0,
                                                           step= 0.1,
                                                           help = tooltips_text['monthly_prior_scale'])
                # add monthly seasonality to model
                model.add_seasonality(name='monthly', 
                                      period = 30.4,
                                      fourier_order = monthly_seasonality_order,
                                      prior_scale = seasonality_prior_scale if set_seasonality_prior_scale else monthly_prior_scale) # add seasonality
            
            weekly_seasonality = st.selectbox('weekly_seasonality', 
                                          options = ('Auto', False, 'Custom'),
                                          index = 2,
                                          help = tooltips_text['add_weekly_seasonality'])
            if weekly_seasonality == 'Custom':
                model.weekly_seasonality = False
                weekly_seasonality_order = st.number_input('Weekly seasonality order',
                                                           min_value = 1,
                                                           max_value=30,
                                                           value=6,
                                                           step=1,
                                                           help = tooltips_text['weekly_order'])
                if set_seasonality_prior_scale is False:
                    weekly_prior_scale = st.number_input('Weekly seasonality prior scale',
                                                           min_value = 0.05,
                                                           max_value=50.0,
                                                           value=6.0,
                                                           step= 0.1,
                                                           help = tooltips_text['weekly_prior_scale'])
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
        add_holidays = st.checkbox('Add holidays', 
                            value = True,
                            key = 'holiday_model',
                            help = tooltips_text['add_holidays'])
        if add_holidays:
            # add public holidays
            add_public_holidays = st.checkbox('Public holidays',
                                              value = True,
                                              help = tooltips_text['add_public_holidays'])
            if add_public_holidays:
                model.add_country_holidays(country_name='PH')
            
            # add set_holidays
            add_set_holidays = st.checkbox('Saved holidays',
                                           value = True,
                                           help = tooltips_text['add_saved_holidays'])
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
                
                model.holidays = pd.concat([holidays_set[h] for h in selected_holidays])
            
            holiday_scale_dict = {'sessions': 3,
                                  'purchases_backend_website': 3,
                                  'bookings_ga': 3}
            
            holiday_scale = st.number_input('holiday_prior_scale',
                                            min_value = 1.0,
                                            max_value = 30.0,
                                            value = float(holiday_scale_dict[param]),
                                            step = 1.0,
                                            help = tooltips_text['holiday_prior_scale'])
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
        
        @st.experimental_memo
        def get_gtrend_data(kw_list, df):
            '''
            Get google trend data for specifie keywords
            '''
            pytrend = TrendReq()
            start = pd.to_datetime(df.ds.min())
            end = pd.to_datetime(df.ds.max())
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
            historicaldf_grp = historicaldf[kw_list].groupby(historicaldf.index.date).mean()
            return historicaldf_grp.fillna(0).asfreq('1D').reset_index()
        
        # add data metrics option
        add_metrics = st.checkbox('Add data metrics',
                                  value = True,
                                  help = tooltips_text['add_metrics'])
        
        exog_num_cols = {'sessions': ['ctr_fb', 'ctr_ga', 'ctr_total', 'ad_costs_fb_total', 'ad_costs_ga', 'ad_costs_total',
                             'landing_page_views', 'impressions_fb', 'impressions_ga', 'pageviews'],
                         'purchases_backend_website': ['ctr_fb', 'ctr_ga', 'ctr_total', 'ad_costs_fb_total', 'ad_costs_ga', 'ad_costs_total',
                             'landing_page_views', 'impressions_fb', 'impressions_ga', 'cancellations',
                             'rejections'],
                         'bookings_ga': ['ctr_ga', 'ad_costs_ga', 'impressions_ga']}
        
        regressors = list()
        metrics_container = st.empty()
        if add_metrics:
            # Select traffic metrics available from data.
            
            with metrics_container.container():
                exogs = st.multiselect('Select data metrics',
                               options = exog_num_cols[param],
                               default = ['ctr_total', 'ad_costs_total'],
                               help = tooltips_text['add_metrics_select'])
                
                # add selected exogenous variables to list of regressors
                regressors.extend(exogs)
                for exog in exogs:
                    evals.loc[:, exog] = data[data.date.isin(evals.ds)][exog].values
                    model.add_regressor(exog)
                
                    
                    # if forecast future
                    if make_forecast_future:
                        future.loc[future.ds.isin(evals.ds), exog] = data[data.date.isin(evals.ds)][exog].values
        
        # gtrends
        add_gtrends = st.checkbox('Add Google trends',
                              value = False,
                              help = tooltips_text['add_google_trends'])

        gtrends_container = st.empty()
        if add_gtrends:
            # keywords
            with gtrends_container.container():
                kw_list = ['gulong.ph', 'gogulong']
                gtrends_st = st.text_area('Enter google trends keywords',
                                            value = ' '.join(kw_list),
                                            help = tooltips_text['gtrend_kw'])
                # selected keywords
                kw_list = gtrends_st.split(' ')
                # cannot generate data for dates in forecast horizon
                gtrends = get_gtrend_data(kw_list, evals)
                for g, gtrend in enumerate(gtrends.columns[1:]):
                    evals.loc[:,gtrend] = gtrends[gtrends['index'].isin(evals.ds)][gtrend]
                    future.loc[future.ds.isin(evals.ds), gtrend] = gtrends[gtrends['index'].isin(evals.ds)][gtrend]
                    model.add_regressor(gtrend)
                    regressors.append(gtrend)
        
        if make_forecast_future:
            # input regressor data for future/forecast dates
            # if selected metrics is not None
            regressor_input = st.empty()
            if add_metrics and len(regressors) > 0:
                # provide input field
                with regressor_input.container():
                    for regressor in regressors:
                        if regressor in data.columns:
                            exog_data = data[data.date.isin(date_series.ds.values)][regressor]
                        else:
                            exog_data = gtrends[regressor]
                        # added key to solve DuplicateWidgetID
                        data_input = st.selectbox(regressor + ' data input type:',
                                             options=['total', 'average'],
                                             index=1,
                                             key = regressor + '_input',
                                             help = tooltips_text['data_input_type'])
                        
                        if data_input == 'total':
                            # if data input is total
                            total = st.number_input('Select {} total over forecast period'.format(regressor),
                                                   min_value = 0.0, 
                                                   value = float(np.nansum(exog_data[-int(forecast_horizon):])),
                                                   step = 0.01,
                                                   help = tooltips_text['data_input_total'])
                            st.write()
                            future.loc[future.index[-int(forecast_horizon):],regressor] = np.full((int(forecast_horizon),), round(total/forecast_horizon, 3))
                        else:
                            # if data input is average
                            average = st.number_input('Select {} average over forecast period'.format(regressor),
                                                   min_value = 0.00, 
                                                   value = np.nanmean(exog_data[-int(forecast_horizon):]),
                                                   step = 0.010,
                                                   help = tooltips_text['data_input_average'])
                            future.loc[future.index[-int(forecast_horizon):],regressor] = np.full((int(forecast_horizon),), round(average, 3))
            else:
                # delete unused fields
                regressor_input.empty()
            
            
            
            # custom regressors (functions applied to dates)
            add_custom_reg = st.checkbox('Add custom regressors',
                                         value = True,
                                         help = tooltips_text['add_custom_regressors'])
            
            custom_reg_container = st.empty()
            if add_custom_reg:
                
                with custom_reg_container.container():
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
            else:
                custom_reg_container.empty()
                
                
    with st.sidebar.expander('Cleaning'):
        st.write('Missing values')
        nan_err_container = st.empty()
        nonan_container = st.empty()
        if make_forecast_future:
            if any(evals.isnull().sum() > 0) or any(future.isnull().sum() > 0):
                # remove no NaN info text
                nonan_container.empty()
                col_NaN = evals.columns[evals.isnull().sum() > 0]
                with nan_err_container.container():
                    # find columns with NaN values
                    war = st.error(f'Found NaN values in {list(col_NaN)}')
                
                
                clean_method = st.selectbox('Select method to remove NaNs',
                         options = ['fill with zero', 'fill with adjcent mean'],
                         index = 0,
                         help = tooltips_text['nan_clean_method'])
                if clean_method == 'fill with zero':
                    evals = evals.fillna(0)
                    future = future.fillna(0)
                    
                elif clean_method == 'fill with adjacent mean':
                    for col in col_NaN:
                        evals.loc[:, col] = evals.loc[:, col].fillna(0.5*(evals.loc[:, col].ffill() + evals.loc[:, col].bfill()))
                        future.loc[:, col] = future.loc[:, col].fillna(0.5*(future.loc[:, col].ffill() + future.loc[:, col].bfill()))
            
                if all(evals.isnull().sum() == 0):
                    # remove NaN error text
                    nan_err_container.empty()
                    with nonan_container.container():
                        st.info('Data contains no NaN values.')
            
            else:
                # remove NaN error text
                nan_err_container.empty()
                with nonan_container.container():
                    st.info('Data contains no NaN values.')

        else:
            # no future forecast
            if any(evals.isnull().sum() > 0):
                col_NaN = evals.columns[evals.isnull().sum() > 0] 
                with nan_err_container.container():
                    # find columns with NaN values
                    st.error(f'Found NaN values in {list(col_NaN)}')
                
                clean_method = st.selectbox('Select method to remove NaNs',
                         options = ['fill with zero', 'fill with adjcent mean'],
                         index = 0,
                         help = tooltips_text['nan_clean_method'])
                
                if clean_method == 'fill with zero':
                    evals = evals.fillna(0)
                    
                elif clean_method == 'fill with adjacent mean':
                    for col in col_NaN:
                        if pd.isna(evals.loc[0, col]) or pd.isna(evals.loc[len(evals)-1, col]):   
                            evals.loc[:, col] = evals.loc[:, col].fillna(0.5*(evals.loc[:, col].ffill() + evals.loc[:, col].bfill()))
                            # if first or last value is NaN
                            evals.loc[:, col] = evals.loc[:, col].bfill().ffill()
                        else:
                            evals.loc[:, col] = evals.loc[:, col].fillna(0.5*(evals.loc[:, col].ffill() + evals.loc[:, col].bfill()))
                
                if all(evals.isnull().sum() == 0):
                    # remove NaN error text
                    nan_err_container.empty()
                    with nonan_container.container():
                        st.info('Data contains no NaN values.')
            
            else: 
                # remove NaN error text
                nan_err_container.empty()
                with nonan_container.container():
                    st.info('Data contains no NaN values.')
            
        
        st.write('Outliers')
        remove_outliers = st.checkbox('Remove outliers', value = False,
                                      help = tooltips_text['outliers'])
        if remove_outliers:
            # option to remove datapoints with value = 0
            remove_zeros = st.checkbox('Remove zero datapoints', 
                                       value = False)
            if remove_zeros:
                evals = evals[evals.y != 0]
                
            
            method = st.selectbox('Choose method',
                         options=['None', 'KNN', 'LOF', 'Isolation Forest'],
                         index = 0,
                         help = tooltips_text['remove_outlier_method'])
            
            
            outliers_df = evals['y'].to_frame()
            if method == 'KNN':
                neighbors = st.number_input('Enter number of neighbors',
                                             min_value = 2,
                                             max_value = 20,
                                             value = 5,
                                             step = 1,
                                             help = tooltips_text['KNN_neighbors'])
                
                clf = KNeighborsClassifier(n_neighbors = int(neighbors)).fit(np.arange(len(outliers_df)).reshape(-1,1),
                                                                        outliers_df['y'].array.reshape(-1,1))
                outliers_df.loc[:,'label'] = clf.predict(outliers_df['y'].array.reshape(-1,1))
                evals = evals[(outliers_df.loc[:,'label'] == 0)]
                
            
            elif method == 'LOF':
                pass
            
            elif method == 'Isolation Forest':
                # Isolation Forest
                estimators = st.number_input('Enter number of estimators',
                                             min_value = 20,
                                             value = 100,
                                             step = 5,
                                             help = tooltips_text['IF_estimators'])
                
                max_samples = st.number_input('Enter max_samples',
                                              min_value = 15,
                                              value = 60,
                                              step = 15,
                                              help = tooltips_text['IF_max_samples'])
            
                clf = IsolationForest(n_estimators=estimators,
                                      max_samples=max_samples,
                                      random_state=101).fit(outliers_df['y'].array.reshape(-1,1))
            
                outliers_df.loc[:,'label'] = clf.predict(outliers_df['y'].array.reshape(-1,1))
                evals = evals[outliers_df.loc[:,'label'] == 1]
                
            else:
                pass
                
        st.write('Transformation')
        transform = st.selectbox('Data transformation method',
                     options = ['None', 'Moving average', 'Logarithm'],
                     index = 0,
                     help = tooltips_text['transform'])
        
        if transform == 'Moving average':
            window = st.slider('Window', 
                               min_value = 1, 
                               max_value = 20, 
                               value = 4, 
                               step = 1,
                               help = tooltips_text['window'])
            
            evals['y'] = evals['y'].rolling(window = window, min_periods = 1).mean().bfill()
        
        if transform == 'Logarithm':
            evals['y'] = evals['y'].apply(np.log)
            
    start_forecast = st.sidebar.checkbox('Launch forecast',
                                 value = False)     
    
    
    if len(regressors) > 0:
        st.subheader('Regressor preview')
        selected_reg = st.multiselect('Select regressors to show',
                                      options = regressors,
                                      default = regressors)
        # plotly scatter plot of regressors
        plot_regressors(evals, selected_reg)
    
    if start_forecast:
        st.dataframe(future)
        model.fit(evals)
        if make_forecast_future:
            #st.dataframe(future)
            forecast = model.predict(future)
        else:
            #st.dataframe(evals)
            forecast = model.predict(evals)
        

        # plot
        st.header('Overview')
        st.plotly_chart(plot_plotly(model, forecast,
                                    uncertainty=True,
                                    changepoints=True,
                                    ylabel = param,
                                    xlabel = 'date',
                                    figsize=(800, 600)))
        
        if make_forecast_future:
            # get forecasted values
            df_preds = forecast.tail(forecast_horizon)
            df_preds.loc[:, 'ds'] = pd.to_datetime(df_preds.loc[:, 'ds'], unit='D').dt.strftime('%Y-%m-%d')
            df_preds = df_preds.set_index('ds')
            
            # display results
            st.subheader('Forecast results')
            fcast_col1, fcast_col2 = st.columns([2, 1])
            with fcast_col1:
                
                cols = ['yhat', 'yhat_lower', 'yhat_upper']
                cols.extend(regressors)
                st.dataframe(df_preds[cols])
                
                st.download_button(label='Export forecast results',
                                   data = convert_csv(df_preds[['yhat', 'yhat_lower', 'yhat_upper']]),
                                   file_name = param +'_forecast_results.csv')
                
            with fcast_col2:    
                view_setting = st.selectbox('View sum or mean',
                             options=['sum', 'mean'],
                             index = 0)
                if view_setting =='sum':
                    st.dataframe(df_preds[['yhat', 'yhat_lower', 'yhat_upper']].sum().rename('total_' + param))
                elif view_setting == 'mean':    
                    st.dataframe(df_preds[['yhat', 'yhat_lower', 'yhat_upper']].mean().rename('average_' + param))
            
                
                
            
        #st.expander('Plot info'):
        st.header('Evaluation and Error analysis')
        
        st.subheader('Global performance')
        mae = round(mean_absolute_error(evals.y, forecast.loc[evals.index,'yhat']), 3)
        mape = round(mean_absolute_percentage_error(evals.y, forecast.loc[evals.index,'yhat'])*100, 3)
        rmse = round(np.sqrt(mean_squared_error(evals.y, forecast.loc[evals.index,'yhat'])), 3)
        
        err1, err2, err3 = st.columns(3)
        with err1:
            st.metric('MAE', 
                      value = mae, 
                      help = tooltips_text['mae'])
        
        with err2:
            st.metric('RMSE', 
                      value = rmse, 
                      help = tooltips_text['rmse'])
        
        with err3:
            st.metric('MAPE', 
                      value = round(mape, 3), 
                      help = tooltips_text['mape'])
            
        st.subheader('Forecast vs Actual')
        truth_vs_forecast = plot_forecast_vs_actual_scatter(evals, forecast)
        st.plotly_chart(truth_vs_forecast)
        
        r2 = round(r2_score(evals.y, forecast.loc[evals.index,'yhat']), 3)
        st.markdown('**<p style="font-size: 20px">R<sup>2</sup> error**: {} </p>'.format(r2), unsafe_allow_html = True)
        with st.expander('Pearson correlation coefficient'):
            st.markdown(tooltips_text['pearson_coeff'], unsafe_allow_html = True)
        
        
        st.header('Impact of components')
        st.plotly_chart(plot_components_plotly(
            model,
            forecast,
            uncertainty=True))
        
        st.write(regressor_coefficients(model))
   
    