# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 13:23:30 2024

@author: vitto
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch.univariate import ARX, GARCH, ZeroMean, arch_model


#create a dataset with the value per month of the Climate policy uncertainty Index 
data_cpu=pd.read_csv() # Climate policy uncertainty index
print (data_cpu)
#create a dataset with the daily values of the VIX Index
data_vix=pd.read_csv() #VIX index data
print (data_vix)
data_news=pd.read_csv() #U.S Daily News Index data
print (data_news)
#create a dataset with the monthly average value of the VIX Index
data_vix.index = pd.to_datetime(data_vix.index)
data_vix_monthly = data_vix.resample('M').mean()
#select datas within the timeframe available for both indexes
data_vix_monthly.drop(data_vix_monthly.tail(4).index,inplace = True)
data_cpu.drop(data_cpu.head(33).index,inplace=True)
data_news.drop(data_news.head(1827).index,inplace = True)
data_news.drop(data_news.tail(6).index,inplace = True)
data_news_daily = data_news.iloc[:,3]
data_news_daily.index= pd.date_range(start = '01-02-1990', end= '09-30-2022')
data_news_daily= data_news_daily[data_news_daily.index.isin(data_vix.index)]
data_vix.drop(data_vix.tail(21).index,inplace = True)

######### Linear regression CPU-VIX
x_cpu = sm.add_constant(data_cpu)
x_cpu = x_cpu.reset_index(drop = True)
data_cpu_noindex= data_cpu.reset_index(drop = True)
y_monthly = data_vix_monthly.reset_index(drop = True)
model2=sm.OLS(y_monthly,x_cpu)
results= model2.fit()
print(results.summary())
########## SARIMAX test with extrogenous variable CPU-VIX
# fit model
y_monthly_log = np.log(y_monthly)
y_monthly_log_diff = y_monthly_log.diff()
data = y_monthly_log_diff.drop(y_monthly_log_diff.index[0])
# check the pacf and acf plots
plt.figure(figsize=[15, 7.5]); # Set dimensions for figure
plt.plot(y_monthly_log_diff)
plt.title("Log Difference of VIX Index")
plt.show()
plot_pacf(y_monthly_log)
plot_acf(y_monthly_log)
#function to try and find the best parameters for the SARIMA model
def optimize_SARIMA(parameters_list, d, D, s, exog):
    """
        Return dataframe with parameters, corresponding AIC and SSE
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
        exog - the exogenous variable
    """
    results = []
    for param in tqdm_notebook(parameters_list):
        try: 
            model = SARIMAX(exog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
        except:
            continue
        aic = model.aic
        results.append([param, aic])
        
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)x(P,Q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    return result_df
p = range(0, 4, 1)
d = 2
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = 0
parameters = product(p, q, P, Q)
parameters_list = list(parameters)
print(len(parameters_list))
optimize_SARIMA(parameters_list,1,4,1,y_monthly_log_diff)
# we find that the best ARIMA value based on AIC is ARIMA(1,4,1)
model = SARIMAX(y_monthly_log_diff, exog=x_cpu, order=(1, 4, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
print(model_fit.summary())
data_news=pd.read_csv(r'C:\Users\User\Desktop\Greening Energy Market and Finance\CLIMATE-RELATED RISK AND COMMODITY MARKET (I.C.)\Assignment - Group work\data_news.csv')
print (data_news)
######### Linear regression Daily-VIX
x_Daily= sm.add_constant(data_news_daily)
x_Daily = x.reset_index(drop = True)
data_news_noindex= data_news_daily.reset_index(drop = True)
y_daily = data_vix.reset_index(drop = True)
y_daily_log = np.log(y)
y_daily_log_diff = y_daily_log.diff()
data = y_log_diff.drop(y_log_diff.index[0])
model2=sm.OLS(y,x)
results= model2.fit()
print(results.summary())
p = range(0, 4, 1)
d = 2
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = 0
parameters = product(p, q, P, Q)
parameters_list = list(parameters)
print(len(parameters_list))
optimize_SARIMA(parameters_list,1,1,2,y)
model = SARIMAX(y_daily_log_diff, exog=x_Daily, order=(1, 4, 1), seasonal_order=(0, 0, 0, 0))
model_fit = model.fit(disp=False)
print(model_fit.summary())
# GARCH MODELS
#GARCH model bt\w VIX and CPU
mod = arch_model(data_vix_monthly,x=data_cpu , mean="ARX", lags=1, power=2)
res = mod.fit(disp="off")
print(res.summary())
#Garch model bt\w VIX and Daily News Index
mod_2 = arch_model(data_vix,x=data_news_daily , mean="ARX", lags=1, power=2)
res_2 = mod_2.fit(disp="off")
print(res_2.summary())