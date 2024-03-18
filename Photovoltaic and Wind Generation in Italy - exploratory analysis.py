
from meteostat import Stations, Daily
from datetime import datetime
import pandas as pd
from sqlalchemy.engine import URL
from sqlalchemy import create_engine
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from sklearn.model_selection import train_test_split


# Update the locations dictionary with the additional cities
locations = {
    'Milan': (45.4642, 9.1900),
    'Rome': (41.9028, 12.4964),
    'Naples': (40.8518, 14.2681),
    'Florence': (43.7696, 11.2558),
    'Palermo': (38.1157, 13.3615),
    'Bari': (41.1171, 16.8719),
    'Turin': (45.0703, 7.6869),
    'Venice': (45.4408, 12.3155),
    'Udine': (46.0626, 13.2378),
    'Bologna': (44.4949, 11.3426),
    'Cagliari': (39.2238, 9.1217),
    'Trento': (46.0731, 11.1211),
    'Reggio Calabria': (38.1105, 15.6613),
    'Messina': (38.1938, 15.5540),
    'Catania': (37.5079, 15.0830),
    'Sassari': (40.7259, 8.5555),
    'Ancona': (43.6158, 13.5189),
    'Reggio Emilia': (44.6983, 10.6312),
    'Verona': (45.4384, 10.9916)
}

# Define the time period for the analysis
start = datetime(2019, 1, 2)
end = datetime(2024, 3, 6)

# Initialize an empty DataFrame for aggregated data
aggregated_data = pd.DataFrame()

for city, (lat, lon) in locations.items():
    # Locate the nearest weather station for each city
    stations = Stations().nearby(lat, lon)
    station = stations.fetch(1)  # Fetch the top result
    
    # Retrieve daily weather data for the specified period
    data = Daily(station, start, end).fetch()
    
    # Aggregate the data from all locations
    if aggregated_data.empty:
        aggregated_data = data
    else:
        aggregated_data = aggregated_data.add(data, fill_value=0)

# Calculate the mean to get a country-wide approximation
mean_data = aggregated_data / len(locations)

print(mean_data)

DATABASE = 'nudsata'
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER=SQLNUS;DATABASE=nusdata;Trusted_Connection=yes'

connection_url = URL.create("mssql+pyodbc", query={"odbc_connect": conn_str})
engine = create_engine(connection_url)



def execute_query(query):
    """
    Executes the given SQL query and returns a DataFrame with the results.
    """
    with engine.begin() as conn:
        df = pd.read_sql_query(query, conn)
    return df

query = "SELECT * FROM ELETTRICO.Renewable_Generation"  # Adjust your query accordingly
df = execute_query(query)

#The df contains the renewable Production data from TERNA 


df.sort_values('Date', inplace=True)

df['Date'] = pd.to_datetime(df['Date']).dt.date

# Group by 'Date' and 'Energy_Source', then sum 'Renewable_Generation_GWh'
daily_data = df.groupby(['Date', 'Energy_Source']).sum().reset_index()
energy_pivoted = daily_data.pivot_table(index='Date', columns='Energy_Source', values='Renewable_Generation_GWh', aggfunc='sum').reset_index()
energy_pivoted= energy_pivoted.dropna(axis=1)

energy_pivoted['Date'] = pd.to_datetime(energy_pivoted['Date'])

mean_data['Date'] = pd.to_datetime(mean_data.index)
weather_data = mean_data.reset_index()
# Merge with weather data
combined_data = pd.merge(energy_pivoted, weather_data, on='Date', how='inner')

for column in combined_data.select_dtypes(include=['float64', 'int64']).columns:
    combined_data[column].fillna(combined_data[column].mean(), inplace=True)

############# ANALYSIS  ###############



# Plotting energy production against average temperature
plt.figure(figsize=(12, 6))
sns.scatterplot(x='tavg', y='Photovoltaic', data=combined_data)
plt.title('Photovoltaic Energy Production vs. Average Temperature')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Photovoltaic Energy Production (GWh)')
plt.show()


# Check correlation between weather conditions and energy production
energy_sources = ['Biomass', 'Geothermal', 'Hydro', 'Photovoltaic', 'Wind']
weather_variables = ['tavg', 'tmin', 'tmax', 'prcp', 'snow', 'wdir', 'wspd', 'wpgt', 'pres', 'tsun']

# Create a new DataFrame with just the variables of interest
focused_data = combined_data[energy_sources + weather_variables]

# Calculate the correlation matrix for the focused data
focused_corr_matrix = focused_data.corr()

# Filter the correlation matrix to show only correlations between weather variables and energy sources
energy_weather_corr = focused_corr_matrix.loc[weather_variables, energy_sources]

print(energy_weather_corr)

correlation_matrix = combined_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(energy_weather_corr, annot=True, cmap='coolwarm')
plt.title('Correlation between Weather Conditions and Energy Production')
plt.show()



################## WIND ###################
combined_data.index = pd.to_datetime(combined_data['Date'])

# Now create the wind_series with the correct index
wind_series = combined_data['Wind'].copy()

# Coerce non-numeric data to NaN and forward-fill any missing values
wind_series = pd.to_numeric(wind_series, errors='coerce').fillna(method='ffill')

# Check the wind_series
print(wind_series.head())

# If everything looks correct, set the frequency
wind_series = wind_series.asfreq('D')
# Set the frequency to daily
wind_series = wind_series.asfreq('D')

# Now let's try the seasonal decomposition again

decomposition = seasonal_decompose(wind_series.dropna(), model='additive', period=365)  # Assuming daily data and a yearly cycle

# Assuming decomposition is already done
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), dpi=100)

# Trend Component
axes[0].plot(trend.index, trend, label='Trend', color='tab:blue')
axes[0].set_ylabel('Trend')
axes[0].legend(loc='upper left')
axes[0].set_title('Seasonal Decomposition')
# Formatting the dates on the x-axis to show months


# Seasonal Component
axes[1].plot(seasonal.index, seasonal, label='Seasonality', color='tab:orange')
axes[1].set_ylabel('Seasonality')
axes[1].legend(loc='upper left')
# Formatting the dates on the x-axis to show months


# Residual Component
axes[2].plot(residual.index, residual, label='Residuals', color='tab:green')
axes[2].set_ylabel('Residuals')
axes[2].legend(loc='upper left')
# Formatting the dates on the x-axis to show months
axes[2].xaxis.set_major_locator(mdates.MonthLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Improve spacing and layout
plt.tight_layout()

# Automatically rotate the dates
for ax in axes:
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')

plt.show()


# Automatically selecting SARIMA parameters
smodel = pm.auto_arima(wind_series.dropna(), seasonal=True, m=12, stepwise=True,
                       suppress_warnings=True, error_action="ignore", max_p=3, max_q=3, max_order=5)

smodel.summary()



# Group by day of the year and calculate max/min
daily_max = wind_series.groupby([wind_series.index.month, wind_series.index.day]).max()
daily_min = wind_series.groupby([wind_series.index.month, wind_series.index.day]).min()

import matplotlib.dates as mdates

# Create a figure and plot
plt.figure(figsize=(12, 6))

# Create a range of dates for the current year for plotting
date_range = pd.date_range(start=f'{wind_series.index.year[-1]}-01-01', end=f'{wind_series.index.year[-1]}-12-31')

# Ensure the date_range doesn't exceed the wind_series index
date_range = date_range[date_range <= wind_series.index[-1]]

# Plotting the band
plt.fill_between(date_range, daily_min.values[:len(date_range)], daily_max.values[:len(date_range)], color='gray', alpha=0.2, label='Historical Max/Min Band')

# Plotting the current year's data
current_year_data = wind_series[wind_series.index.year == wind_series.index.year[-1]]
plt.plot(current_year_data.index, current_year_data, label='Current Year Wind Production', color='blue')

plt.title('Daily Wind Energy Production: Current Year vs Historical Max/Min (GWh)')
plt.xlabel('Date')
plt.ylabel('Wind Energy Production')
plt.legend()

# Improve formatting of the x-axis to show months
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.xticks(rotation=45)
# Highlight the last available day
plt.scatter(current_year_data.index[-1], current_year_data.iloc[-1], color='red', zorder=5)
plt.annotate(f'Last Day: {current_year_data.iloc[-1]:.2f}', 
             (current_year_data.index[-1], current_year_data.iloc[-1]), 
             textcoords="offset points", xytext=(-50,10), ha='center', color='red')

plt.show()



features = ['wpgt', 'wspd', 'tavg', 'prcp', 'wdir']  # Select based on your EDA
X = combined_data[features]
y = combined_data['Wind']  # Target variable

# Splitting the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mean_values_X_train = X_train.mean()
# If y_train is continuous and has missing values
mean_y_train = y_train.mean()
# Apply imputation to training data
X_train = X_train.fillna(mean_values_X_train)
# Apply the same means to test data
X_test= X_test.fillna(mean_values_X_train)
y_train= y_train.fillna(mean_y_train)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'MSE: {mean_squared_error(y_test, y_pred)}')


# Assuming y_test are your actual values and y_pred are the predictions from your model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted Values (GWh)')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # Diagonal line
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residuals vs. Predicted Values (GWh)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
plt.show()



feature_importance = model.coef_
features = X_train.columns  # Make sure this matches your preprocessed feature set

# Sort features by importance
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features[sorted_idx])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

correlation_matrix = combined_data.corr()

# Extracting the 'Wind' column correlations, sorting them, and removing the self-correlation
wind_correlations = correlation_matrix['Wind'].sort_values(ascending=False).drop('Wind')

# Plotting
plt.figure(figsize=(10, 8))
sns.barplot(x=wind_correlations.values, y=wind_correlations.index)
plt.title('Correlation of Variables with Wind Energy Production')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Variables')
plt.show()




############## PHOTOVOLTAIC ###########

# Now create the photo_series with the correct index
photo_series = combined_data['Photovoltaic'].copy()

# Coerce non-numeric data to NaN and forward-fill any missing values
photo_series = pd.to_numeric(photo_series, errors='coerce').fillna(method='ffill')

# Set the frequency to daily
photo_series = photo_series.asfreq('D')

# Now let's try the seasonal decomposition again
decomposition = seasonal_decompose(photo_series.dropna(), model='additive', period=365)  # Assuming daily data and a yearly cycle


# Assuming decomposition is already done
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 10), dpi=100)

# Trend Component
axes[0].plot(trend.index, trend, label='Trend', color='tab:blue')
axes[0].set_ylabel('Trend')
axes[0].legend(loc='upper left')
axes[0].set_title('Seasonal Decomposition')
# Formatting the dates on the x-axis to show months


# Seasonal Component
axes[1].plot(seasonal.index, seasonal, label='Seasonality', color='tab:orange')
axes[1].set_ylabel('Seasonality')
axes[1].legend(loc='upper left')
# Formatting the dates on the x-axis to show months


# Residual Component
axes[2].plot(residual.index, residual, label='Residuals', color='tab:green')
axes[2].set_ylabel('Residuals')
axes[2].legend(loc='upper left')
# Formatting the dates on the x-axis to show months
axes[2].xaxis.set_major_locator(mdates.MonthLocator())
axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# Improve spacing and layout
plt.tight_layout()

# Automatically rotate the dates
for ax in axes:
    for label in ax.get_xticklabels(which='major'):
        label.set(rotation=45, horizontalalignment='right')

plt.show()




# Automatically selecting SARIMA parameters
smodel = pm.auto_arima(photo_series.dropna(), seasonal=True, m=12, stepwise=True,
                       suppress_warnings=True, error_action="ignore", max_p=3, max_q=3, max_order=5)

smodel.summary()



# Group by day of the year and calculate max/min
daily_max = photo_series.groupby([wind_series.index.month, wind_series.index.day]).max()
daily_min = photo_series.groupby([wind_series.index.month, wind_series.index.day]).min()


# Create a figure and plot
plt.figure(figsize=(12, 6))

# Create a range of dates for the current year for plotting
date_range = pd.date_range(start=f'{photo_series.index.year[-1]}-01-01', end=f'{photo_series.index.year[-1]}-12-31')

# Ensure the date_range doesn't exceed the wind_series index
date_range = date_range[date_range <= photo_series.index[-1]]

# Plotting the band
plt.fill_between(date_range, daily_min.values[:len(date_range)], daily_max.values[:len(date_range)], color='gray', alpha=0.2, label='Historical Max/Min Band')

# Plotting the current year's data
current_year_data = photo_series[wind_series.index.year == wind_series.index.year[-1]]
plt.plot(current_year_data.index, current_year_data, label='Current Year Wind Production', color='blue')

plt.title('Daily Photovoltaic Energy Production: Current Year vs Historical Max/Min (GWh)')
plt.xlabel('Date')
plt.ylabel('Photovoltaic Energy Production')
plt.legend()

# Improve formatting of the x-axis to show months
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

plt.xticks(rotation=45)
# Highlight the last available day
plt.scatter(current_year_data.index[-1], current_year_data.iloc[-1], color='red', zorder=5)
plt.annotate(f'Last Day: {current_year_data.iloc[-1]:.2f}', 
             (current_year_data.index[-1], current_year_data.iloc[-1]), 
             textcoords="offset points", xytext=(-50,10), ha='center', color='red')

plt.show()



features = ['wpgt', 'wspd', 'tavg', 'prcp', 'wdir','tmin','tmax','pres']  # Select based on your EDA
X = combined_data[features]
y = combined_data['Photovoltaic']  # Target variable

# Splitting the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mean_values_X_train = X_train.mean()
# If y_train is continuous and has missing values
mean_y_train = y_train.mean()
# Apply imputation to training data
X_train = X_train.fillna(mean_values_X_train)
# Apply the same means to test data
X_test= X_test.fillna(mean_values_X_train)
y_train= y_train.fillna(mean_y_train)



model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'MSE: {mean_squared_error(y_test, y_pred)}')


# Assuming y_test are your actual values and y_pred are the predictions from your model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.title('Actual vs. Predicted Values (GWh)')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)  # Diagonal line
plt.show()

residuals = y_test - y_pred

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.title('Residuals vs. Predicted Values ')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
plt.show()

feature_importance = model.coef_
features = X_train.columns  # Make sure this matches your preprocessed feature set

# Sort features by importance
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
plt.yticks(range(len(sorted_idx)), features[sorted_idx])
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

correlation_matrix = combined_data.corr()

# Extracting the 'Photovoltaic' column correlations, sorting them, and removing the self-correlation
photo_correlations = correlation_matrix['Photovoltaic'].sort_values(ascending=False).drop('Photovoltaic')

# Plotting
plt.figure(figsize=(10, 8))
sns.barplot(x=photo_correlations.values, y=wind_correlations.index)
plt.title('Correlation of Variables with Wind Energy Production')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Variables')
plt.show()

