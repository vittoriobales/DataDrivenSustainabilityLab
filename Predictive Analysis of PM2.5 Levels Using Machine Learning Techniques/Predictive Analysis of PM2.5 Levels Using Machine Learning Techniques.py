# -*- coding: utf-8 -*-
"""

@author: vitto
"""

import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error



def fetch_data_for_pollutant(country_code, pollutant, limit=100000):
    base_url = "https://api.openaq.org/v1/measurements"
    params = {'country': country_code, 'limit': 100, 'parameter': pollutant}
    all_data = []
    while limit > 0:
        response = requests.get(base_url, params=params)
        if response.status_code == 429:
            print("Rate limit exceeded. Waiting before retrying...")
            time.sleep(10)  # Adjust based on the API's rate limit reset time
            continue  # Skip to the next iteration to retry
        elif response.status_code != 200:
            print(f"Error fetching data: {response.status_code}")
            break  # Exit the loop on other errors
        data = response.json().get('results', [])
        all_data.extend(data)
        limit -= len(data)
        if not data:
            break  # Exit loop if no more data is returned
        print(f"Fetched {len(data)} records for {country_code}, {pollutant}. Total: {len(all_data)}")
    return all_data

def fetch_pm_data(countries, pollutants=['pm25', 'pm10']):
    df_list = []
    for country in countries:
        for pollutant in pollutants:
            data = fetch_data_for_pollutant(country, pollutant)
            df_list.append(pd.DataFrame(data))
            time.sleep(1)  # Respectful delay between requests for different pollutants
    return pd.concat(df_list, ignore_index=True)

# Example usage
countries = ['IT', 'US', 'FR', 'DE', 'CN'] # Example with Italy; add more as needed
df = fetch_pm_data(countries)

# Convert the 'date' column to datetime and extract day of year
df['date'] = pd.to_datetime(df['date'].apply(lambda x: x['utc']))
df['day_of_year'] = df['date'].dt.dayofyear

print(df.head())

# Filter for PM2.5 data if you haven't already done so
df_pm25 = df[df['parameter'] == 'pm25'].copy()

# Convert date strings to datetime objects (if not already done)
df_pm25['date'] = pd.to_datetime(df_pm25['date'])

# Add any additional preprocessing needed
# For example, extracting day of year for temporal analysis and modeling
df_pm25['day_of_year'] = df_pm25['date'].dt.dayofyear

# Summary statistics for PM2.5 by location
pm25_summary = df_pm25.groupby('location')['value'].describe()
print(pm25_summary)


plt.figure(figsize=(10, 6))
sns.boxplot(x='location', y='value', data=df_pm25)
plt.xticks(rotation=90)
plt.title('PM2.5 Levels Across Locations')
plt.xlabel('Location')
plt.ylabel('PM2.5 µg/m³')
plt.tight_layout()
plt.show()

X = df_pm25[['day_of_year']]  # Simplified feature set for demonstration
y = df_pm25['value']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

