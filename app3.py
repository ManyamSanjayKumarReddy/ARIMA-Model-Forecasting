import streamlit as st
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Main Data
data = {
    'date': ['2023-02-01', '2023-02-02', '2023-02-03', '2023-02-04', '2023-02-05', '2023-02-06', '2023-02-09', '2023-02-10', '2023-02-11', '2023-02-12', '2023-02-15', '2023-02-16', '2023-02-17', '2023-02-20', '2023-02-21'],
    'quality': [92, 98, 98, 98.5, 98.2, 98.3, 98.2, 98, 97, 92, 90, 96, 97, 98.2, 98.8]
}
df = pd.DataFrame(data)

# Convert into Date Time Format
df['date'] = pd.to_datetime(df['date'])

# Making Date as Index
df.set_index('date', inplace=True)

# Doing the re indexing based on the dates
date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
complete_df = df.reindex(date_range)

# Doing Value Imputation
complete_df['quality'] = complete_df['quality'].ffill()


# Fitting the AutoRegressive Intergrated Moving Averages Model
model = ARIMA(complete_df['quality'], order = (5, 0, 1))
model_fit = model.fit()

# Next 30 day forecast
forecast = model_fit.forecast(steps = 30)

# Creating DF with next 30 days 
forecast_dates = pd.date_range(start = complete_df.index[-1] + pd.Timedelta(days = 1), periods = 30, freq = 'D')
forecast_df = pd.DataFrame({'date': forecast_dates, 'forecasted_quality': forecast})


st.title('Quality Forecasting with ARIMA')

st.write('Original Data:')
st.write(df)
st.write('Complete Data with Forward Fill:')
st.write(complete_df)

st.subheader('Forecasted Quality for Next 30 Days')
st.write(forecast_df["forecasted_quality"])

st.write('Forecast Plot:')
plt.figure(figsize = (12, 6))
plt.plot(complete_df.index, complete_df['quality'], label = 'Original Quality')
plt.plot(forecast_df['date'], forecast_df['forecasted_quality'], label = 'Forecasted Quality')
plt.xlabel('Date')
plt.ylabel('Quality')
plt.title('Quality Forecasting with ARIMA')
plt.legend()
st.pyplot(plt)


st.sidebar.title('ARIMA Forecasting App')
st.sidebar.write('This app demonstrates ARIMA forecasting for quality percentage based on provided data. It includes the following features:')

st.sidebar.subheader('1. Original Data')
st.sidebar.write('View the original data used for analysis.')
# st.sidebar.write(df)


st.sidebar.subheader('2. Complete Data with Forward Fill')
st.sidebar.write('View the data with forward-filled missing values.')
# st.sidebar.write(complete_df)


st.sidebar.subheader('3. Forecasted Quality for Next 30 Days')
st.sidebar.write('View the forecasted quality values for the next 30 days.')
# st.sidebar.write(complete_df.iloc[-1].name + pd.DateOffset(days=1), forecast)

