# Import necessary libraries
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Set up API and base URL for fetching data
api_key = "l333ljg4122qws9kxkb4hly7a8dje27vk46c7zkceih11wmnrj7lqreku176"
base_url = "https://metals-api.com/api"

# Function to fetch data for a given metal and timeframe
def fetch_data(symbol, start_date, end_date):
    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    all_data = {}

    while start_date <= end_date:
        current_end_date = min(start_date + timedelta(days=15), end_date)
        params = {
            "access_key": api_key,
            "base": "USD",
            "symbols": symbol,
            "start_date": start_date.strftime(date_format),
            "end_date": current_end_date.strftime(date_format)
        }
        response = requests.get(f"{base_url}/timeseries", params=params)

        if response.status_code == 200:
            data = response.json()
            if data.get('success', False):
                all_data.update(data.get("rates", {}))
            else:
                error_info = data.get('error', {}).get('info', 'Unknown error')
                st.error(f"API request failed: {error_info}")
                return None
        else:
            st.error(f"Error fetching data. Status code: {response.status_code}, Response: {response.text}")
            return None

        start_date = current_end_date + timedelta(days=1)

    return all_data if all_data else None

# Streamlit App Configuration
st.set_page_config(page_title="Price Prediction", layout="wide")

# Sidebar for user inputs
with st.sidebar:
    st.image(
        "https://media.licdn.com/dms/image/v2/C560BAQGC6QNyba_n5w/company-logo_200_200/company-logo_200_200/0/1630666228337/minexx_logo?e=2147483647&v=beta&t=Edza3G0e46BmdKdBC9S-zMrVpMXLiE6_D056T3--TFI",
        width=150)
    st.title("Price Predictor")
    st.info("Select a prediction period to fetch data and predict future prices.")
    
    current_date = st.date_input("Current Date", value=datetime.now().date())
    
    metal_symbol_map = {"TIN": "TIN", "TUNGSTEN": "TUNGSTEN"}
    metal = st.selectbox("Select Metal", ["TIN", "TUNGSTEN"])
    metal_symbol = metal_symbol_map[metal]

    prediction_period = st.selectbox("Select Prediction Period", ["1 Week", "3 Weeks", "1 Month", "3 Months", "6 Months"])

    period_days = {
        "1 Week": 7,
        "3 Weeks": 21,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180
    }
    
    start_date = datetime(2024, 8, 15)  
    end_date = start_date + timedelta(days=period_days.get(prediction_period, 30))

# Button to fetch the data
fetch_button = st.button(f"Fetch {metal} Data")

# Main section for displaying data and results
st.title(f"{metal} Price Prediction Dashboard")

# Fetch data and model training when button is clicked
if fetch_button:
    data = fetch_data(metal_symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))

    if data:
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={"index": "ds", metal_symbol: "y"})
        df = df[["ds", "y"]]

        # Handle missing values
        df['y'] = df['y'].fillna(method='ffill')

        # Display data
        st.subheader(f"📊 Fetched {metal} Data")
        st.write(df.head(30))

        # Plot the data
        st.subheader(f"📈 {metal} Price Over Time")
        st.line_chart(df.set_index('ds')['y'])
        
        # Prophet model training and storing in session_state
        st.subheader(f"🔮 {metal} Prophet Forecast")
        model = Prophet(
            changepoint_prior_scale=0.1,
            yearly_seasonality=True,
            weekly_seasonality=True
        )
        model.fit(df)
        st.session_state['prophet_model'] = model  # Store the trained model in session_state

        # Making future predictions
        future = model.make_future_dataframe(periods=period_days.get(prediction_period, 30), freq='D')
        forecast = model.predict(future)
        fig1 = model.plot(forecast)
        st.pyplot(fig1)
        st.session_state['forecast'] = forecast
        
        # ARIMA model evaluation (optional)
        try:
            arima_model = ARIMA(df['y'], order=(5, 1, 0))
            arima_result = arima_model.fit()
            arima_forecast = arima_result.get_forecast(steps=period_days.get(prediction_period, 30))
            arima_pred = arima_forecast.predicted_mean
        except Exception as e:
            st.write(f"ARIMA Model Error: {e}")

    else:
        st.write("⚠️ No data fetched. Please check the date range or API details.")

# Predict price for a specific date
st.subheader(f"📅 Predict {metal} Price for a Specific Date")

user_input_date = st.date_input("Select the date for which you want to predict the price", min_value=datetime.now().date())

if user_input_date:
    try:
        pred_date = pd.to_datetime(user_input_date)
        
        # Assume the model is already trained after fetching data
        model = st.session_state.get('prophet_model')
        current_forecast = st.session_state.get('forecast')
        max_date = current_forecast['ds'].max() if current_forecast is not None else pd.Timestamp.now()

        additional_days = (pred_date - max_date).days

        if additional_days > 0:
            future = model.make_future_dataframe(periods=additional_days + 1, freq='D')
            forecast = model.predict(future)
            st.session_state['forecast'] = forecast  # Update the session_state with the new forecast
        else:
            forecast = current_forecast

        predicted_price_row = forecast[forecast['ds'] == pred_date]

        if not predicted_price_row.empty:
            predicted_price = predicted_price_row['yhat'].values[0]
            st.success(f"The predicted price of {metal} on {user_input_date} is: ${predicted_price:.2f}")
            st.balloons()
        else:
            st.error(f"Prediction is not available for {user_input_date}. Please try a date within the model's forecast range.")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Custom CSS for styling
st.markdown("""
    <style>
        .css-18e3th9 {
            padding: 1.5rem 1rem;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 1rem;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .css-1v0mbdj {
            display: flex;
            justify-content: center;
        }
        .css-1adrfps {
            color: #FF6347;
        }
    </style>
""", unsafe_allow_html=True)
