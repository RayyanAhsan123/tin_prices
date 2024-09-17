import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Set up your API and base URL for fetching data
api_key = "l333ljg4122qws9kxkb4hly7a8dje27vk46c7zkceih11wmnrj7lqreku176"
base_url = "https://metals-api.com/api"

# Function to fetch data for a given metal and timeframe
def fetch_data(symbol, start_date, end_date):
    date_format = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, date_format)
    end_date = datetime.strptime(end_date, date_format)

    all_data = {}

    # Fetch data in smaller chunks, e.g., 15 days
    while start_date <= end_date:
        current_end_date = min(start_date + timedelta(days=30), end_date)  # 30-day chunk
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

        start_date = current_end_date + timedelta(days=1)  # Move to the next chunk

    return all_data if all_data else None

# Streamlit App Configuration
st.set_page_config(page_title=" Price Prediction", layout="wide")

# Sidebar for user inputs
with st.sidebar:
    st.image(
        "https://media.licdn.com/dms/image/v2/C560BAQGC6QNyba_n5w/company-logo_200_200/company-logo_200_200/0/1630666228337/minexx_logo?e=2147483647&v=beta&t=Edza3G0e46BmdKdBC9S-zMrVpMXLiE6_D056T3--TFI",
        width=150)
    st.title("Price Predictor")
    st.info("Select a start date to fetch data and predict future prices.")
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    st.markdown(f"### Current Date: {current_date}")

    # Metal selection with correct symbols
    metal_symbol_map = {"TIN": "TIN", "TUNGSTEN": "TUNGSTEN"}
    metal = st.selectbox("Select Metal", ["TIN", "TUNGSTEN"])
    metal_symbol = metal_symbol_map[metal]

    # User input for start date
    start_date = st.date_input("Start Date", datetime(2024, 8, 20))

    # User input for prediction period
    prediction_period = st.selectbox("Select Prediction Period", ["1 Week", "3 Weeks", "1 Month", "3 Months", "6 Months"])

    # Calculate the end date based on selected prediction period
    if prediction_period == "1 Week":
        end_date = start_date + timedelta(weeks=1)
    elif prediction_period == "3 Weeks":
        end_date = start_date + timedelta(weeks=3)
    elif prediction_period == "1 Month":
        end_date = start_date + timedelta(days=30)
    elif prediction_period == "3 Months":
        end_date = start_date + timedelta(days=3 * 30)
    elif prediction_period == "6 Months":
        end_date = start_date + timedelta(days=6 * 30)

    # Display the calculated end date
    st.write(f"Prediction period will end on: {end_date.strftime('%Y-%m-%d')}")

# Convert dates to strings for API
start_date_str = start_date.strftime('%Y-%m-%d')
end_date_str = end_date.strftime('%Y-%m-%d')

# Button to fetch the data
fetch_button = st.button(f"Fetch {metal} Data")

# Main section for displaying data and results
st.title(f"{metal} Price Prediction Dashboard")

# Fetch data only when the button is clicked
if fetch_button:
    data = fetch_data(metal_symbol, start_date_str, end_date_str)

    if data:
        df = pd.DataFrame.from_dict(data, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={"index": "ds", metal_symbol: "y"})
        df = df[["ds", "y"]]

        # Handle missing values by filling with the mean or interpolation
        df['y'] = df['y'].fillna(method='ffill')

        # Display data
        st.subheader(f"📊 Fetched {metal} Data")
        st.write(df.head(30))

        # Plot the data
        st.subheader(f"📈 {metal} Price Over Time")
        st.line_chart(df.set_index('ds')['y'])
        
        # Calculate number of prediction days based on the user-inputted prediction period or custom date
        prediction_days = (datetime.strptime("2024-09-18", '%Y-%m-%d') - start_date).days
        # Prophet model training and forecasting
        st.subheader(f"🔮 {metal} Prophet Forecast")
        model = Prophet(
            changepoint_prior_scale=0.1,
            yearly_seasonality=True,
            weekly_seasonality=True
        )
        model.fit(df)

        # Use the calculated number of prediction days
        future = model.make_future_dataframe(periods=prediction_days)
        forecast = model.predict(future)

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # Store forecast for later use
        st.session_state['forecast'] = forecast

        # ARIMA Model evaluation in the background without displaying results
        try:
            # Fit the ARIMA model with data, ensuring no missing values
            arima_model = ARIMA(df['y'], order=(5, 1, 0))
            arima_result = arima_model.fit()

            # Make ARIMA predictions (used in backend)
            arima_forecast = arima_result.get_forecast(steps=prediction_days)
            arima_pred = arima_forecast.predicted_mean

        except Exception as e:
            st.write(f"ARIMA Model Error: {e}")

    else:
        st.write("⚠️ No data fetched. Please check the date range or API details.")

# Get user input for a specific prediction date
st.subheader(f"📅 Predict {metal} Price for a Specific Date")
user_input = st.text_input("Enter the date for which you want to predict the price (YYYY-MM-DD):")

if user_input:
    try:
        # Ensure the entered date is valid
        pred_date = datetime.strptime(user_input, '%Y-%m-%d')

        # Retrieve the forecast data
        forecast = st.session_state.get('forecast')

        if forecast is None:
            st.error("Forecast data is not available. Fetch the data first.")
        else:
            # Ensure the entered date is within the forecasted range
            if pred_date < forecast['ds'].min() or pred_date > forecast['ds'].max():
                st.error(f"Please enter a date within the forecast range: {forecast['ds'].min().strftime('%Y-%m-%d')} to {forecast['ds'].max().strftime('%Y-%m-%d')}")
            else:
                # Find the predicted price for the entered date
                predicted_price = forecast[forecast['ds'] == user_input]['yhat'].values[0]
                st.success(f"The predicted price of {metal} on {user_input} is: ${predicted_price:.2f}")
                st.balloons()

    except ValueError:
        st.error("Invalid date format. Please enter a valid date in YYYY-MM-DD format.")
    except Exception as e:
        st.error(f"Error predicting price: {e}")

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
