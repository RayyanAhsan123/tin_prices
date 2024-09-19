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
    
    # Use date picker to select the current date (shows the calendar)
    current_date = st.date_input("Current Date", value=datetime.now().date())
    
    # Metal selection
    metal_symbol_map = {"TIN": "TIN", "TUNGSTEN": "TUNGSTEN"}
    metal = st.selectbox("Select Metal", ["TIN", "TUNGSTEN"])
    metal_symbol = metal_symbol_map[metal]

    # User input for prediction period
    prediction_period = st.selectbox("Select Prediction Period", ["1 Week", "3 Weeks", "1 Month", "3 Months", "6 Months"])

    # Calculate the end date based on selected prediction period
    period_days = {
        "1 Week": 7,
        "3 Weeks": 21,
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180
    }

    # Internal start date (hidden from dashboard but used in the app)
    start_date = datetime(2024, 8, 1)  # Set this as a fixed or default start date

    end_date = start_date + timedelta(days=period_days.get(prediction_period, 15))

    # Removed the line that displays the end date
    # st.write(f"Prediction period will end on: {end_date.strftime('%Y-%m-%d')}")

# Button to fetch the data
fetch_button = st.button(f"Fetch {metal} Data")

# Main section for displaying data and results
st.title(f"{metal} Price Prediction Dashboard")
# Prophet model training and forecasting
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
      # ARIMA Model evaluation
        try:
            arima_model = ARIMA(df['y'], order=(5, 1, 0))
            arima_result = arima_model.fit()
            arima_forecast = arima_result.get_forecast(steps=prediction_days)
            arima_pred = arima_forecast.predicted_mean
        except Exception as e:
            st.write(f"ARIMA Model Error: {e}")

    else:
        st.write("⚠️ No data fetched. Please check the date range or API details.")
        # Predict price for a specific date using both calendar and manual text input
st.subheader(f"📅 Predict {metal} Price for a Specific Date")

# Give users an option to either pick a date via a calendar or manually enter it
use_calendar = st.radio("Choose input method:", ("Calendar", "Manual Entry (YYYY-MM-DD)"))

if use_calendar == "Calendar":
    user_input = st.date_input("Select the date for which you want to predict the price:", datetime.now().date())
else:
    user_input = st.text_input("Enter the date for which you want to predict the price (YYYY-MM-DD):")

# Process the input date for predictions
if user_input:
    try:
        if use_calendar == "Manual Entry (YYYY-MM-DD)":
            # If manual entry, parse the date from the input text
            pred_date = datetime.strptime(user_input, '%Y-%m-%d').date()
        else:
            # If using calendar, no need to parse, use the selected date
            pred_date = user_input

        # Retrieve the forecast from session_state
        forecast = st.session_state.get('forecast')

        if forecast is None:
            st.error("Forecast data is not available. Fetch the data first.")
        else:
            min_date = forecast['ds'].min()
            max_date = forecast['ds'].max()

            if pred_date > max_date.date():
                st.warning(f"Extending forecast to include {pred_date}.")

                # Ensure the Prophet model is available in session_state
                model = st.session_state.get('prophet_model')
                if model is None:
                    st.error("The model is not available. Fetch the data and train the model first.")
                else:
                    # Calculate additional days needed
                    additional_days = (pred_date - max_date.date()).days

                    # Extend the future dataframe by the additional days
                    future = model.make_future_dataframe(periods=additional_days + 1, freq='D')
                    forecast = model.predict(future)
                    st.session_state['forecast'] = forecast

            # After extending, check if the date is now in range
            min_date = forecast['ds'].min()
            max_date = forecast['ds'].max()

            if min_date.date() <= pred_date <= max_date.date():
                predicted_price = forecast[forecast['ds'] == pred_date.strftime('%Y-%m-%d')]['yhat'].values[0]
                st.success(f"The predicted price of {metal} on {pred_date} is: ${predicted_price:.2f}")
                st.balloons()
            else:
                st.error(f"Please enter a valid date within the forecast range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    except ValueError:
        st.error("Please enter a valid date in the format YYYY-MM-DD.")


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
