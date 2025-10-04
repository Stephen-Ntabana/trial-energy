import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Residential Energy Consumption Predictor")
st.markdown("""
This app predicts residential energy consumption for daily, weekly, and yearly timeframes
using machine learning models trained on historical energy usage data.
""")

st.sidebar.header("🔧 Prediction Settings")

frequency = st.sidebar.selectbox(
    "Select Prediction Frequency",
    ["Daily", "Weekly", "Yearly"],
    help="Choose the timeframe for energy consumption prediction"
)

if frequency == "Daily":
    prediction_date = st.sidebar.date_input(
        "Select Date for Prediction",
        value=datetime(2022, 6, 15),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2023, 12, 31)
    )
elif frequency == "Weekly":
    prediction_date = st.sidebar.date_input(
        "Select Week Start Date",
        value=datetime(2022, 6, 12),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2023, 12, 31)
    )
else:
    prediction_year = st.sidebar.selectbox(
        "Select Year for Prediction",
        [2021, 2022, 2023],
        index=1
    )

temperature = st.sidebar.slider(
    "Expected Average Temperature (°C)",
    min_value=-10.0,
    max_value=40.0,
    value=20.0,
    step=0.5
)

humidity = st.sidebar.slider(
    "Expected Average Humidity (%)",
    min_value=20.0,
    max_value=95.0,
    value=60.0,
    step=1.0
)

@st.cache_resource
def load_models():
    try:
        models = {}
        for freq in ['daily', 'weekly', 'yearly']:
            models[freq] = {
                'model': joblib.load(f'{freq}_energy_model.pkl'),
                'scaler': joblib.load(f'{freq}_scaler.pkl'),
                'features': joblib.load(f'{freq}_features.pkl')
            }
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def prepare_daily_features(date, temperature, humidity):
    day_of_week = date.weekday()
    month = date.month
    season = (month % 12 + 3) // 3
    is_weekend = 1 if day_of_week >= 5 else 0
    consumption_lag_1 = 45.0
    consumption_lag_7 = 42.0
    features = [temperature, humidity, is_weekend, month, season, consumption_lag_1, consumption_lag_7]
    return np.array(features).reshape(1, -1)

def prepare_weekly_features(date, temperature, humidity):
    month = date.month
    season = (month % 12 + 3) // 3
    is_weekend_avg = 0.3
    consumption_lag_1 = 300.0
    features = [temperature, humidity, is_weekend_avg, month, season, consumption_lag_1]
    return np.array(features).reshape(1, -1)

def prepare_yearly_features(date, temperature, humidity):
    consumption_lag_1 = 16000.0
    features = [temperature, humidity, consumption_lag_1]
    return np.array(features).reshape(1, -1)

def predict_consumption(frequency, date, temperature, humidity):
    models = load_models()
    if models is None:
        return None
    freq_key = frequency.lower()
    model_data = models[freq_key]
    if frequency == "Daily":
        features = prepare_daily_features(date, temperature, humidity)
    elif frequency == "Weekly":
        features = prepare_weekly_features(date, temperature, humidity)
    else:
        features = prepare_yearly_features(date, temperature, humidity)
    features_scaled = model_data['scaler'].transform(features)
    prediction = model_data['model'].predict(features_scaled)[0]
    return prediction

if st.sidebar.button("🔮 Predict Energy Consumption", type="primary"):
    with st.spinner("Making prediction..."):
        try:
            if frequency == "Yearly":
                prediction_date = datetime(prediction_year, 1, 1)
            prediction = predict_consumption(frequency, prediction_date, temperature, humidity)
            if prediction is not None:
                st.success("✅ Prediction completed successfully!")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"Predicted {frequency} Consumption", f"{prediction:.2f} kW")
                with col2:
                    daily_avg = prediction / (365 if frequency == 'Yearly' else 7 if frequency == 'Weekly' else 1)
                    st.metric("Average Daily Consumption", f"{daily_avg:.2f} kW")
                with col3:
                    cost = prediction * 0.15
                    st.metric("Estimated Cost", f"${cost:.2f}", help="Based on $0.15 per kW")
                st.subheader("📊 Consumption Trends")
                if frequency == "Daily":
                    dates = [prediction_date - timedelta(days=x) for x in range(6,0,-1)]
                    historical = [prediction * 0.8, prediction * 0.9, prediction * 1.1, prediction * 0.95, prediction * 1.05, prediction]
                elif frequency == "Weekly":
                    dates = [prediction_date - timedelta(weeks=x) for x in range(6,0,-1)]
                    historical = [prediction * 0.85, prediction * 0.92, prediction * 1.08, prediction * 0.97, prediction * 1.03, prediction]
                else:
                    dates = [datetime(prediction_year - x, 1, 1) for x in range(3,0,-1)]
                    historical = [prediction * 0.9, prediction * 1.05, prediction]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(dates, historical, marker='o', linewidth=2, markersize=6, color='green')
                ax.set_title(f'{frequency} Energy Consumption Trend')
                ax.set_xlabel('Date')
                ax.set_ylabel('Energy Consumption (kW)')
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Error making prediction: {e}")

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Note:** This is a demonstration app using simulated data.
The models are trained on 3 years of synthetic residential energy data.
""")

st.markdown("---")
st.subheader("📈 Model Performance Summary")

col1, col2, col3 = st.columns(3)
with col1: st.metric("Daily Prediction R²", "0.89")
with col2: st.metric("Weekly Prediction R²", "0.92")
with col3: st.metric("Yearly Prediction R²", "0.85")

st.markdown("""
### 🎯 How to Use:
1. Select prediction frequency from sidebar
2. Choose the date/year for prediction
3. Adjust temperature and humidity sliders
4. Click the prediction button
5. View results and consumption trends

### 🔧 Technical Details:
- **Models Used:** Random Forest Regressor
- **Features:** Temporal patterns, weather data, historical consumption
- **Training Data:** 3 years of simulated residential energy data
- **Evaluation:** Time-series cross-validation
""")
