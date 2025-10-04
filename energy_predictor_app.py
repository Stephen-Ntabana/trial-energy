import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Energy Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Title and description
st.title("⚡ Residential Energy Consumption Predictor")
st.markdown("""
This app predicts residential energy consumption for daily, weekly, and yearly timeframes
using machine learning models.
""")

# Pre-trained model parameters (instead of loading from files)
class EnergyPredictor:
    def __init__(self):
        self.models = {
            'daily': self.create_daily_model(),
            'weekly': self.create_weekly_model(), 
            'yearly': self.create_yearly_model()
        }
        
        self.scalers = {
            'daily': StandardScaler(),
            'weekly': StandardScaler(),
            'yearly': StandardScaler()
        }
        
        # Pre-fit scalers with typical value ranges
        self._pre_fit_scalers()
    
    def create_daily_model(self):
        """Create a pre-trained daily prediction model"""
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        # Model would be pre-trained in real scenario
        return model
    
    def create_weekly_model(self):
        """Create a pre-trained weekly prediction model"""
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        return model
    
    def create_yearly_model(self):
        """Create a pre-trained yearly prediction model"""
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        return model
    
    def _pre_fit_scalers(self):
        """Pre-fit scalers with typical value ranges"""
        # Daily features: [temp, humidity, is_weekend, month, season, lag1, lag7]
        daily_data = np.array([
            [20, 60, 0, 6, 3, 45, 42],
            [25, 70, 1, 7, 3, 50, 48],
            [15, 50, 0, 1, 1, 60, 55],
            [30, 80, 1, 8, 3, 40, 38]
        ])
        self.scalers['daily'].fit(daily_data)
        
        # Weekly features: [temp, humidity, is_weekend_avg, month, season, lag1]
        weekly_data = np.array([
            [22, 65, 0.3, 6, 3, 300],
            [18, 55, 0.4, 1, 1, 350],
            [28, 75, 0.2, 8, 3, 280]
        ])
        self.scalers['weekly'].fit(weekly_data)
        
        # Yearly features: [temp, humidity, lag1]
        yearly_data = np.array([
            [20, 60, 16000],
            [22, 65, 16500],
            [18, 55, 15500]
        ])
        self.scalers['yearly'].fit(yearly_data)

# Initialize predictor
@st.cache_resource
def load_predictor():
    return EnergyPredictor()

# Sidebar for user inputs
st.sidebar.header("🔧 Prediction Settings")

# Frequency selection
frequency = st.sidebar.selectbox(
    "Select Prediction Frequency",
    ["Daily", "Weekly", "Yearly"],
    help="Choose the timeframe for energy consumption prediction"
)

# Date input based on frequency
if frequency == "Daily":
    prediction_date = st.sidebar.date_input(
        "Select Date for Prediction",
        value=datetime(2023, 6, 15),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 12, 31)
    )
elif frequency == "Weekly":
    prediction_date = st.sidebar.date_input(
        "Select Week Start Date",
        value=datetime(2023, 6, 12),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2024, 12, 31)
    )
else:  # Yearly
    prediction_year = st.sidebar.selectbox(
        "Select Year for Prediction",
        [2022, 2023, 2024],
        index=1
    )

# Additional features
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

# Feature preparation functions
def prepare_daily_features(date, temperature, humidity):
    """Prepare features for daily prediction"""
    day_of_week = date.weekday()
    month = date.month
    season = (month % 12 + 3) // 3
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Simulated lag features (in real scenario, these come from historical data)
    consumption_lag_1 = 45.0 + (temperature - 20) * 0.5
    consumption_lag_7 = 42.0 + (temperature - 20) * 0.3
    
    features = [temperature, humidity, is_weekend, month, season, consumption_lag_1, consumption_lag_7]
    return np.array(features).reshape(1, -1)

def prepare_weekly_features(date, temperature, humidity):
    """Prepare features for weekly prediction"""
    month = date.month
    season = (month % 12 + 3) // 3
    is_weekend_avg = 0.3
    
    # Simulated lag feature
    consumption_lag_1 = 300.0 + (temperature - 20) * 5
    
    features = [temperature, humidity, is_weekend_avg, month, season, consumption_lag_1]
    return np.array(features).reshape(1, -1)

def prepare_yearly_features(date, temperature, humidity):
    """Prepare features for yearly prediction"""
    # Simulated lag feature
    consumption_lag_1 = 16000.0 + (temperature - 20) * 100
    
    features = [temperature, humidity, consumption_lag_1]
    return np.array(features).reshape(1, -1)

# Prediction function with simulated ML logic
def predict_consumption(frequency, date, temperature, humidity):
    """Make prediction based on user inputs using simulated ML logic"""
    
    predictor = load_predictor()
    freq_key = frequency.lower()
    
    # Prepare features based on frequency
    if frequency == "Daily":
        features = prepare_daily_features(date, temperature, humidity)
        # Base prediction with adjustments
        base_consumption = 45.0
        temp_effect = (temperature - 20) * 0.8
        seasonal_effect = 1.2 if date.month in [12, 1, 2, 6, 7, 8] else 1.0
        weekend_effect = 1.15 if date.weekday() >= 5 else 1.0
        prediction = base_consumption * seasonal_effect * weekend_effect + temp_effect
        
    elif frequency == "Weekly":
        features = prepare_weekly_features(date, temperature, humidity)
        # Base prediction with adjustments
        base_consumption = 315.0
        temp_effect = (temperature - 20) * 6
        seasonal_effect = 1.25 if date.month in [12, 1, 2, 6, 7, 8] else 1.0
        prediction = base_consumption * seasonal_effect + temp_effect
        
    else:  # Yearly
        features = prepare_yearly_features(date, temperature, humidity)
        # Base prediction with adjustments
        base_consumption = 16400.0
        temp_effect = (temperature - 20) * 120
        prediction = base_consumption + temp_effect
    
    # Add some random variation for realism
    prediction = prediction * np.random.uniform(0.95, 1.05)
    
    return max(prediction, 10)  # Ensure positive value

# Prediction button
if st.sidebar.button("🔮 Predict Energy Consumption", type="primary"):
    with st.spinner("Making prediction..."):
        try:
            if frequency == "Yearly":
                prediction_date = datetime(prediction_year, 1, 1)
            
            prediction = predict_consumption(frequency, prediction_date, temperature, humidity)
            
            st.success("✅ Prediction completed successfully!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label=f"Predicted {frequency} Consumption",
                    value=f"{prediction:.2f} kW",
                    delta=None
                )
            
            with col2:
                if frequency == "Yearly":
                    daily_avg = prediction / 365
                    timeframe = "Daily"
                elif frequency == "Weekly":
                    daily_avg = prediction / 7
                    timeframe = "Daily"
                else:
                    daily_avg = prediction
                    timeframe = "Hourly Avg"
                
                st.metric(
                    label=f"Average {timeframe} Consumption",
                    value=f"{daily_avg:.2f} kW",
                    delta=None
                )
            
            with col3:
                cost = prediction * 0.15  # Assuming $0.15 per kW
                st.metric(
                    label="Estimated Cost",
                    value=f"${cost:.2f}",
                    help="Based on average electricity rate of $0.15 per kW"
                )
            
            # Visualization
            st.subheader("📊 Consumption Trends")
            
            # Create sample trend data
            if frequency == "Daily":
                dates = [prediction_date - timedelta(days=x) for x in range(6, 0, -1)]
                historical = [prediction * 0.8, prediction * 0.9, prediction * 1.1, 
                            prediction * 0.95, prediction * 1.05, prediction]
                title_suffix = "Daily"
            elif frequency == "Weekly":
                dates = [prediction_date - timedelta(weeks=x) for x in range(6, 0, -1)]
                historical = [prediction * 0.85, prediction * 0.92, prediction * 1.08, 
                            prediction * 0.97, prediction * 1.03, prediction]
                title_suffix = "Weekly"
            else:  # Yearly
                dates = [datetime(prediction_year - x, 1, 1) for x in range(3, 0, -1)]
                historical = [prediction * 0.9, prediction * 1.05, prediction]
                title_suffix = "Yearly"
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(dates, historical, marker='o', linewidth=2, markersize=6, color='green')
            ax.set_title(f'Energy Consumption Trend ({title_suffix})')
            ax.set_xlabel('Date')
            ax.set_ylabel('Energy Consumption (kW)')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Additional insights
            st.subheader("💡 Insights")
            
            if temperature > 25:
                st.info("🔆 High temperature detected: Cooling systems may increase energy usage.")
            elif temperature < 10:
                st.info("❄️ Low temperature detected: Heating systems may increase energy usage.")
            
            if humidity > 80:
                st.warning("💧 High humidity: May increase AC usage for dehumidification.")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Additional information
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Note:** This is a demonstration app using simulated prediction models.
In production, models would be trained on historical energy data.
""")

# Main area additional info
st.markdown("---")
st.subheader("📈 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Daily Prediction", "~45 kW avg", "Real-time patterns")

with col2:
    st.metric("Weekly Prediction", "~315 kW avg", "Seasonal trends")

with col3:
    st.metric("Yearly Prediction", "~16,400 kW avg", "Long-term analysis")

st.markdown("""
### 🎯 How to Use:
1. **Select prediction frequency** from sidebar
2. **Choose the date/year** for prediction
3. **Adjust temperature and humidity** sliders based on weather forecast
4. **Click the prediction button** 
5. **View results** and consumption trends

### 🔧 Technical Details:
- **Algorithm:** Random Forest Regressor (simulated)
- **Features:** Temperature, humidity, temporal patterns, historical consumption
- **Data:** Simulated residential energy patterns
- **Output:** Energy consumption in kilowatts (kW)

### 🌡️ Weather Impact:
- **Hot days (>25°C):** Higher AC usage
- **Cold days (<10°C):** Higher heating usage  
- **High humidity:** Increased dehumidification load
- **Weekends:** Typically 15-20% higher consumption
""")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Residential Energy Consumption Predictor | Machine Learning Project"
    "</div>",
    unsafe_allow_html=True
)
