import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from pages.p2_Data_Analysis import load_data

def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) 
                      for i in range(df.shape[1])]
    return vif_data.sort_values('VIF', ascending=False)

def create_time_features(df):
    # Basic temporal features
    df['hour'] = df['Time'].dt.hour
    df['day_of_week'] = df['Time'].dt.dayofweek  # Monday=0, Sunday=6
    df['month'] = df['Time'].dt.month
    
    # Cyclical encoding using sine/cosine
    df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)
    
    df['month_sin'] = np.sin(2 * np.pi * (df['month']-1)/12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month']-1)/12)
    
    return df.drop(columns=['hour', 'day_of_week', 'month'])

def main():
    st.title("Price Prediction Modeling")
    
    try:
        # Load and prepare data
        fmi_data = load_data()
        fmi_data = create_time_features(fmi_data)
        
        lags = [1, 2, 24]
        for lag in lags:
            fmi_data[f'price_lag_{lag}'] = fmi_data['Price'].shift(lag)
        
        # THEN calculate rolling stats on LAGGED price
        windows = [3, 6, 12]
        for window in windows:
            fmi_data[f'price_rolling_mean_{window}'] = (
                fmi_data['price_lag_1']  # Use lagged price, not current price
                .rolling(window=window, min_periods=1)
                .mean()
            )
            fmi_data[f'price_rolling_std_{window}'] = (
                fmi_data['price_lag_1']
                .rolling(window=window, min_periods=1)
                .std()
            )
        
        fmi_data = fmi_data.dropna(subset=[f'price_lag_24'])
        
        # Define features and split data
        time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 
                        'month_sin', 'month_cos']
        weather_features = ['rain (mm)','shortwave_radiation (W/m²)', 'sunshine_duration (s)',
                           'cloud_cover (%)','wind_direction_100m (°)', 'temperature_2m (°C)','relative_humidity_2m (%)' ,
                           'snowfall (cm)','wind_speed_100m (km/h)']
        lag_features = ['price_lag_1','price_lag_2','price_lag_24']
        rolling_features = ['price_rolling_mean_3','price_rolling_std_3',
            'price_rolling_mean_6','price_rolling_std_6', 'price_rolling_mean_12','price_rolling_std_12']

        all_features = time_features + weather_features + lag_features + rolling_features
        
        X = fmi_data[all_features]
        y = fmi_data['Price']

        # Time-based split
        split_idx = int(len(fmi_data) * 0.7)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # --------------------------
        # TASK 1: VIF Results
        # --------------------------
        st.header("Multicollinearity Analysis (VIF)")
        vif_results = calculate_vif(X_train)
        
        col1, col2 = st.columns([2, 1])  # Adjusted column ratio
        with col1:
            st.dataframe(
                vif_results.style.format({"VIF": "{:.1f}"}),
                height=600,
                use_container_width=True
            )
        
        with col2:
            st.markdown("**VIF Interpretation Guide:**")
            st.markdown("- VIF < 5: Low multicollinearity")
            st.markdown("- 5 ≤ VIF < 10: Moderate multicollinearity")
            st.markdown("- VIF ≥ 10: High multicollinearity")
            st.markdown("\n")  # Add some spacing
            st.markdown("*Note: High VIF values indicate collinear features that may need investigation*")
            
        # --------------------------
        # TASK 2: Model Metrics
        # --------------------------
        st.header("Model Performance")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        linear_model = LinearRegression()
        linear_model.fit(X_train_scaled, y_train)

        # Calculate predictions
        y_pred_train = linear_model.predict(X_train_scaled)
        y_pred_test = linear_model.predict(X_test_scaled)

        # Create metrics dataframe
        def get_metrics(y_true, y_pred):
            return {
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R²': r2_score(y_true, y_pred)
            }

        metrics_df = pd.DataFrame({
            'Training': get_metrics(y_train, y_pred_train),
            'Testing': get_metrics(y_test, y_pred_test)
        }).T
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Training Set")
            st.metric("MAE", f"{metrics_df.loc['Training','MAE']:.2f}")
            st.metric("RMSE", f"{metrics_df.loc['Training','RMSE']:.2f}")
            st.metric("R²", f"{metrics_df.loc['Training','R²']:.2f}")
        
        with col2:
            st.markdown("### Test Set")
            st.metric("MAE", f"{metrics_df.loc['Testing','MAE']:.2f}", 
                     delta=f"{metrics_df.loc['Testing','MAE'] - metrics_df.loc['Training','MAE']:.2f}")
            st.metric("RMSE", f"{metrics_df.loc['Testing','RMSE']:.2f}", 
                     delta=f"{metrics_df.loc['Testing','RMSE'] - metrics_df.loc['Training','RMSE']:.2f}")
            st.metric("R²", f"{metrics_df.loc['Testing','R²']:.2f}", 
                     delta=f"{metrics_df.loc['Testing','R²'] - metrics_df.loc['Training','R²']:.2f}")

        # --------------------------
        # TASK 3: Statsmodels Summary
        # --------------------------
        st.header("Regression Summary")
        
        if st.checkbox("Show Full Statistical Summary"):
            X_train_sm = sm.add_constant(X_train_scaled, has_constant='add')
            sm_model = sm.OLS(y_train, X_train_sm).fit()
            
            summary = sm_model.summary().as_text()
            st.text_area("Regression Summary", 
                        value=summary,
                        height=800,
                        disabled=True)

    except Exception as e:
        st.error(f"Error in modeling: {str(e)}")

if __name__ == "__main__":
    main()
