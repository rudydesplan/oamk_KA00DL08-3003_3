import streamlit as st
import pandas as pd

def show_conclusions():
    st.header("Conclusions")
    
    # Section 1: Key Weather Drivers
    with st.expander("1. Key Weather Drivers of Electricity Prices"):
        st.subheader("Significant Weather Factors")
        
        st.markdown("""
        **a) Temperature (`temperature_2m (°C)`)**
        - **Coefficient**: -0.1258 (p < 0.001)  
        - **Interpretation**: 1°C decrease → +0.13 €/MWh  
        - **Mechanism**:  
          - Colder temperatures increase heating demand (electric heating dominates in Finland) 
          - Impacts energy imports: Finland relies on electricity imports during cold spells.""")

        st.markdown("""
        **b) Wind Speed (`wind_speed_100m (km/h)`)**
        - **Coefficient**: -0.1329 (p < 0.001)  
        - **Interpretation**: 10 km/h increase → -1.33 €/MWh  
        - **Mechanism**:  
          - Higher wind speeds boost domestic wind power generation (Finland's wind capacity: ~5.7 GW in 2023)
          - Reduces reliance on imported power from Sweden/Norway.""")

        st.markdown("""
        **c) Solar Radiation (`shortwave_radiation (W/m²)`)**
        - **Coefficient**: -0.4302 (p < 0.001)  
        - **Interpretation**: 100 W/m² increase → -43 €/MWh  
        - **Mechanism**:
          - Despite Finland's low solar penetration (<1% of mix), sunlight reduces commercial lighting demand.
          - Correlates with high-pressure systems that often bring cold, still weather (paradoxical relationship)""")

        st.markdown("""
        **d) Cloud Cover (`cloud_cover (%)`)**
        - **Coefficient**: -0.0612 (p < 0.001)  
        - **Interpretation**: 10% increase → -0.61 €/MWh  
        - **Counterintuitive Finding**:
          - Overcast conditions often accompany strong coastal winds in the Baltic region.""")

    # Section 2: Temporal Patterns
    with st.expander("2. Temporal Patterns"):
        st.subheader("Time-Based Price Patterns")
        
        cols = st.columns(3)
        with cols[0]:
            st.markdown("""
            **Daily Cycle (`hour_sin/hour_cos`)**  
            - Evening peaks demand visible in coefficients (17:00-20:00)  
            - `hour_cos` = -0.45(peak evening prices)""")
            
        with cols[1]:
            st.markdown("""
            **Weekly Cycle (`day_sin/day_cos`)**  
            - Weekend demand drop (-5%)  
            - Possible Lower industrial demand reflected in coefficients""")
            
        with cols[2]:
            st.markdown("""
            **Seasonal Cycle (`month_sin/month_cos`)**  
            - Winter premium: +20-30% prices higher than summer  
            - Aligns with Finland's heating-dominated demand seasonality""")

    # Section 3: Market Dynamics
    with st.expander("3. Market Dynamics Dominance"):
        st.subheader("Price Autocorrelation")
        st.markdown("""
        **Lagged Price Impact**  
        - `price_lag_1`: 6.60 (85% variance explained)  
        - `price_lag_24`: 0.76 (daily cycle)  
        
        **Interpretation**:  
        - Electricity markets are strongly autoregressive due to:  
          - Generator commitment decisions  
          - Cross-border trading patterns  
        - Weather acts as secondary modifier to core market dynamics.  

    # Section 4: Model Validation
    with st.expander("4. Model Validation"):
        st.subheader("Performance Metrics")
        metrics = pd.DataFrame({
            'Metric': ['R²', 'MAE (€/MWh)', 'Durbin-Watson'],
            'Training': [0.93, 0.89, 1.975],
            'Testing': [0.91, 2.88, '-']
        })
        st.table(metrics.set_index('Metric'))

    # Final Conclusion
    st.header("Overall Conclusion", divider='rainbow')
    conclusion = """
    Weather conditions **modify but don't dominate** Finnish electricity prices:
    
    1. **Primary Drivers**:  
       - Market dynamics (85% variance)  
       - Temperature (heating demand)  
       - Wind (domestic generation)
    
    2. **Secondary Factors**:  
       - Solar radiation (lighting demand)  
       - Cloud cover (wind proxy)
    
    3. **Insignificant Factors**:  
       - Rainfall  
       - Humidity
    
    **Market Strategy**: Weather derivatives could hedge 15% non-autoregressive risk
    """
    st.markdown(conclusion)

# Run in Streamlit
if __name__ == "__main__":
    show_conclusions()
