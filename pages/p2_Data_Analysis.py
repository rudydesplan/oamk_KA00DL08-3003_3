import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

@st.cache_data
def load_data():
    fmi_data = pd.read_csv("fmi_weather_and_price_completed.csv")
    fmi_data['Time'] = pd.to_datetime(fmi_data['Time'], format='%Y/%m/%d %H:%M:%S')
    fmi_data = fmi_data.sort_values('Time')
    return fmi_data

def main():
    st.title("FMI Weather and Price Analysis")
    
    try:
        fmi_data = load_data()
        numerical_columns = fmi_data.drop(columns=['Time']).columns
        
        # Special treatment variables
        zero_inflated_vars = ['rain (mm)', 'snowfall (cm)', 'sunshine_duration (s)']
        log_scale_vars = ['shortwave_radiation (W/m²)']
        outlier_vars = ['Price']

        # --------------------------
        # Temporal Analysis
        # --------------------------
        st.header("Temporal Analysis")
        fig = px.line(fmi_data, x='Time', y='Price', 
                     title='Price Evolution Over Time')
        st.plotly_chart(fig, use_container_width=True)

        
        # --------------------------
        # Univariate Analysis
        # --------------------------
        st.header("Univariate Analysis")
        selected_var = st.selectbox("Select Numerical Variable", numerical_columns)

        if selected_var == 'wind_direction_100m (°)':
            st.subheader("Wind Direction Distribution")
            fig = px.bar_polar(fmi_data, 
                             theta=selected_var,
                             title="Wind Direction Distribution",
                             color_discrete_sequence=['skyblue'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                # Boxplot
                box_config = {
                    'log_x': selected_var in log_scale_vars,
                    'points': False if selected_var in outlier_vars else 'outliers'
                }
                fig = px.box(fmi_data, x=selected_var, 
                           title=f"Boxplot of {selected_var}", **box_config)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Histogram
                data = fmi_data[selected_var].dropna()
                
                if selected_var in outlier_vars:
                    filtered_data = data[(data >= -1) & (data <= 30)]
                    edges = np.histogram_bin_edges(filtered_data, bins='auto')
                    fig = px.histogram(filtered_data, x=selected_var,
                                     nbins=len(edges)-1, range_x=[-1, 30],
                                     title=f"{selected_var} Distribution (Focus Range)")
                else:
                    edges = np.histogram_bin_edges(data, bins='auto')
                    hist_config = {
                        'nbins': len(edges)-1,
                        'title': f"Histogram of {selected_var}"
                    }
                    if selected_var in log_scale_vars:
                        hist_config['log_x'] = True
                    fig = px.histogram(data, x=selected_var, **hist_config)

                fig.update_traces(opacity=0.75)
                fig.add_vline(x=data.mean(), line_dash="dash", line_color="red",
                            annotation_text="Mean")
                st.plotly_chart(fig, use_container_width=True)

                # Additional non-zero histogram for zero-inflated variables
                if selected_var in zero_inflated_vars:
                    filtered_non_zero = data[data > 0]
                    if not filtered_non_zero.empty:
                        edges_nonzero = np.histogram_bin_edges(filtered_non_zero, bins='auto')
                        fig_non_zero = px.histogram(filtered_non_zero, x=selected_var,
                                                  nbins=len(edges_nonzero)-1, log_x=True,
                                                  title=f"Non-Zero {selected_var} Distribution")
                        st.plotly_chart(fig_non_zero, use_container_width=True)

        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
