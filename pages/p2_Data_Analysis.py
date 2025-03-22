import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    df = pd.read_csv('fmi_weather_and_price_completed.csv')
    df['Time'] = pd.to_datetime(df['Time'], format='%Y/%m/%d %H:%M:%S')
    df = df.sort_values('Time')
    return df

def main():
    st.title("Exploratory Data Analysis")
    
    try:
        fmi_data = load_data()

        # --------------------------
        # 1. Numerical Descriptive Stats
        # --------------------------
        st.header("Numerical Descriptive Statistics")
        numerical_columns = fmi_data.drop(columns=['Time']).columns
      
        numeric_descriptive_stats = fmi_data.drop(columns=['Time']).describe().T
        st.dataframe(numeric_descriptive_stats.style.format("{:.2f}"), 
                    height=500,
                    use_container_width=True)

        # --------------------------
        # Numerical Analysis
        # --------------------------
        st.header("Numerical Variables Analysis")

        edges = np.histogram_bin_edges(fmi_data, bins='auto')
        
        col1, col2 = st.columns(2)
        with col1:
            var = st.selectbox("Select Numerical Variable", numerical_columns)
            fig = px.box(fmi_data, x=var, title=f"Boxplot of {var}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(fmi_data, x=var, nbins=len(edges)-1, title=f"Distribution of {var}")
            fig.add_vline(x=fmi_data[var].mean(), line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)

        # --------------------------
        # Bivariate Analysis
        # --------------------------
        st.header("Bivariate Analysis")
        col1 = st.columns(1)
        
        with col1:
            num_feat = st.selectbox("Select Numerical Feature", 
                                   [c for c in numerical_columns if c != 'Price'])
            fig = px.scatter(df, 
                            x=num_feat, 
                            y='Price', 
                            trendline="lowess",
                            title=f"Price vs {num_feat}")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

if __name__ == "__main__":
    main()
