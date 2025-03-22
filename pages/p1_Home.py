# pages/1_Home.py
import streamlit as st

def main():
    st.title("Project Overview")
    st.markdown("""
    ## Weather conditions effect on Electriciy Price Analysis
    
    **Dataset:** Nordpool Spot electric prices in Finland X Historical Weather data
    
    **Objective:** Study what kind of relationship weather conditions have with electricity prices in Finland.
    """)
    
if __name__ == "__main__":
    main()
