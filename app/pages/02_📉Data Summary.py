import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report



st.header("Exploratory Data Analysis")
st.divider()
st.write("This section provides an overview of your dataset using pandas_profiling. It will help you understand the data distribution, missing values, correlations, and other statistics.")

if "data_name" in st.session_state:
    st.markdown(f"Loaded Data: <span style='color:orange;'>{st.session_state.data_name}</span>", unsafe_allow_html=True)
else:
    st.session_state.data_name = "Not loaded"
    st.markdown(f"Loaded Data: <span style='color:orange;'>{st.session_state.data_name}</span>", unsafe_allow_html=True)

st.divider()
if 'df' in st.session_state:
    profile_df = ProfileReport(st.session_state.df)
    st_profile_report(profile_df)
else:
    st.warning("Please upload a dataset first.")


