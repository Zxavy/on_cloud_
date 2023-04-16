import json
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_extras.badges import badge

st.set_page_config(page_title="AutoML", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url = "https://assets5.lottiefiles.com/packages/lf20_9qN1iLEfN5.json"
lottie_json = load_lottieurl(lottie_url)



col1, col2, col3= st.columns(3)

with col3:
    title_animation = st_lottie(lottie_json,
                            speed=1,
                            reverse=False,
                            loop=True,
                            quality="high",
                            height=150,
                            width=550,
                            key=None,
                    )

with  col1:
    st.header("AutoML")
    st.write("A tool for automating machine learning pipeline.")
    st.text("Author: Saad Ahmad")
    badge(type='github', url='https://github.com/Zxavy')






st.divider()
st.header("Summary")

st.write("""
Welcome to AutoML! This application is designed to help you quickly build and evaluate regression models for time-series data, specifically focusing on memory usage. The application guides you through the process of loading your data, profiling it, preprocessing it, training models, and visualizing the results.
""")
st.write("## 📚 Learn More about Machine Learning")
st.write("""
If you're new to machine learning or want to learn more about regression and time-series data, check out the following resources:
- [Introduction to Machine Learning](https://www.youtube.com/watch?v=Gv9_4yMHFhI)
- [A Gentle Introduction to Regression](https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/)
- [A Comprehensive Guide to Time Series Analysis](https://towardsdatascience.com/a-comprehensive-guide-to-time-series-analysis-8e4432e3ff95)
""")

st.write("## 🤖 Models Used in this Application")
st.write("""
This application supports several regression models, including:
- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Random Forest (in works...)
- Gradient Boosting
- XGBoost (in works...)
""")

st.write("## ⏳ Time-Series Data and its Components")
st.write("""
Time-series data is a type of data that is collected at regular time intervals. The primary components of time-series data include:
- Trend: The overall direction of the data over time (upward, downward, or stationary)
- Seasonality: Regular, predictable fluctuations in the data that occur over a specific time period (daily, weekly, yearly, etc.)
- Noise: Random variations in the data that are not part of the underlying trend or seasonality
""")

st.write("## 🧭 Application Navigation")
st.write("""
Use the sidebar to navigate through the different steps of the application:
1. Load Data: Upload your time-series data file (CSV format)
2. Data Profiling: Explore your data and visualize its statistical properties
3. Data Preprocessing: Clean and preprocess your data for better model performance
4. Model Training: Train and tune regression models on your preprocessed data
5. Model Evaluation: Evaluate the performance of your trained models
6. Visualization: Visualize the predictions of your best-performing model
""")



lottie_url_2 = "https://assets5.lottiefiles.com/packages/lf20_DXljHQsLLA.json"
lottie_json_2 = load_lottieurl(lottie_url_2)
end_animation = st_lottie(lottie_json_2,
                        speed=1,
                        reverse=True,
                        loop=True,
                        quality="high",
                        height=550,
                        width=550,
                        key=None,
                )