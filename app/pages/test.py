import streamlit as st
from easygoogletranslate import EasyGoogleTranslate

st.set_page_config(page_title="AutoML", layout="wide")
st.title("AutoML: Automated Machine Learning for Regression and Time-Series Data")


translator = EasyGoogleTranslate(source_language='en',target_language='de',timeout=10)

# Function to translate text to German
def translate_text(text, source_language='en', target_language='de'):
    result = translator.translate(text, source_language=source_language, target_language=target_language)
    return result

def toggle_language(language):
    if language == 'en':
        st.session_state.language = 'de'
        st.session_state.button_label = 'Translate to English'
    else:
        st.session_state.language = 'en'
        st.session_state.button_label = 'Translate to German'
    st.experimental_rerun()


if 'language' not in st.session_state:
    st.session_state.language = 'en'

if 'button_label' not in st.session_state:
    st.session_state.button_label = 'Translate to German'

# Recreate the button after rerun
translate_button = st.button(st.session_state.button_label)

if translate_button:
    toggle_language(st.session_state.language)



if st.session_state.language == 'de':
    st.title(translate_text('AutoML: Automatisiertes maschinelles Lernen für Regression und Zeitreihendaten'))
    st.write(translate_text("""
    Willkommen bei AutoML! Diese Anwendung soll Ihnen dabei helfen, schnell Regression-Modelle für Zeitreihendaten aufzubauen und zu bewerten, wobei der Schwerpunkt auf dem Speicherverbrauch liegt. Die Anwendung führt Sie durch den Prozess des Ladens Ihrer Daten, der Profilerstellung, der Vorverarbeitung, des Trainings von Modellen und der Visualisierung der Ergebnisse.
    """))
    st.write('## 📚 Mehr zum Thema Maschinelles Lernen erfahren')
    st.write(translate_text("""
    Wenn Sie neu im Bereich des maschinellen Lernens sind oder mehr über Regression und Zeitreihendaten erfahren möchten, werfen Sie einen Blick auf die folgenden Ressourcen:
    - Einführung ins maschinelle Lernen
    - Eine sanfte Einführung in Regression
    - Eine umfassende Anleitung zur Zeitreihenanalyse
    """))
    st.write('## 🤖 In dieser Anwendung verwendete Modelle')
    st.write(translate_text("""
    Diese Anwendung unterstützt mehrere Regressionsmodelle, darunter:
    - Lineare Regression
    - Ridge-Regression
    - Lasso-Regression
    - Elastic Net
    - Zufälliger Wald (in Arbeit...)
    - Gradienten-Boosting
    - XGBoost (in Arbeit...)
    """))
    st.write('## ⏳ Zeitreihendaten und ihre Komponenten')
    st.write(translate_text("""
    Zeitreihendaten sind eine Art von Daten, die in regelmäßigen Zeitabständen erhoben werden. Die wichtigsten Komponenten von Zeitreihendaten sind:
    - Trend: Die Gesamtrichtung der Daten im Laufe der Zeit (aufwärts, abwärts oder stationär)
    - Saisonalität: Regelmäßige, vorhersagbare Schwankungen in den Daten, die über einen bestimmten Zeitraum auftreten (täglich, wöchentlich, jährlich usw.)
    - Rauschen: Zufällige Variationen in den Daten, die nicht Teil des zugrunde liegenden Trends oder der Saisonalität sind
    """))

    st.write('## 🧭 Navigation in der Anwendung')
    st.write(translate_text("""
    Verwenden Sie die Seitenleiste, um durch die verschiedenen Schritte der Anwendung zu navigieren:
    1. Daten laden: Laden Sie Ihre Zeitreihendatendatei hoch (CSV-Format)
    2. Datenprofilierung: Untersuchen Sie Ihre Daten und visualisieren Sie ihre statistischen Eigenschaften
    3. Daten Vorverarbeitung: Reinigen und vorverarbeiten Sie Ihre Daten für eine bessere Modellleistung
    4. Modelltraining: Trainieren und optimieren Sie Regressionsmodelle auf Ihren vorverarbeiteten Daten
    5. Modellbewertung: Bewertung der Leistung Ihrer trainierten Modelle
    6. Visualisierung: Visualisierung der Vorhersagen Ihres bestleistenden Modells
    """))

else:
    st.write("""
    Welcome to AutoML! This application is designed to help you quickly build and evaluate regression models for time-series data, specifically focusing on memory usage. The application guides you through the process of loading your data, profiling it, preprocessing it, training models, and visualizing the results.
    """)
    st.write('## 📚 Learn More about Machine Learning')
    st.write("""
    If you're new to machine learning or want to learn more about regression and time-series data, check out the following resources:
    - Introduction to Machine Learning
    - A Gentle Introduction to Regression
    - A Comprehensive Guide to Time Series Analysis
    """)
    st.write('## 🤖 Models Used in this Application')
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
    st.write('## ⏳ Time-Series Data and its Components')
    st.write("""
    Time-series data is a type of data that is collected at regular time intervals. The primary components of time-series data include:
    - Trend: The overall direction of the data over time (upward, downward, or stationary)
    - Seasonality: Regular, predictable fluctuations in the data that occur over a specific time period (daily, weekly, yearly, etc.)
    - Noise: Random variations in the data that are not part of the underlying trend or seasonality
    """)
    st.write('## 🧭 Application Navigation')
    st.write("""
    Use the sidebar to navigate through the different steps of the application:
    1. Load Data: Upload your time-series data file (CSV format)
    2. Data Profiling: Explore your data and visualize its statistical properties
    3. Data Preprocessing: Clean and preprocess your data for better model performance
    4. Model Training: Train and tune regression models on your preprocessed data
    5. Model Evaluation: Evaluate the performance of your trained models
    6. Visualization: Visualize the predictions of your best-performing model
    """)

    