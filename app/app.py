import streamlit as st
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils.exception import CustomException
from raw_preprocessing import DataPreprocessor
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import random
import traceback
from contextlib import contextmanager
import ast
import time
import pickle
import base64
import os
import glob
from mt_streamlit import ModelTrainer



@contextmanager
def st_exceptions():
    try:
        yield
    except Exception as e:
        st.error(str(e))
        st.error(traceback.format_exc())

def load_data(file):
    with st_exceptions():
        df = pd.read_csv(file, index_col=None)
    return df

def load_data_from_folder():
    folder_path = "./artifacts/data/sample_data"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    if not csv_files:
        st.warning("No CSV files found in the specified folder.")
        return None

    dataset_options = ["None"] + [os.path.basename(f) for f in csv_files]
    dataset_choice = st.selectbox("Select a dataset from the folder", dataset_options)

    if dataset_choice != "None":
        with st.spinner("Loading dataset..."):
            df = pd.read_csv(os.path.join(folder_path, dataset_choice))
            st.session_state.df = df
            st.dataframe(df)
    else:
        st.warning("Please select a dataset from the folder.")


def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    return rmse

def display_model_results(model, X_test, y_test, target_col):
    y_pred = model.predict(X_test)
    rmse = calculate_rmse(y_test, y_pred)
    st.write(f"RMSE: {rmse:.2f}")

    fig = px.line(x=X_test.index, y=[y_test, y_pred], labels={'x': 'Time', 'y': 'RAM Usage'}, title=f"Actual vs Predicted {target_col}")
    fig.update_layout(showlegend=True)
    fig.update_traces(line=dict(width=2))
    fig.add_scatter(x=X_test.index, y=y_test, mode='lines', name="Actual", line=dict(color='white'))
    fig.add_scatter(x=X_test.index, y=y_pred, mode='lines', name="Predicted", line=dict(color='orange'))
    st.plotly_chart(fig)

def create_random_param_grid(model_choice):
    random_param_grid = {}
    
    if model_choice in ["Ridge Regression", "Lasso Regression", "Elastic Net"]:
        random_param_grid['alpha'] = sorted([random.uniform(0.01, 1.0) for _ in range(5)])

    if model_choice == "Random Forest" or model_choice == "Gradient Boosting":
        random_param_grid['n_estimators'] = sorted([random.randint(10, 500) for _ in range(5)])
        random_param_grid['max_depth'] = sorted([random.randint(1, 50) for _ in range(5)])

    if model_choice == "XGBoost":
        random_param_grid['learning_rate'] = sorted([random.uniform(0.01, 0.5) for _ in range(5)])
        random_param_grid['max_depth'] = sorted([random.randint(1, 20) for _ in range(5)])
        random_param_grid['n_estimators'] = sorted([random.randint(10, 500) for _ in range(5)])
    return random_param_grid


def split_data(df, target_col, test_size):
    with st_exceptions():
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        n_splits = int(1 / test_size)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        return X_train, X_test, y_train, y_test
    return None, None, None, None

def tune_hyperparameters(model, X_train, y_train, param_grid, search_method, n_iter=None):
    if search_method == "RandomizedSearchCV":
        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=5, n_jobs=-1, random_state=42)
    elif search_method == "GridSearchCV":
        search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    else:
        model.fit(X_train, y_train)
        return model, None

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def train_models_and_get_best(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_rmse = float('inf')
    best_name = ""

    for model_name, model in models.items():
        with st_exceptions():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = calculate_rmse(y_test, y_pred)

            if rmse < best_rmse:
                best_model = model
                best_rmse = rmse
                best_name = model_name
    return best_model, best_rmse, best_name

def get_binary_file_downloader(filename, data):
    b64 = base64.b64encode(data).decode() 
    href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">Download Trained Model</a>'
    return href

df = None
st.title("AutoML")
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Select an option", ["Upload", "Profiling", "Preprocessing", "Model Training"])

if choice == "Upload":
    st.header("Upload Your Dataset")
    st.divider()
    st.write("Please upload your dataset in CSV format or select a dataset from a folder located outside the Streamlit application folder. The dataset will be stored in the session state and used for further processing.")
    
    load_data_from_folder()
    st.divider()
    file = st.file_uploader("Or, upload your own dataset (CSV format)")
    if file:
        with st_exceptions():
            df = load_data(file)
            st.session_state.df = df
            df.to_csv('dataset.csv', index=None)
            st.dataframe(df)

elif choice == "Profiling":
    st.header("Exploratory Data Analysis")
    st.divider()
    st.write("This section provides an overview of your dataset using pandas_profiling. It will help you understand the data distribution, missing values, correlations, and other statistics.")
    st.divider()
    if 'df' in st.session_state:
        profile_df = ProfileReport(st.session_state.df)
        st_profile_report(profile_df)
    else:
        st.warning("Please upload a dataset first.")

elif choice == "Preprocessing":
    st.header("Data Preprocessing")
    st.divider()
    st.write("In this section, you can preprocess your data. It is done using a custom preprocessing script. After preprocessing, a pandas_profiling report will be generated for the preprocessed data.")
    st.divider()
    if 'df' in st.session_state:
        df = st.session_state.df
        preprocessor = DataPreprocessor(df)

        if st.button("Start Preprocessing"):
            start_time = time.time()
            progress_bar = st.progress(0)
            for i in range(100):
                progress_bar.progress(i + 1, text="Preprocessing Dataset")
            preprocessed_df = preprocessor.preprocess_data()
            st.write("Preprocessed data:")
            st.dataframe(preprocessed_df)
            st.session_state.preprocessed_df = preprocessed_df
            st.success(f"Preprocessed data saved successfully. Time taken: {time.time() - start_time:.2f} seconds")
            
                # st.subheader("Pandas Profiling for Preprocessed Data")
                # preprocessed_profile = ProfileReport(preprocessed_df)
                # st_profile_report(preprocessed_profile)


elif choice == "Model Training":
    st.header("Model Training")
    st.divider()
    st.write("In this section, you can train different regression models on your preprocessed dataset. You can also tune the hyperparameters using built-in methods like RandomizedSearchCV and GridSearchCV.")

    if 'preprocessed_df' in st.session_state:
        preprocessed_df = st.session_state.preprocessed_df

        #st.write(f"Preprocessed Data Loaded: {preprocessed_df.shape[0]} rows x {preprocessed_df.shape[1]} columns")

        st.sidebar.subheader("Train-Test Split")
        target_col = st.sidebar.selectbox("Select the target column", preprocessed_df.columns)
        test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.1)

        #st.write("The Train-Test Split slider helps you adjust the ratio of data used for training and testing. A smaller test size means more data will be used for training.")

        if 'split_data_done' not in st.session_state:
            X_train, X_test, y_train, y_test = split_data(preprocessed_df, target_col, test_size)
            st.session_state.split_data_done = True
            st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = X_train, X_test, y_train, y_test

        if 'split_data_done' in st.session_state:
            X_train, X_test, y_train, y_test = st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test
            X_train = X_train.astype(float)
            X_test = X_test.astype(float)

        st.sidebar.subheader("Model Selection")
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Elastic Net": ElasticNet(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": xgb.XGBRegressor(objective ='reg:squarederror')
        }
        model_choice = st.sidebar.selectbox("Select the model", list(models.keys()))

        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = model_choice
        elif st.session_state.selected_model != model_choice:
            st.session_state.selected_model = model_choice
            st.session_state.randomsearch_iter_done = False

        st.sidebar.subheader("Model Parameters")
        model = models[model_choice]

        st.markdown(f'Current Test Size: <span style="color:orange;">{test_size:.2f}</span>', unsafe_allow_html=True)
        st.markdown(f'Currect Selected Model: <span style="color:aqua;">{model_choice}</span>', unsafe_allow_html=True)
        #st.markdown(f'Currect Dataset: <span style="color:aqua;">{st.session_state.dataset_name}</span>', unsafe_allow_html=True)

        st.divider()

        if model_choice in ["Ridge Regression", "Lasso Regression", "Elastic Net"]:
            alpha = st.sidebar.slider("Alpha", 0.0, 1.0, 0.5, 0.1)
            model.set_params(alpha=alpha)

        if model_choice == "Random Forest" or model_choice == "Gradient Boosting":
            n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100, 10)
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5, 1)
            model.set_params(n_estimators=n_estimators, max_depth=max_depth)

        if model_choice == "XGBoost":
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, 0.01)
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5, 1)
            n_estimators = st.sidebar.slider("Number of Estimators", 10, 500, 100, 10)
            model.set_params(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators)


        st.sidebar.subheader("Hyperparameter Tuning")
        tuning_choice = st.sidebar.selectbox("Select tuning method", ["None", "RandomizedSearchCV", "GridSearchCV"])
        param_grid = {}

        #st.write("Hyperparameter tuning helps find the best model parameters. You can choose between 'RandomizedSearchCV', 'GridSearchCV' or 'None' for no tuning.")

        if tuning_choice != "None":
            user_input = st.sidebar.text_area("Enter param_grid as a dictionary (e.g., {'alpha': [0.1, 1.0]})", height=150)

            try:
                param_grid = ast.literal_eval(user_input)
            except:
                st.sidebar.warning("Invalid input. Please enter a valid dictionary.")

        if tuning_choice == "RandomizedSearchCV":
            n_iter = st.sidebar.slider("Number of Iterations", 10, 100, 20, 5)
            if 'randomsearch_iter_done' not in st.session_state:
                st.session_state.randomsearch_iter_done = True
                st.session_state.n_iter = n_iter

        model, best_params = tune_hyperparameters(model, X_train, y_train, param_grid, tuning_choice, n_iter if tuning_choice == "RandomizedSearchCV" else None)

        training_tabs = st.selectbox("Select Training Option", ["Basic Model Training", "Training with Tuning", "Train All Models"])

        
        if training_tabs == "Basic Model Training":
            st.write("Basic Model Training:")
            st.write("This button will train the selected model using the default or specified parameters without any hyperparameter tuning.")
            if st.button("Basic Model Training"):
                start_time = time.time()
                with st.spinner("Training the basic model..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.1)
                        progress_bar.progress(i + 1, text=f"Training {model_choice} model:")
                    model.fit(X_train, y_train)
                    st.session_state.model = model
                    st.success(f"{model_choice} trained successfully. Time taken: {time.time() - start_time:.2f} seconds")
                    display_model_results(model, X_test, y_test, target_col)

                    model_binary = pickle.dumps(st.session_state.model)
                    st.download_button(
                        label="Download Trained Model",
                        data=model_binary,
                        file_name=f"{model_choice}_trained_model.pkl",
                        mime="application/octet-stream",
                    )

        elif training_tabs == "Training with Tuning":
            st.write("Training with Tuning:")
            st.write("This button will train the selected model using hyperparameter tuning with the specified method (RandomizedSearchCV or GridSearchCV).")
     
            if st.button("Start Training with Tuning", disabled=False):
                start_time = time.time()
                with st.spinner("Training with hyperparameter tuning..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.1)
                        progress_bar.progress(i + 1, text="Looking for Best Parameters & Training:")
                    model, best_params = tune_hyperparameters(model, X_train, y_train, param_grid, tuning_choice, n_iter if tuning_choice == "RandomizedSearchCV" else None)
                    st.success(f"Hyperparameter tuning complete. Best parameters: {best_params}. Time taken: {time.time() - start_time:.2f} seconds")

                    st.session_state.model = model
                    st.success(f"{model_choice} trained successfully.")
                    display_model_results(model, X_test, y_test, target_col)
                    
                    model_binary = pickle.dumps(st.session_state.model)
                    st.download_button(
                        label="Download Trained Model",
                        data=model_binary,
                        file_name=f"{model_choice}_trained_model.pkl",
                        mime="application/octet-stream",
                    )
            

            st.write("Generate Random Parameter Grid:")
            st.write("This button will generate a random parameter grid for the selected model. Use this as a starting point for hyperparameter tuning.")
            if st.button("Generate Random Parameter Grid", disabled=False):
                start_time = time.time()
                with st.spinner("Generating random parameter grid..."):
                    param_grid = create_random_param_grid(model_choice)
                    st.write(f"Generated Random Parameter Grid for {model_choice}:")
                    st.write(param_grid)
                    st.success(f"Random parameter grid generated. Time taken: {time.time() - start_time:.2f} seconds")

                    
        elif training_tabs == "Train All Models":
            st.write("Train All Models:")
            st.write("Training all models with default hyperparameters...")
            if st.button("Train All Models", disabled=False):
                start_time = time.time()
                with st.spinner("Training all models..."):
                    model_trainer = ModelTrainer()
                    best_model_name, best_model, best_rmse = model_trainer.train_all_models(X_train, X_test, y_train, y_test)
       
                    st.success(f"Best Model: {best_model_name} with RMSE: {best_rmse:.2f}. Time taken: {time.time() - start_time:.2f} seconds")

                    st.write(f"Results for the best model, {best_model_name}, on the test dataset:")
                    display_model_results(best_model, X_test, y_test, target_col)

                    model_binary = pickle.dumps(best_model)
                    st.download_button(
                        label="Download Trained Model",
                        data=model_binary,
                        file_name=f"{best_model_name.replace(' ', '_')}_trained_model.pkl",
                        mime="application/octet-stream",
                        key=f"download_button_{best_model_name.replace(' ', '_')}"
                    )


        else:
            st.warning("Please preprocess the dataset first.")

       









