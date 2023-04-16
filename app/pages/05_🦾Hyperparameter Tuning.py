import ast
import time
import pickle
import streamlit as st
import xgboost as xgb

from contextlib import contextmanager
from mt_streamlit import ModelTrainer
from components.utils.logger import logging
from components.utils.exception import CustomException

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

model_trainer = ModelTrainer()
st.header("Model Training with Tuning")
st.divider()

st.write("In this section, you can train different regression models on your preprocessed dataset with hyperparameter tuning using built-in methods like RandomizedSearchCV and GridSearchCV.")

if 'preprocessed_df' in st.session_state:
    preprocessed_df = st.session_state.preprocessed_df

    st.sidebar.subheader("Train-Test Split")
    target_col = st.sidebar.selectbox("Select the target column", preprocessed_df.columns)
    test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.1)

    if 'split_data_done' not in st.session_state:
        X_train, X_test, y_train, y_test = model_trainer.split_data(preprocessed_df, target_col, test_size)
        st.session_state.split_data_done = True
        st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = X_train, X_test, y_train, y_test

    if 'split_data_done' in st.session_state:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test

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

        st.sidebar.subheader("Parameter Grid")
        user_param_grid = st.sidebar.text_area("Enter parameter grid (dictionary format):")
        if user_param_grid:
            try:
                user_param_grid = ast.literal_eval(user_param_grid)
            except ValueError as e:
                st.sidebar.error("Invalid parameter grid. Please enter a valid dictionary.")

        if st.sidebar.button("Generate Parameter Grid"):
            generated_param_grid = model_trainer.create_random_param_grid(model_choice)
            st.sidebar.write(f"Generated parameter grid for {model_choice}:")
            st.sidebar.write(generated_param_grid)

        st.markdown(f'Current Test Size: <span style="color:orange;">{test_size:.2f}</span>', unsafe_allow_html=True)
        st.markdown(f'Currect Selected Model: <span style="color:aqua;">{model_choice}</span>', unsafe_allow_html=True)

        st.divider()

        training_tabs = st.selectbox("Select Training Option", ["Tuning with RandomizedSearchCV", "Tuning with GridSearchCV"])

        if training_tabs == "Tuning with RandomizedSearchCV":
            n_iter = st.slider("Number of Iterations", 10, 100, 20, 5)
            cv = st.slider("Number of Cross-Validation Folds", 2, 10, 5, 1)

            with st.expander("Tuning with RandomizedSearchCV Info"):
                st.write("This tab allows you to train the selected model with hyperparameter tuning using RandomizedSearchCV. You can adjust the test size using the slider in the sidebar.")
            if st.button("Train Model with RandomizedSearchCV"):
                start_time = time.time()
                with st.spinner("Looking for Best Parameters & Training the model with RandomizedSearchCV"):
                    model, best_params = model_trainer.tune_hyperparameters(models[model_choice], X_train, y_train, user_param_grid or generated_param_grid, "RandomizedSearchCV", n_iter=n_iter, cv=cv)
                    st.success(f"{model_choice} trained with RandomizedSearchCV successfully. Time taken: {time.time() - start_time:.2f} seconds")
                    st.write(f"Best parameters: {best_params}")

                    model_trainer.display_model_results(model, X_test, y_test, target_col)

                    model_binary = pickle.dumps(model)
                    st.download_button(
                        label="Download Trained Model",
                        data=model_binary,  
                        file_name=f"{model_choice}_trained_model_RandomizedSearchCV.pkl",
                        mime="application/octet-stream",
                    )

        elif training_tabs == "Tuning with GridSearchCV":
            cv = st.slider("Number of Cross-Validation Folds", 2, 10, 5, 1)
            with st.expander("Tuning with GridSearchCV Info"):
                st.write("This tab allows you to train the selected model with hyperparameter tuning using GridSearchCV. You can adjust the test size using the slider in the sidebar.")
            if st.button("Train Model with GridSearchCV"):
                start_time = time.time()
                with st.spinner("Looking for Best Parameters & Training the model with GridSearchCV..."):
                    model, best_params = model_trainer.tune_hyperparameters(models[model_choice], X_train, y_train, user_param_grid or generated_param_grid, "GridSearchCV", cv=cv)
                    st.success(f"{model_choice} trained with GridSearchCV successfully. Time taken: {time.time() - start_time:.2f} seconds")
                    st.write(f"Best parameters: {best_params}")

                    model_trainer.display_model_results(model, X_test, y_test, target_col)

                    model_binary = pickle.dumps(model)
                    st.download_button(
                        label="Download Trained Model",
                        data=model_binary,
                        file_name=f"{model_choice}_trained_model_GridSearchCV.pkl",
                        mime="application/octet-stream",
                    )

else:
    st.warning(f"Preprocessed Data Required!")
