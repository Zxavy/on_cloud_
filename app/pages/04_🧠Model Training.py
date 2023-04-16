import ast
import time
import pickle
import streamlit as st
import xgboost as xgb

import pandas as pd
from contextlib import contextmanager
from mt_streamlit import ModelTrainer
from components.utils.logger import logging
from components.utils.exception import CustomException

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


model_trainer = ModelTrainer()
st.header("Model Training")
st.divider()


st.write("In this section, you can train different regression models on your preprocessed dataset. You can also tune the hyperparameters using built-in methods like RandomizedSearchCV and GridSearchCV.")

if 'preprocessed_df' in st.session_state:
    preprocessed_df = st.session_state.preprocessed_df
    st.markdown(f"Preprocessing: <span style='color:green;'>&#x2714;</span>", unsafe_allow_html=True)

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
            "None": None,
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Elastic Net": ElasticNet(),
            "Random Forest": RandomForestRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": xgb.XGBRegressor(objective ='reg:squarederror')
        }
        model_choice = st.sidebar.selectbox("Select the model", list(models.keys()))

        st.markdown(f'Current Test Size: <span style="color:orange;">{test_size:.2f}</span>', unsafe_allow_html=True)
        st.markdown(f'Currect Selected Model: <span style="color:aqua;">{model_choice}</span>', unsafe_allow_html=True)

        st.divider()

        training_tabs = st.selectbox("Select Training Option", ["Basic Model Training", "Train All Models", "Model Report"])
        if training_tabs == "Basic Model Training":
            with st.expander("Basic Model Training Info"):
                st.write("This tab allows you to train the selected model using the default or specified parameters without any hyperparameter tuning. You can adjust the test size using the slider in the sidebar.")
       
            if model_choice == "Ridge Regression":
                alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
                models[model_choice].set_params(alpha=alpha)
            elif model_choice == "Lasso Regression":
                alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
                models[model_choice].set_params(alpha=alpha)
            elif model_choice == "Elastic Net":
                alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
                l1_ratio = st.slider("L1 Ratio", 0.1, 1.0, 0.5)
                models[model_choice].set_params(alpha=alpha, l1_ratio=l1_ratio)
            elif model_choice == "Random Forest":
                n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                max_depth = st.slider("Max Depth", 1, 50, 10)
                models[model_choice].set_params(n_estimators=n_estimators, max_depth=max_depth)
            elif model_choice == "Gradient Boosting":
                n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
                max_depth = st.slider("Max Depth", 1, 50, 3)
                models[model_choice].set_params(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            elif model_choice == "XGBoost":
                n_estimators = st.slider("Number of Estimators", 10, 200, 100)
                learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
                max_depth = st.slider("Max Depth", 1, 50, 3)
                models[model_choice].set_params(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
            
            
            if st.button("Train Model"):
                start_time = time.time()
                with st.spinner("Training the basic model..."):
                    progress_bar = st.progress(0)
                    for i in range(100):
                        progress_bar.progress(i + 1, text=f"Training {model_choice} model:")

                    model = models[model_choice]
                    model.fit(X_train, y_train)

                    st.session_state.model = model
                    st.session_state.selected_model_name = model_choice
                    
                    st.success(f"{model_choice}trained successfully. Time taken: {time.time() - start_time:.2f} seconds")
                    model_trainer.display_model_results(model, X_test, y_test, target_col)

                    model_binary = pickle.dumps(st.session_state.model)
                    st.download_button(
                        label="Download Trained Model",
                        data=model_binary,
                        file_name=f"{model_choice}_trained_model.pkl",
                        mime="application/octet-stream",
                    )

        elif training_tabs == "Train All Models":
            with st.expander("Train All Models Info"):
                st.write("This tab allows you to train all available models with default hyperparameters. The model with the best performance (lowest RMSE) will be displayed along with its results on the test dataset.")
            if st.button("Train All Models"):
                start_time = time.time()
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.1)
                    progress_bar.progress(i + 1, text="Training all models with default hyperparameters...")
                model_trainer = ModelTrainer()
                best_model_name, best_model, best_rmse = model_trainer.train_all_models(X_train, X_test, y_train, y_test)

                st.session_state.model = best_model
                st.session_state.selected_model_name = best_model_name
                st.success(f"Best Model: {best_model_name} with RMSE: {best_rmse:.2f}. Time taken: {time.time() - start_time:.2f} seconds")

                st.write(f"Results for the best model {best_model_name} on the test dataset:")
                model_trainer.display_model_results(best_model, X_test, y_test, target_col)

                model_binary = pickle.dumps(best_model)
                st.download_button(
                    label="Download Trained Model",
                    data=model_binary,
                    file_name=f"{best_model_name.replace(' ', '_')}_trained_model.pkl",
                    mime="application/octet-stream",
                    key=f"download_button_{best_model_name.replace(' ', '_')}"
                )

        elif training_tabs == "Model Report":
            if "model" in st.session_state:
                model = st.session_state.model
                selected_model_name = st.session_state.selected_model_name 
                st.header("Model Report")
                st.markdown(f"Model: <span style='color:aqua;'>{selected_model_name}</span>", unsafe_allow_html=True) 
                        
                with st.expander("Model Parameters"):
                    st.write(model.get_params())
                        
                with st.expander("Feature Importance"):
                    if hasattr(model, "feature_importances_"):
                        feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
                        st.write(feature_importances)
                        st.bar_chart(feature_importances)
                    else:
                        st.warning("This model does not support feature importance.")
            else:
                st.warning("No model has been trained yet. Please train a model first.")


else:
    st.warning(f"Preprocessed Data Required!")




