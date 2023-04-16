import sys
sys.path.append("C:\\Users\\Saad\\Desktop\\automl\\source")

import base64
import random
import traceback
import itertools
from math import sqrt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from contextlib import contextmanager

from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

from components.utils.logger import logging
from components.utils.exception import CustomException

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ModelTrainer:

    @contextmanager
    def st_exceptions(self):
        try:
            yield
        except Exception as e:
            st.error(str(e))
            st.error(traceback.format_exc())


    def get_binary_file_downloader(self, filename, data):
        b64 = base64.b64encode(data).decode() 
        href = f'<a href="data:file/octet-stream;base64,{b64}" download="{filename}">Download Trained Model</a>'
        return href
    
    def calculate_search_space_size(self, param_grid):
        return len(list(itertools.product(*param_grid.values())))

    def calculate_rmse(self, y_true, y_pred):
        self.mse = mean_squared_error(y_true, y_pred)
        self.rmse = sqrt(self.mse)
        return self.rmse

    def display_model_results(self, model, X_test, y_test, target_col):
        y_pred = model.predict(X_test)
        rmse = self.calculate_rmse(y_test, y_pred)
        st.write(f"RMSE: {rmse:.2f}")

        fig = px.line(x=X_test.index, y=[y_test, y_pred], labels={'x': 'Time', 'y': 'RAM Usage'}, title=f"Actual vs Predicted {target_col}")
        fig.update_layout(showlegend=True)
        fig.update_traces(line=dict(width=2))
        fig.add_scatter(x=X_test.index, y=y_test, mode='lines', name="Actual", line=dict(color='white'))
        fig.add_scatter(x=X_test.index, y=y_pred, mode='lines', name="Predicted", line=dict(color='orange'))
        st.plotly_chart(fig)

    def split_data(self, df, target_col, test_size):
        with self.st_exceptions():
            X = df.drop(target_col, axis=1)
            y = df[target_col]

            n_splits = int(1 / test_size)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            return X_train, X_test, y_train, y_test
        return None, None, None, None

    def tune_hyperparameters(self, model, X_train, y_train, param_grid, search_method, n_iter=None, cv=5, scoring=None):
        if search_method == "RandomizedSearchCV":
            search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=n_iter, cv=cv, n_jobs=-1, random_state=42)
        elif search_method == "GridSearchCV":
            search = GridSearchCV(model, param_grid, cv=cv, n_jobs=-1, scoring=make_scorer(scoring) if scoring else None)
        else:
            model.fit(X_train, y_train)
            return model, None
        search.fit(X_train, y_train)
        return search.best_estimator_, search.best_params_

    def create_random_param_grid(self, model_choice):
        random_param_grid = {}
        
        if model_choice in ["Ridge Regression", "Lasso Regression", "Elastic Net"]:
            random_param_grid['alpha'] = sorted([random.uniform(0.01, 1.5) for _ in range(10)])

        if model_choice == "Random Forest" or model_choice == "Gradient Boosting":
            random_param_grid['n_estimators'] = sorted([random.randint(10, 1000) for _ in range(10)])
            random_param_grid['max_depth'] = sorted([random.randint(1, 50) for _ in range(10)])

        if model_choice == "XGBoost":
            random_param_grid['learning_rate'] = sorted([random.uniform(0.01, 0.5) for _ in range(10)])
            random_param_grid['max_depth'] = sorted([random.randint(1, 20) for _ in range(10)])
            random_param_grid['n_estimators'] = sorted([random.randint(10, 1000) for _ in range(10)])
        return random_param_grid

    def train_all_models(self, X_train, X_test, y_train, y_test):
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Elastic Net": ElasticNet(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor()
        }

        best_model_name = None
        best_model = None
        best_rmse = float("inf")

        for model_name, model in models.items():
                with self.st_exceptions():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    rmse = self.calculate_rmse(y_test, y_pred)

                    if rmse < best_rmse:
                        best_model = model
                        best_rmse = rmse
                        best_model_name  = model_name
        return best_model_name, best_model, best_rmse