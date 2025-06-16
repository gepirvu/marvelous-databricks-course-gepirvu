"""Model training module for LightGBM regression on insurance data.

Implements hyperparameter tuning, final model training, and evaluation logic.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

from insurance.config import ProjectConfig


class ModelTrainer:
    """Trains and evaluates a LightGBM regression model based on project config."""

    def __init__(self, df: pd.DataFrame, config: ProjectConfig) -> None:
        self.df = df
        self.config = config
        self.model = lgb.LGBMRegressor(force_col_wise=True)
        self.best_model = None

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the internal DataFrame into training and test sets.

        Returns:
            tuple: A tuple containing:
                - X_train (pd.DataFrame): Training features
                - X_test (pd.DataFrame): Testing features
                - y_train (pd.Series): Training target
                - y_test (pd.Series): Testing target

        """
        X = self.df.drop(columns=[self.config.target, "Id"])
        y = self.df[self.config.target]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[dict, float]:
        """Perform grid search to tune LightGBM hyperparameters.

        Args:
            X_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Training target values.

        Returns:
            tuple: A dictionary of the best parameters and the best R² score.

        """
        param_grid = {
            "learning_rate": self.config.parameters["learning_rate"],
            "n_estimators": self.config.parameters["n_estimators"],
            "num_leaves": self.config.parameters["num_leaves"],
        }

        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, scoring="r2", cv=5)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        return grid_search.best_params_, grid_search.best_score_

    def train_final_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> LGBMRegressor:
        """Train the final LightGBM model using the best hyperparameters.

        Args:
            X_train (pd.DataFrame): Training feature set.
            y_train (pd.Series): Training target values.

        Returns:
            Any: The trained LightGBM model.

        """
        self.best_model.fit(X_train, y_train, feature_name=X_train.columns.tolist())
        return self.best_model

    def evaluate(
        self,
        y_true: np.ndarray | pd.Series | list[float],
        y_pred: np.ndarray | pd.Series | list[float],
    ) -> dict[str, float]:
        """Evaluate the model using regression metrics.

        Args:
            y_true (Union[np.ndarray, pd.Series, list[float]]): True target values.
            y_pred (Union[np.ndarray, pd.Series, list[float]]): Predicted target values.

        Returns:
            dict[str, float]: Dictionary with evaluation metrics such as R², RMSE, and MAE.

        """
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
        }
