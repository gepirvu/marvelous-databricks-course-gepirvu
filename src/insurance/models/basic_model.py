"""Basic model implementation for house price prediction.

This module includes:
- Training and evaluating a LightGBM model using sklearn pipelines.
- Logging models and datasets with MLflow.
- Registering the model in Unity Catalog.
- Loading and predicting with the latest model version.
"""

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow import MlflowClient
from mlflow.data.dataset_source import DatasetSource
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from insurance.config import ProjectConfig, Tags


class BasicModel:
    """LightGBM model trainer for house price prediction."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        self.config = config
        self.spark = spark

        self.num_features = self.config.num_features
        self.cat_features = self.config.cat_features
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic
        self.model_name = f"{self.catalog_name}.{self.schema_name}.insurance_model_basic"
        self.tags = tags.dict()

    def load_data(self) -> None:
        """Load training and testing data from Delta tables.

        Splits data into features (X_train, X_test) and target (y_train, y_test).
        """
        logger.info("🔄 Loading training and test data...")
        self.train_set_spark = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set")
        self.train_set = self.train_set_spark.toPandas()
        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()
        self.data_version = "0"
        self.train_set.head(5)

        self.X_train = self.train_set[self.num_features + self.cat_features]
        self.y_train = self.train_set[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]
        logger.info("✅ Data loaded.")

    def prepare_features(self) -> None:
        """Encode categorical features and define a preprocessing pipeline.

        Creates a ColumnTransformer for one-hot encoding categorical features while passing through numerical
        features. Constructs a pipeline combining preprocessing and LightGBM regression model.
        """
        logger.info("🔧 Preparing preprocessing pipeline...")
        self.preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        self.pipeline = Pipeline(
            steps=[("preprocessor", self.preprocessor), ("regressor", LGBMRegressor(**self.parameters))]
        )
        logger.info("✅ Preprocessing ready.")

    from sklearn.model_selection import GridSearchCV

    def tune_hyperparameters(self) -> tuple[dict, float]:
        """Perform grid search to tune LightGBM hyperparameters."""
        logger.info("🔍 Tuning hyperparameters...")

        param_grid = {
            "regressor__learning_rate": self.config.parameters["learning_rate"],
            "regressor__n_estimators": self.config.parameters["n_estimators"],
            "regressor__num_leaves": self.config.parameters["num_leaves"],
        }

        grid_search = GridSearchCV(estimator=self.pipeline, param_grid=param_grid, scoring="r2", cv=5, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        self.pipeline = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_

        logger.info(f"✅ Best Params: {self.best_params_}, Best R2: {self.best_score_}")
        return self.best_params_, self.best_score_

    def train(self) -> None:
        """Train the model only if not trained by hyperparameter tuning."""
        if hasattr(self, "best_params_"):
            logger.info("⚠️ Skipping training: already trained during hyperparameter tuning.")
        else:
            logger.info("🚀 Training LightGBM model...")
            self.pipeline.fit(self.X_train, self.y_train)

    def log_model(self) -> None:
        """Log the model using MLflow."""
        logger.info("📝 Logging model to MLflow...")
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            y_pred = self.pipeline.predict(self.X_test)

            # Metrics
            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(self.best_params_)
            mlflow.log_metric("best_r2_cv", self.best_score_)
            mlflow.log_metric("mse", mean_squared_error(self.y_test, y_pred))
            mlflow.log_metric("mae", mean_absolute_error(self.y_test, y_pred))
            mlflow.log_metric("r2_score", r2_score(self.y_test, y_pred))

            signature = infer_signature(self.X_train, y_pred)
            dataset = mlflow.data.from_spark(
                self.train_set_spark,
                table_name=f"{self.catalog_name}.{self.schema_name}.train_set",
                version=self.data_version,
            )
            mlflow.log_input(dataset, context="training")

            mlflow.sklearn.log_model(
                sk_model=self.pipeline, artifact_path="lightgbm-pipeline-model", signature=signature
            )
            logger.info("✅ Model logged to MLflow.")

    def register_model(self) -> None:
        """Register model in Unity Catalog."""
        logger.info("📦 Registering model to Unity Catalog...")
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model", name=self.model_name, tags=self.tags
        )
        logger.info(f"✅ Model registered: version {registered_model.version}")

        MlflowClient().set_registered_model_alias(
            name=self.model_name, alias="latest-model", version=registered_model.version
        )

    def retrieve_current_run_dataset(self) -> DatasetSource:
        """Retrieve MLflow run dataset.

        :return: Loaded dataset source
        """
        run = mlflow.get_run(self.run_id)
        source = mlflow.data.get_source(run.inputs.dataset_inputs[0].dataset)
        logger.info("📂 Dataset loaded from run metadata.")
        return source.load()

    def retrieve_current_run_metadata(self) -> tuple[dict, dict]:
        """Retrieve MLflow run metadata.

        :return: Tuple containing metrics and parameters dictionaries
        """
        run = mlflow.get_run(self.run_id)
        logger.info("📑 Retrieved run metadata.")
        return run.data.metrics, run.data.params

    def load_latest_model_and_predict(self, input_data: pd.DataFrame) -> np.ndarray:
        """Load the latest model from MLflow (alias=latest-model) and make predictions.

        Alias latest is not allowed -> we use latest-model instead as an alternative.

        :param input_data: Pandas DataFrame containing input features for prediction.
        :return: Pandas DataFrame with predictions.
        """
        logger.info("🔄 Loading latest model for inference...")
        model = mlflow.sklearn.load_model(f"models:/{self.model_name}@latest-model")
        logger.info("✅ Model loaded successfully.")
        return model.predict(input_data)
