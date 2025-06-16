"""FeatureLookUp model implementation."""

import mlflow
import numpy as np
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from loguru import logger
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from insurance.config import ProjectConfig, Tags


class FeatureLookUpModel:
    """A class to manage FeatureLookupModel for insurance cost prediction."""

    def __init__(self, config: ProjectConfig, tags: Tags, spark: SparkSession) -> None:
        self.config = config
        self.spark = spark
        self.workspace = WorkspaceClient()
        self.fe = feature_engineering.FeatureEngineeringClient()

        # Extract settings from the config
        self.num_features = self.config.num_features  # ['age', 'bmi', 'children']
        self.cat_features = self.config.cat_features  # ['sex', 'smoker', 'region']
        self.target = self.config.target
        self.parameters = self.config.parameters
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name

        # Define table names and function name
        self.feature_table_name = f"{self.catalog_name}.{self.schema_name}.insurance_features"
        self.experiment_name = self.config.experiment_name_fe
        self.tags = tags.dict()

    def create_feature_table(self) -> None:
        """Create or update the insurance_features table and populate it.

        This table stores features related to insurance.
        """
        self.spark.sql(f"""
        CREATE OR REPLACE TABLE {self.feature_table_name} (
            Id STRING NOT NULL,
            age BIGINT,
            bmi DOUBLE,
            children bigint
        )
        TBLPROPERTIES (delta.enableChangeDataFeed = true);
        """)

        self.spark.sql(f"ALTER TABLE {self.feature_table_name} ADD CONSTRAINT insurance_fe_pk PRIMARY KEY(Id);")
        logger.info("✅ Feature table created.")

        # Simulate auto-generation of Id and insert from train/test
        for dataset in ["train_set", "test_set"]:
            self.spark.sql(f"""
            INSERT INTO {self.feature_table_name}
            SELECT
                Id,
                age,
                bmi,
                children
            FROM {self.catalog_name}.{self.schema_name}.{dataset}
            """)

        logger.info("✅ Feature table populated from train/test sets.")

    def load_data(self) -> None:
        """Load train and test sets with synthetic Ids for FeatureLookup."""
        self.train_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.train_set").drop(
            "age", "bmi", "children"
        )

        self.test_set = self.spark.table(f"{self.catalog_name}.{self.schema_name}.test_set").toPandas()

        self.train_set = self.train_set.withColumn("Id", self.train_set["Id"].cast("string"))

        logger.info("✅ Data loaded successfully.")

    def feature_engineering(self) -> None:
        """Perform feature lookup and prepare pandas-ready datasets."""
        self.training_set = self.fe.create_training_set(
            df=self.train_set,
            label=self.target,
            feature_lookups=[
                FeatureLookup(table_name=self.feature_table_name, feature_names=self.num_features, lookup_key="Id")
            ],
            exclude_columns=["update_timestamp_utc"],
        )

        self.training_df = self.training_set.load_df().toPandas()

        # Remaining categorical columns are assumed to be still in original datasets
        self.X_train = self.training_df[self.num_features + self.cat_features]
        self.y_train = self.training_df[self.target]
        self.X_test = self.test_set[self.num_features + self.cat_features]
        self.y_test = self.test_set[self.target]

        print(self.X_train["age"].isna().sum())  # count of NaN
        print(np.isinf(self.X_train["age"]).sum())  # count of inf/-inf
        print(self.X_train[self.X_train["age"].isna()])  # show NaN rows

        logger.info("✅ Feature engineering completed.")

    def train(self) -> None:
        """Train the model and log results to MLflow.

        Uses a pipeline with preprocessing and LightGBM regressor.
        """
        logger.info("🚀 Training LightGBM model...")

        params = {"learning_rate": 0.1, "n_estimators": 100, "num_leaves": 31}

        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), self.cat_features)], remainder="passthrough"
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", LGBMRegressor(**params))])

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(tags=self.tags) as run:
            self.run_id = run.info.run_id
            pipeline.fit(self.X_train, self.y_train)
            y_pred = pipeline.predict(self.X_test)

            mlflow.log_param("model_type", "LightGBM with preprocessing")
            mlflow.log_params(params)
            mlflow.log_metric("mse", mean_squared_error(self.y_test, y_pred))
            mlflow.log_metric("mae", mean_absolute_error(self.y_test, y_pred))
            mlflow.log_metric("r2_score", r2_score(self.y_test, y_pred))

            signature = infer_signature(self.X_train, y_pred)

            self.fe.log_model(
                model=pipeline,
                flavor=mlflow.sklearn,
                artifact_path="insurance-model-fe-lightgbm",
                training_set=self.training_set,
                signature=signature,
            )

            logger.info("✅ Model trained and logged to MLflow.")

    def register_model(self) -> str:
        """Register the trained model to MLflow registry.

        Registers the model and sets alias to 'latest-model'.
        """
        model_name = f"{self.catalog_name}.{self.schema_name}.insurance_model_fe_lightgbm"

        registered_model = mlflow.register_model(
            model_uri=f"runs:/{self.run_id}/insurance-model-fe-lightgbm", name=model_name, tags=self.tags
        )

        latest_version = registered_model.version
        client = MlflowClient()
        client.set_registered_model_alias(name=model_name, version=latest_version, alias="latest-model")

        logger.info("✅ Model registered.")
        return latest_version

    def load_latest_model_and_predict(self, X: DataFrame) -> DataFrame:
        """Load the trained model from MLflow using Feature Engineering Client and make predictions.

        Loads the model with the alias 'latest-model' and scores the batch.
        :param X: DataFrame containing the input features.
        :return: DataFrame containing the predictions.
        """
        model_uri = f"models:/{self.catalog_name}.{self.schema_name}.insurance_model_fe_lightgbm@latest-model"
        return self.fe.score_batch(model_uri=model_uri, df=X)

    def update_feature_table(self) -> None:
        """Update the insurance_features table with the latest records from train and test sets.

        Executes SQL queries to insert new records based on timestamp.
        """
        queries = [
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Id, age, bmi, children
            FROM {self.config.catalog_name}.{self.config.schema_name}.train_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
            f"""
            WITH max_timestamp AS (
                SELECT MAX(update_timestamp_utc) AS max_update_timestamp
                FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            )
            INSERT INTO {self.feature_table_name}
            SELECT Id, age, bmi, children
            FROM {self.config.catalog_name}.{self.config.schema_name}.test_set
            WHERE update_timestamp_utc >= (SELECT max_update_timestamp FROM max_timestamp)
            """,
        ]

        for query in queries:
            logger.info("Executing SQL update query...")
            self.spark.sql(query)
        logger.info("Insurance features table updated successfully.")

    def model_improved(self, test_set: DataFrame) -> bool:
        """Evaluate the model performance on the test set.

        Compares the current model with the latest registered model using MAE.
        :param test_set: DataFrame containing the test data.
        :return: True if the current model performs better, False otherwise.
        """
        X_test = test_set.drop(self.config.target)

        predictions_latest = self.load_latest_model_and_predict(X_test).withColumnRenamed(
            "prediction", "prediction_latest"
        )

        current_model_uri = f"models:/{self.catalog_name}.{self.schema_name}.insurance_model_fe_lightgbm@latest-model"
        predictions_current = self.fe.score_batch(model_uri=current_model_uri, df=X_test).withColumnRenamed(
            "prediction", "prediction_current"
        )

        test_set = test_set.select("Id", self.config.target)

        logger.info("Predictions are ready.")

        # Join the DataFrames on the 'Id' column
        df = test_set.join(predictions_current, on="Id").join(predictions_latest, on="Id")

        # Calculate the absolute error for each model
        df = df.withColumn("error_current", F.abs(df[self.config.target] - df["prediction_current"]))
        df = df.withColumn("error_latest", F.abs(df[self.config.target] - df["prediction_latest"]))

        # Calculate the Mean Absolute Error (MAE) for each model
        mae_current = df.agg(F.mean("error_current")).collect()[0][0]
        mae_latest = df.agg(F.mean("error_latest")).collect()[0][0]

        # Compare models based on MAE
        logger.info(f"MAE for Current Model: {mae_current}")
        logger.info(f"MAE for Latest Model: {mae_latest}")

        if mae_current < mae_latest:
            logger.info("Current Model performs better.")
            return True
        else:
            logger.info("New Model performs worse.")
            return False
