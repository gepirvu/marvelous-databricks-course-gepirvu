# src/insurance/data_processor.py
"""Data preprocessing module for the insurance dataset.

Handles encoding, missing value imputation, dataset splitting,
and catalog persistence using Spark.
"""

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split

from insurance.config import ProjectConfig


class DataProcessor:
    """A class for preprocessing and managing DataFrame operations."""

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df
        self.config = config
        self.spark = spark

    def preprocess(self) -> None:
        """Preprocess the DataFrame stored in self.df."""
        for col in self.config.num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        for col in self.config.cat_features:
            self.df[col] = self.df[col].astype("category")

        self.df[self.config.target] = pd.to_numeric(self.df[self.config.target], errors="coerce")
        self.df.dropna(subset=self.config.num_features, inplace=True)

        # Create final feature list after preprocessing
        feature_columns = [col for col in self.df.columns if col != self.config.target]
        relevant_columns = feature_columns + [self.config.target]
        self.df = self.df[relevant_columns]

        print("\n[DataProcessor] Data after preprocessing:")
        print(self.df.head())

    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame into training and test sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
            random_state (int): Controls the shuffling applied before the split for reproducibility.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and test sets.

        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """Save training and test sets as Delta tables in the Unity Catalog.

        Args:
            train_set (pd.DataFrame): The training dataset to persist.
            test_set (pd.DataFrame): The test dataset to persist.

        Returns:
            None

        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed (CDF) for the train and test Delta tables.

        This allows tracking row-level changes (insert, update, delete)
        for downstream processing or auditing.

        Returns:
            None

        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
