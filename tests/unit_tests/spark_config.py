"""Spark configuration module for local testing."""

from pydantic_settings import BaseSettings


class SparkConfig(BaseSettings):
    """Configuration class for local Spark session.

    Loads Spark-related settings from environment variables if available.
    Defaults are optimized for local development and unit testing.
    """

    master: str = "local[1]"
    app_name: str = "local_test"
    spark_executor_cores: str = "1"
    spark_executor_instances: str = "1"
    spark_sql_shuffle_partitions: str = "1"
    spark_driver_bindAddress: str = "127.0.0.1"


# Instantiate the configuration
spark_config = SparkConfig()
