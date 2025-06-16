# Databricks notebook source
%pip install ../dist/insurance-0.0.1-py3-none-any.whl
# COMMAND ----------
import mlflow
from loguru import logger
from pyspark.sql import SparkSession

from insurance.config import ProjectConfig, Tags
from insurance.models.feature_lookup_model import FeatureLookUpModel

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()
tags_dict = {"git_sha": "261989george", "branch": "feature3"}
tags = Tags(**tags_dict)

config = ProjectConfig.from_yaml(config_path="../project_config.yml")

# COMMAND ----------
print (config)
print(config.experiment_name_fe)

# COMMAND ----------
# COMMAND ----------
# Initialize model
fe_model = FeatureLookUpModel(config=config, tags=tags, spark=spark)
# COMMAND ----------
# Create feature table
fe_model.create_feature_table()

# COMMAND ----------
# Load data
fe_model.load_data()

# COMMAND ----------
# Perform feature engineering task
fe_model.feature_engineering()