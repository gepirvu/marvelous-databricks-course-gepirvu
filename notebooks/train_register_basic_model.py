# Databricks notebook source

import mlflow
from pyspark.sql import SparkSession

from insurance.config import ProjectConfig, Tags
from insurance.models.basic_model import BasicModel

from dotenv import load_dotenv
from marvelous.common import is_databricks
import os
# COMMAND ----------
# If you have DEFAULT profile set and are logged in with DEFAULT profile,
# skip these lines

if not is_databricks():
    load_dotenv()
    profile = os.environ.get("PROFILE", "marvmlopstechgeorge")
    mlflow.set_tracking_uri(f"databricks://{profile}")
    mlflow.set_registry_uri(f"databricks-uc://{profile}")

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "feature2"})

# COMMAND ----------
# Initialize model with the config path
basic_model = BasicModel(config=config, tags=tags, spark=spark)
# COMMAND ----------
# Initialize model
basic_model.load_data()
basic_model.prepare_features()

# COMMAND ----------
# Optionally tune hyperparameters
# cross-validation, best params, final model fitted on full data
basic_model.tune_hyperparameters()

# COMMAND ----------
basic_model.train()

# COMMAND ----------
basic_model.log_model()
# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=["/Shared/insurance-basic"], filter_string="tags.branch='feature2'"
).run_id[0]
print(run_id)
# COMMAND ----------
model = mlflow.sklearn.load_model(f"runs:/{run_id}/lightgbm-pipeline-model")
print(model)

# COMMAND ----------
# Retrieve dataset for the current run

basic_model.retrieve_current_run_dataset()

# COMMAND ----------
# Retrieve metadata for the current run
basic_model.retrieve_current_run_metadata()

# COMMAND ----------
# Register model
basic_model.register_model()

# COMMAND ----------
# Predict on the test set

test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)

X_test = test_set.drop(config.target).toPandas()
predictions_df = basic_model.load_latest_model_and_predict(X_test)

# Convert full test set (including target) to pandas
test_set_pd = test_set.toPandas()

predictions_df
# COMMAND ----------
# Predict on the test set
results = X_test.copy()
results["predicted"] = predictions_df
results["actual"] = test_set_pd[config.target]

print(results.head())
# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(results["actual"], results["predicted"], alpha=0.7)
plt.plot([results["actual"].min(), results["actual"].max()],
         [results["actual"].min(), results["actual"].max()],
         color="red", linestyle="--", label="Ideal Fit")

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs. Actual Insurance Charges")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
