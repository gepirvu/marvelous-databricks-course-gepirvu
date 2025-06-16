# Databricks notebook source
import mlflow
from pyspark.sql import SparkSession
from dotenv import load_dotenv
from insurance.config import ProjectConfig, Tags
from insurance.models.custom_model import CustomModel

from insurance import __version__ as insurance_v
import os

# COMMAND ----------
# Default profile:
load_dotenv()
profile = os.environ.get("PROFILE", "marvmlopstechgeorge")
mlflow.set_tracking_uri(f"databricks://{profile}")
mlflow.set_registry_uri(f"databricks-uc://{profile}")
# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config.yml")
spark = SparkSession.builder.getOrCreate()
tags = Tags(**{"git_sha": "abcd12345", "branch": "feature2"})

# COMMAND ----------
# Initialize model with the config path
custom_model = CustomModel(
    config=config, tags=tags, spark=spark,
    code_paths=[f"../dist/insurance-{insurance_v}-py3-none-any.whl"]
)
# COMMAND ----------
custom_model.load_data()
custom_model.prepare_features()
# COMMAND ----------
custom_model.tune_hyperparameters()
# COMMAND ----------
# Train + log the model (runs everything including MLflow logging)
#custom_model.train()
custom_model.log_model()
# COMMAND ----------
run_id = mlflow.search_runs(experiment_names=["/Shared/insurance-custom"]).run_id[0]
run_id
# COMMAND ----------
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/pyfunc-insurance-model")
model
# COMMAND ----------
# Retrieve dataset for the current run
custom_model.retrieve_current_run_dataset()
# COMMAND ----------
# Register model
custom_model.register_model()
# COMMAND ----------
# Predict on the test set
test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set").limit(10)
test_set.head(10)
# COMMAND ----------
X_test = test_set.drop(config.target, "Id").toPandas()

predictions_df = custom_model.load_latest_model_and_predict(X_test)
# COMMAND ----------
test_set_pd = test_set.toPandas()
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