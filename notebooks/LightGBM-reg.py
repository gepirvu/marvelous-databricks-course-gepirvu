# COMMAND ----------
import lightgbm as lgb
import matplotlib.pyplot as plt

# import sys
# sys.path.append(str(Path.cwd().parent / "src"))
import yaml
from loguru import logger
from marvelous.logging import setup_logging
from marvelous.timer import Timer
from pyspark.sql import SparkSession

from insurance.config import ProjectConfig
from insurance.data_preprocessing import DataProcessor
from insurance.model_trainer import ModelTrainer

# COMMAND ----------
config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")
setup_logging(log_file=f"/Volumes/{config.catalog_name}/{config.schema_name}/logs/marvelous-1.log")

setup_logging(log_file="logs/marvelous-1.log")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()
df = spark.read.option("header", True).csv(config.data_path).toPandas()

with Timer() as preprocess_timer:
    processor = DataProcessor(df, config, spark)
    processor.preprocess()

train_set, test_set = processor.split_data()  # for UC save

    # Save to catalog
logger.info("Saving data to catalog")
processor.save_to_catalog(train_set, test_set)

logger.info(f"Data preprocessing: {preprocess_timer}")

# COMMAND ----------
with Timer() as model_train:
    trainer = ModelTrainer(train_set, config)
    X_train, X_test, y_train, y_test = trainer.split()

    best_params, best_score = trainer.tune_hyperparameters(X_train, y_train)

    final_model = trainer.train_final_model(X_train, y_train)
    y_pred = final_model.predict(X_test)
    metrics = trainer.evaluate(y_test, y_pred)

    print("Best params:", best_params)
    print("R²:", metrics["r2"], "RMSE:", metrics["rmse"], "MAE:", metrics["mae"])
logger.info(f"Model trainer and tuning: {model_train}")

# COMMAND ----------
logger.info("Training set shape: %s", X_train.shape)
logger.info("Test set shape: %s", X_test.shape)

# COMMAND ----------
lgb.plot_importance(final_model)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()

# COMMAND ----------
# Enable change data feed (only once!)
# logger.info("Enable change data feed")
# processor.enable_change_data_feed()
