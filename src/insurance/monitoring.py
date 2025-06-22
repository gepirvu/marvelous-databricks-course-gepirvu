"""Model monitoring module."""

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.catalog import (
    MonitorInferenceLog,
    MonitorInferenceLogProblemType,
)
from loguru import logger
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType, StructField, StructType

from insurance.config import ProjectConfig


def create_or_refresh_monitoring(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create or refresh a monitoring table for model serving data.

    This function processes the inference data from a Delta table,
    parses the request and response JSON fields, joins with test and inference sets,
    and writes the resulting DataFrame to a Delta table for monitoring purposes.

    :param config: Configuration object containing catalog and schema names.
    :param spark: Spark session used for executing SQL queries and transformations.
    :param workspace: Workspace object used for managing quality monitors.
    """
    inf_table = spark.sql(f"SELECT * FROM {config.catalog_name}.{config.schema_name}.`model-serving-fe_payload`")

    request_schema = StructType(
        [
            StructField(
                "dataframe_records",
                ArrayType(
                    StructType(
                        [
                            StructField("Id", StringType(), True),
                            StructField("sex", StringType(), True),
                            StructField("smoker", StringType(), True),
                            StructField("region", StringType(), True),
                            StructField("age", IntegerType(), True),
                            StructField("bmi", DoubleType(), True),
                            StructField("children", IntegerType(), True),
                        ]
                    )
                ),
                True,
            )
        ]
    )

    response_schema = StructType(
        [
            StructField("predictions", ArrayType(DoubleType()), True),
            StructField(
                "databricks_output",
                StructType(
                    [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
                ),
                True,
            ),
        ]
    )

    inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

    inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

    df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))

    df_final = df_exploded.withColumn("timestamp_ms", (F.col("request_time").cast("long") * 1000)).select(
        F.col("request_time").alias("timestamp"),
        F.col("timestamp_ms"),
        F.col("databricks_request_id"),
        F.col("execution_duration_ms"),
        F.col("record.Id").alias("Id"),
        F.col("record.sex"),
        F.col("record.smoker"),
        F.col("record.region"),
        F.col("record.age"),
        F.col("record.bmi"),
        F.col("record.children"),
        F.col("parsed_response.predictions")[0].alias("prediction"),
        F.lit("insurance_model_fe_lightgbm").alias("model_name"),
    )

    df_final = df_final.withColumn("Id", F.col("Id").cast("bigint"))
    test_set = spark.table(f"{config.catalog_name}.{config.schema_name}.test_set")
    inference_set_skewed = spark.table(f"{config.catalog_name}.{config.schema_name}.inference_data_skewed")

    # Join prediction with ground truth
    df_final_with_status = (
        df_final.join(test_set.select("Id", "charges"), on="Id", how="left")
        .withColumnRenamed("charges", "charges_test")
        .join(inference_set_skewed.select("Id", "charges"), on="Id", how="left")
        .withColumnRenamed("charges", "charges_inference")
        .withColumn("charges", F.coalesce(F.col("charges_test"), F.col("charges_inference")).cast("double"))
        .drop("charges_test", "charges_inference")
        .withColumn("prediction", F.col("prediction").cast("double"))
        .dropna(subset=["charges", "prediction"])
    )

    insurance_features = (
        spark.table(f"{config.catalog_name}.{config.schema_name}.insurance_features")
        .withColumnRenamed("age", "age_feature")
        .withColumnRenamed("bmi", "bmi_feature")
        .withColumnRenamed("children", "children_feature")
        .withColumn("Id", F.col("Id").cast("bigint"))
    )

    df_final_with_features = df_final_with_status.join(insurance_features, on="Id", how="left")

    df_final_with_features.write.format("delta").mode("append").saveAsTable(
        f"{config.catalog_name}.{config.schema_name}.model_monitoring"
    )

    try:
        workspace.quality_monitors.get(f"{config.catalog_name}.{config.schema_name}.model_monitoring")
        workspace.quality_monitors.run_refresh(
            table_name=f"{config.catalog_name}.{config.schema_name}.model_monitoring"
        )
        logger.info("Lakehouse monitoring table exist, refreshing.")
    except NotFound:
        create_monitoring_table(config=config, spark=spark, workspace=workspace)
        logger.info("Lakehouse monitoring table is created.")


def create_monitoring_table(config: ProjectConfig, spark: SparkSession, workspace: WorkspaceClient) -> None:
    """Create a new monitoring table for model monitoring.

    This function sets up a monitoring table using the provided configuration,
    SparkSession, and workspace. It also enables Change Data Feed for the table.

    :param config: Configuration object containing catalog and schema names
    :param spark: SparkSession object for executing SQL commands
    :param workspace: Workspace object for creating quality monitors
    """
    logger.info("Creating new monitoring table..")

    monitoring_table = f"{config.catalog_name}.{config.schema_name}.model_monitoring"

    workspace.quality_monitors.create(
        table_name=monitoring_table,
        assets_dir=f"/Workspace/Shared/lakehouse_monitoring/{monitoring_table}",
        output_schema_name=f"{config.catalog_name}.{config.schema_name}",
        inference_log=MonitorInferenceLog(
            problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_REGRESSION,
            prediction_col="prediction",
            timestamp_col="timestamp",
            granularities=["30 minutes"],
            model_id_col="model_name",
            label_col="charges",
        ),
    )

    # Important to update monitoring
    spark.sql(f"ALTER TABLE {monitoring_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
