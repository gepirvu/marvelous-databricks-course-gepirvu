# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
import pandas as pd

# COMMAND ----------

# File location and type
file_location = "dbfs:/Volumes/mlops_dev/pirvugeo/data/insurance.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = (
    spark.read.format(file_type)
    .option("inferSchema", infer_schema)
    .option("header", first_row_is_header)
    .option("sep", delimiter)
    .load(file_location)
)

dataset = df.toPandas()
dataset.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Checking for missing values

# COMMAND ----------

dataset.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ####Handling categorical variables

# COMMAND ----------

dataset["sex"].unique()

# COMMAND ----------

dataset["sex"] = dataset["sex"].apply(lambda x: 0 if x == "female" else 1)

# COMMAND ----------

dataset["smoker"] = dataset["smoker"].apply(lambda x: 0 if x == "no" else 1)
dataset.head()

# COMMAND ----------

dataset["region"].unique()

# COMMAND ----------

region_dummies = pd.get_dummies(dataset["region"], drop_first=True)
region_dummies

# COMMAND ----------

dataset = pd.concat([region_dummies, dataset], axis=1)


# COMMAND ----------

dataset.drop(["region"], axis=1, inplace=True)

# COMMAND ----------

dataset.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Split Dataset in training and test set

# COMMAND ----------

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# COMMAND ----------

X

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Building and training the model

# COMMAND ----------

import lightgbm as lgb

model = lgb.LGBMRegressor(force_col_wise=True)

# COMMAND ----------

model.fit(X_train, y_train)

# COMMAND ----------

# inference
y_pred = model.predict(X_test)

# COMMAND ----------

y_pred

# COMMAND ----------

# MAGIC %md
# MAGIC ###Evaluating the model

# COMMAND ----------

# MAGIC %md
# MAGIC ####1.R-Squared

# COMMAND ----------

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

# COMMAND ----------

r2

# COMMAND ----------

# MAGIC %md
# MAGIC ####2.Adjusted R-Squared

# COMMAND ----------

k = X_test.shape[1]
n = X_test.shape[0]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

# COMMAND ----------

adj_r2

# COMMAND ----------

# MAGIC %md
# MAGIC ####3.k-Fold Cross Validation

# COMMAND ----------

from sklearn.model_selection import cross_val_score

r2s = cross_val_score(estimator=model, X=X, y=y, scoring="r2", cv=10)


# COMMAND ----------

r2s

# COMMAND ----------

print(f"Average R-Squared: {r2s.mean() * 100:.3f} %")
print(f"Standard Deviation: {r2s.std() * 100:.3f} %")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine-Tuning with GridSearch

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

# we test with the following parameters. We select the default value and 2 other values near the default. example: num_leaves 31, then 32,33, 39,30
parameters = [
    {
        "num_leaves": [29, 30, 31, 32, 33],
        "learning_rate": [0.08, 0.09, 0.1, 0.11, 0.12],
        "n_estimators": [80, 90, 100, 110, 120],
    }
]

grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring="r2", cv=10)


# COMMAND ----------

grid_search.fit(X, y)

# COMMAND ----------

best_r2 = grid_search.best_score_
best_params = grid_search.best_params_
print(f"Best R-Squared: {best_r2 * 100:.2f} %")
print("Best Parameters:", best_params)
# COMMAND ----------
best_params = {"learning_rate": 0.08, "n_estimators": 80, "num_leaves": 30}

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train)

# COMMAND ----------
y_pred = final_model.predict(X_test)

# COMMAND ----------
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
