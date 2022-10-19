# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Databricks notebook source
# MAGIC %md
# MAGIC Copyright (c) Microsoft Corporation. 
# MAGIC Licensed under the MIT license. 
# MAGIC # AutoML Metrics
# MAGIC 
# MAGIC Pull down the model performance metrics and calculate the feature importances from the Databricks AutoML best run.

# COMMAND ----------

# MAGIC %sql
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true;
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.autoCompact = true;
# MAGIC USE growth_factors;

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
import os
import databricks.automl_runtime
from pyspark.sql import functions as f
import numpy as np
import pyspark.pandas as ps
import uuid
import shutil
import pandas as pd
from shap import KernelExplainer, summary_plot

# COMMAND ----------

# Extract user information from the notebook environment
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

expId = mlflow.get_experiment_by_name(f"/Users/{useremail}/databricks_automl/revenue_growth_factors_automl").experiment_id

mlflow_df = spark.read.format("mlflow-experiment").load(expId)

refined_mlflow_df = mlflow_df.select(f.col('run_id'), f.col("experiment_id"), f.explode(f.map_concat(f.col("metrics"))), f.col('start_time'), f.col("end_time")) \
                .filter("key != 'model'") \
                .select("run_id", "experiment_id", "key", f.col("value").cast("float"), f.col('start_time'), f.col("end_time")) \
                .groupBy("run_id", "experiment_id", "start_time", "end_time") \
                .pivot("key") \
                .sum("value") \
                .withColumn("trainingDuration", f.col("end_time").cast("integer")-f.col("start_time").cast("integer")) # example of added column

# choose the best model
best_model = refined_mlflow_df[refined_mlflow_df["best_iteration"] != np.nan].limit(1)

# create a metric table
best_model = best_model[["run_id", "val_roc_auc_score", "val_accuracy_score", "val_precision_score", "val_recall_score", "val_f1_score"]]

best_model = best_model.withColumnRenamed("val_roc_auc_score", "AUC")\
                       .withColumnRenamed("val_accuracy_score", "Accuracy")\
                       .withColumnRenamed("val_precision_score", "Precision")\
                       .withColumnRenamed("val_recall_score", "Recall")\
                       .withColumnRenamed("val_f1_score", "F1")\
                       .pandas_api()

metric_df = ps.melt(best_model, id_vars = ["run_id"], value_vars = ["AUC", "Accuracy", "Precision", "Recall", "F1"], var_name = "Metric", value_name = "Value")\
              .drop("run_id", axis = 1)\
              .to_spark()

run_id = best_model['run_id'][0]

# COMMAND ----------

display(metric_df)

# COMMAND ----------

target_col = "growth"

model_uri = f"runs:/{run_id}/model"
growth_factors_model = mlflow.sklearn.load_model(model_uri)

# COMMAND ----------

df = spark.table("ecommerce_feature_eng").toPandas().drop(['user_id', 'year', 'month'], axis = 1)

split_X = df.drop([target_col], axis=1)
split_y = df[target_col]

# Split out train data
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, random_state=10682915, stratify=split_y)

# Split remaining data equally for validation and test
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, random_state=10682915, stratify=split_y_rem)

# Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=10682915)

# Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
example = X_val.sample(n=min(100, X_val.shape[0]), random_state=10682915)

# Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
predict = lambda x: growth_factors_model.predict(pd.DataFrame(x, columns=X_train.columns))
explainer = KernelExplainer(predict, train_sample, link="identity")
shap_values = explainer.shap_values(example, l1_reg=False)

# Create a Dataframe with the features and global feature importance (mean absolute value)
features = pd.Series(example.columns)
global_importances = pd.DataFrame(shap_values).mean().abs()

feature_importance_df = spark.createDataFrame(pd.DataFrame(dict(Feature = features, Value = global_importances)))

# COMMAND ----------

feature_importance_df.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results to Data Lake
# MAGIC Persist the model results to Delta tables on the Data Lake

# COMMAND ----------

metric_df.write.format('delta').option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/auto_ml_model_metrics/auto_ml_model_metrics").mode('overwrite').saveAsTable("auto_ml_model_metrics")

feature_importance_df.write.format('delta').option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/auto_ml_important_features/auto_ml_important_features").mode('overwrite').saveAsTable("auto_ml_important_features")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM auto_ml_model_metrics

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM auto_ml_important_features
# MAGIC ORDER BY value DESC

# COMMAND ----------

# MAGIC %md Next, let's build our data model for Power BI reporting: <a href="$./07_Data_Model_Silver_to_Gold">`Data Modeling`</a>
