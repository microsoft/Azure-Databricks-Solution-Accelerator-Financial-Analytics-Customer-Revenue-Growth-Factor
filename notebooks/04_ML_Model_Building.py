# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Databricks notebook source
# MAGIC %md
# MAGIC Copyright (c) Microsoft Corporation. 
# MAGIC Licensed under the MIT license. 
# MAGIC # ML Model Building
# MAGIC 
# MAGIC Pre-process data and use the data to build a Spark machine learning model in this notebook using the following steps:
# MAGIC 
# MAGIC 1. Training-test split
# MAGIC 1. Data pre-processing (one-hot encoding, vectorizor)
# MAGIC 1. Build machine learning model
# MAGIC 1. Calculate model performance metrics
# MAGIC 1. Extract model feature importances
# MAGIC 1. Save results to data lake

# COMMAND ----------

# MAGIC %sql
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true;
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.autoCompact = true;
# MAGIC USE growth_factors;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports

# COMMAND ----------

# MAGIC %md 
# MAGIC Install com.microsoft.azure:synapseml_2.12:0.10.1 on the cluster. Use DBR 10.4 ML LTS or greater.

# COMMAND ----------

from synapse.ml.lightgbm import *
from pyspark.sql.functions import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import *
from pyspark.ml.evaluation import *
from pyspark.ml.classification import *
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read In Data From Delta Lake

# COMMAND ----------

# feature engineered table
df = spark.table("ecommerce_feature_eng")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-Test Split
# MAGIC Split data into a 70-30 training-test split

# COMMAND ----------

(trainDF, testDF) = df.randomSplit([.7, .3], seed = 123)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ML Pre-Processing & Model Building
# MAGIC 1. Pre-process data by encoding categorical columns and assembling them into a vector format expected for model building.
# MAGIC 2. Build a Spark pipeline binary classifier model to predict growth using LightGBM
# MAGIC 3. Use this model to score the test dataset to get model performance metrics

# COMMAND ----------

# Target column (label)
target_col = 'growth'

# ID columns
id_col_1 = 'user_id'
id_col_2 = 'year'
id_col_3 = 'month'

# Separate into Categorical, Target, and Numeric Columns

# Create categorical column list with all of the columns that contain int and string values
categorical_cols = ['brand_apple_purchased_binary', 'brand_samsung_purchased_binary', 'brand_xiaomi_purchased_binary', 
                    'brand_huawei_purchased_binary', 'brand_acer_purchased_binary', 'subcategory_smartphone_purchased_binary', 
                    'subcategory_audio_purchased_binary', 'subcategory_clocks_purchased_binary', 
                    'subcategory_tablet_purchased_binary', 'subcategory_telephone_purchased_binary', 
                    'product_id_1004856_purchased_binary', 'product_id_1004767_purchased_binary', 
                    'product_id_1005115_purchased_binary', 'product_id_4804056_purchased_binary', 'product_id_1004833_purchased_binary']

numeric_cols = ['sessions_per_user_per_month', 'avg_session_duration_per_user_per_month', 'avg_conversion_rate_per_user_per_month',
                'avg_order_value_per_user_per_month', 'avg_cart_abandon_rate']

stages = [] # stages in our Pipeline

# Category Indexing with StringIndexer - Use OneHotEncoder to convert categorical variables into binary SparseVectors
string_indexes = [StringIndexer(inputCol = c, outputCol = 'idx_' + c, handleInvalid = 'keep') for c in categorical_cols]
onehot_indexes = [OneHotEncoder(inputCols = ['idx_' + c], outputCols = ['ohe_' + c]) for c in categorical_cols]
stages += string_indexes + onehot_indexes

# Transform all numeric features into a vector using VectorAssembler
assembler_inputs = ['ohe_' + c for c in categorical_cols] + numeric_cols
assembler = VectorAssembler(inputCols = assembler_inputs, outputCol = 'features', handleInvalid = 'keep')
stages += [assembler]

# Create an indexed label from your target variable
label_string_idx = StringIndexer(inputCol = target_col, outputCol = 'label', handleInvalid = 'keep')
stages += [label_string_idx]

# Set a random seed variable for reproducibility
random_seed_val = 12345

# Light GBM Classifier
lgbm = LightGBMClassifier(learningRate = 0.1, numIterations = 100, numLeaves = 50)
stages += [lgbm]

lgbmPipeline = Pipeline(stages = stages)
lgbmPipelineModel = lgbmPipeline.fit(trainDF)
lgbmDF = lgbmPipelineModel.transform(testDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Metrics
# MAGIC Calculate classification model metrics using the test dataset

# COMMAND ----------

mce = MulticlassClassificationEvaluator()
bce = BinaryClassificationEvaluator()

accuracy = mce.setMetricName('accuracy').evaluate(lgbmDF)
precision = mce.setMetricName('weightedPrecision').evaluate(lgbmDF)
recall = mce.setMetricName('weightedRecall').evaluate(lgbmDF)
f1 = mce.setMetricName('f1').evaluate(lgbmDF)
auc = bce.setMetricName('areaUnderROC').evaluate(lgbmDF)

# model metrics df
model_metrics = spark.createDataFrame(
    [
        ('Accuracy', f'{accuracy:.2f}'),
        ('Precision', f'{precision:.2f}'),
        ('Recall', f'{recall:.2f}'),
        ('F1 Score', f'{f1:.2f}'),
        ('AUC', f'{auc:.2f}'),
    ],
    ['Metric', 'Value']
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importances
# MAGIC Use the model feature importances to determine the top revenue growth factors and their relative importances

# COMMAND ----------

# Custom function to extract feature names and importance - partly borrowed from https://gist.github.com/timlrx/1d5fdb0a43adbbe32a9336ba5c85b1b2#file-featureimportanceselector-py
def ExtractFeatureImp(featureImp, df, featuresCol):
    list_extract = []
    for i in df.schema[featuresCol].metadata['ml_attr']['attrs']:
        list_extract = list_extract + df.schema[featuresCol].metadata['ml_attr']['attrs'][i]
    varlist = pd.DataFrame(list_extract)
    varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
    return(varlist.sort_values('score', ascending = False))
  
varlist = ExtractFeatureImp(lgbmPipelineModel.stages[-1].getFeatureImportances(), lgbmDF, 'features')

# important features df
important_features = spark.createDataFrame(varlist)
important_features = important_features.drop('idx')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results to Data Lake
# MAGIC Persist the model results to Delta tables on the Data Lake

# COMMAND ----------

model_metrics.write.format('delta').option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/pyspark_model_metrics/pyspark_model_metrics").mode('overwrite').saveAsTable("pyspark_model_metrics")

important_features.write.format('delta').option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/pyspark_important_features/pyspark_important_features").mode('overwrite').saveAsTable("pyspark_important_features")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM pyspark_model_metrics

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM pyspark_important_features
# MAGIC ORDER BY score DESC

# COMMAND ----------

# MAGIC %md Let's automate this model building process and deploy a model using AutoML and MLflow: <a href="$./05_AutoML_and_Deployment">`AutoML & MLflow`</a>
