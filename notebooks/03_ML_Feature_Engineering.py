# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Databricks notebook source
# MAGIC %md
# MAGIC Copyright (c) Microsoft Corporation. 
# MAGIC Licensed under the MIT license. 
# MAGIC # Feature Engineering
# MAGIC 
# MAGIC Analyze the transformed data and select the features that will be used in the model using the following steps in this notebook:
# MAGIC 
# MAGIC 1. Exploratory data analysis
# MAGIC 2. Remove outliers
# MAGIC 3. Correlation analysis
# MAGIC 4. Feature selection
# MAGIC 5. Save results to data lake

# COMMAND ----------

# MAGIC %sql
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true;
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.autoCompact = true;
# MAGIC USE growth_factors;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports

# COMMAND ----------

import pyspark
from pyspark.sql.functions import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.ml.feature import *
from pyspark.ml.stat import *
import pyspark.pandas as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in Transformed Data from Delta Lake Table

# COMMAND ----------

# transformed df
df = spark.table("ecommerce_silver")

# COMMAND ----------

# print number of rows and columns
print('Columns:', len(df.columns))
print('Rows:', df.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Distribution of Count vs. Binary Features

# COMMAND ----------

# distribution of apple brand view count
display(df.groupBy('brand_apple_viewed_count').count().orderBy(desc('count')))

# COMMAND ----------

# distribution of apple brand view binary
display(df.groupBy('brand_apple_viewed_binary').count().orderBy(desc('count')))

# COMMAND ----------

# distribution of smartphone subcategory view count
display(df.groupBy('subcategory_smartphone_viewed_count').count().orderBy(desc('count')))

# COMMAND ----------

# distribution of smartphone subcategory binary
display(df.groupBy('subcategory_smartphone_viewed_binary').count().orderBy(desc('count')))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Keep Binary Features
# MAGIC 
# MAGIC Because the features that measure counts of views, add to cart, and purchases are very right skewed, it makes sense to either bin the upper values or use the binary feature instead. Because the binary features have balanced classes, we are going to use the binary features.

# COMMAND ----------

# only keep binary columns
df = df.select('user_id', 'year', 'month', 'growth', 'sessions_per_user_per_month', 'avg_session_duration_per_user_per_month', 
              'avg_conversion_rate_per_user_per_month', 'avg_order_value_per_user_per_month', 'avg_cart_abandon_rate', 
              'brand_apple_viewed_binary', 'brand_samsung_viewed_binary', 'brand_xiaomi_viewed_binary', 'brand_huawei_viewed_binary', 
              'brand_lenovo_viewed_binary', 
              'brand_apple_added_binary', 'brand_samsung_added_binary', 'brand_xiaomi_added_binary', 
              'brand_huawei_added_binary', 'brand_acer_added_binary', 
              'brand_apple_purchased_binary', 'brand_samsung_purchased_binary', 'brand_xiaomi_purchased_binary',
              'brand_huawei_purchased_binary', 'brand_acer_purchased_binary', 
              'subcategory_smartphone_viewed_binary', 'subcategory_audio_viewed_binary', 'subcategory_clocks_viewed_binary', 
              'subcategory_tablet_viewed_binary', 'subcategory_telephone_viewed_binary', 
              'subcategory_smartphone_added_binary', 'subcategory_audio_added_binary', 'subcategory_clocks_added_binary', 
              'subcategory_tablet_added_binary', 'subcategory_telephone_added_binary', 
              'subcategory_smartphone_purchased_binary', 'subcategory_audio_purchased_binary', 'subcategory_clocks_purchased_binary', 
              'subcategory_tablet_purchased_binary',
              'subcategory_telephone_purchased_binary',
              'product_id_1004856_viewed_binary', 'product_id_1005115_viewed_binary', 'product_id_1004767_viewed_binary',
              'product_id_4804056_viewed_binary', 'product_id_1005105_viewed_binary',
              'product_id_1004856_added_binary', 'product_id_1004767_added_binary', 'product_id_1005115_added_binary',
              'product_id_4804056_added_binary', 'product_id_1004833_added_binary', 
              'product_id_1004856_purchased_binary', 'product_id_1004767_purchased_binary', 'product_id_1005115_purchased_binary', 
              'product_id_4804056_purchased_binary', 'product_id_1004833_purchased_binary')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribution of Continous Variables

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert to Pandas Dataframe for Visualization

# COMMAND ----------

# convert to pandas dataframe to use for visualizations
psdf = df.pandas_api()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Remove Outliers

# COMMAND ----------

# DBTITLE 1,User Sessions with Outliers
# Initialize a new figure
fig, ax = plt.subplots(figsize=(8, 4), dpi=80)

# Distribution of Sessions Per User Per Month
ax.hist(psdf['sessions_per_user_per_month'], bins = 100)
ax.set_title("Distribution of Sessions per User per Month with Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,User Sessions without Outliers
# remove outliers from sessions per user per month
psdf_filtered = psdf[psdf['sessions_per_user_per_month'] <= 60]

# Initialize a new figure
fig, ax = plt.subplots(figsize=(8, 4), dpi=80)

# Distribution of Sessions Per User Per Month
ax.hist(psdf_filtered['sessions_per_user_per_month'], bins = 100)
ax.set_title("Distribution of Sessions per User per Month no Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,Session Duration with Outliers
# Initialize a new figure
fig, ax = plt.subplots(figsize=(8, 4), dpi=80)

# Distribution of Sessions Per User Per Month
ax.hist(psdf_filtered['avg_session_duration_per_user_per_month'], bins = 100)
ax.set_title("Distribution of Session Duration per User per Month with Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,Session Duration without Outliers
# remove outliers from session duration per user per month
psdf_filtered = psdf_filtered[psdf_filtered['avg_session_duration_per_user_per_month'] <= 4000]

# Initialize a new figure
fig, ax = plt.subplots(figsize=(8, 4), dpi=80)

# Distribution of Sessions Per User Per Month
ax.hist(psdf_filtered['avg_session_duration_per_user_per_month'], bins = 100)
ax.set_title("Distribution of Session Duration per User per Month no Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,Conversion Rate with Outliers
# Initialize a new figure
fig, ax = plt.subplots(figsize=(8, 4), dpi=80)

# Distribution of Sessions Per User Per Month
ax.hist(psdf_filtered['avg_conversion_rate_per_user_per_month'], bins = 100)
ax.set_title("Distribution of Avg Conversion Rate per Month with Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,Conversion Rate without Outliers
# remove invalid values for avg conversion rate
psdf_filtered = psdf_filtered[psdf_filtered['avg_conversion_rate_per_user_per_month'] <= 1]

# Initialize a new figure
fig, ax = plt.subplots(figsize=(8, 4), dpi=80)

# Distribution of Sessions Per User Per Month
ax.hist(psdf_filtered['avg_conversion_rate_per_user_per_month'], bins = 100)
ax.set_title("Distribution of Avg Conversion Rate per Month no Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,Order Value with Outliers
# Initialize a new figure
fig, ax = plt.subplots(figsize = (8, 4), dpi = 80)

# Distribution of Avg Order Value per Month with Outliers
ax.hist(psdf_filtered['avg_order_value_per_user_per_month'], bins = 100)
ax.set_title("Distribution of Avg Order Value per Month with Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,Order Value without Outliers
# remove outliers from avg order value per user per month
psdf_filtered = psdf_filtered[psdf_filtered['avg_order_value_per_user_per_month'] <= 2000]

# Initialize a new figure
fig, ax = plt.subplots(figsize = (8, 4), dpi = 80)

# Distribution of Avg Order Value per Month no Outliers
ax.hist(psdf_filtered['avg_order_value_per_user_per_month'], bins = 100)
ax.set_title("Distribution of Avg Order Value per Month no Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,Cart Abandon Rate with Outliers
# Initialize a new figure
fig, ax = plt.subplots(figsize = (8, 4), dpi = 80)

# Distribution of Avg Cart Abandon Rate with Outliers
ax.hist(psdf_filtered['avg_cart_abandon_rate'], bins = 100)
ax.set_title("Distribution of Avg Cart Abandon Rate with Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,Cart Abandon Rate without Outliers
psdf_filtered = psdf_filtered[(psdf_filtered['avg_cart_abandon_rate'] <= 1) & (psdf_filtered['avg_cart_abandon_rate'] >= 0)]

# Initialize a new figure
fig, ax = plt.subplots(figsize = (8, 4), dpi = 80)

# Distribution of Avg Cart Abandon Rate no Outliers
ax.hist(psdf_filtered['avg_cart_abandon_rate'], bins = 100)
ax.set_title("Distribution of Avg Cart Abandon Rate no Outliers")
ax.set_xlabel("# of Sessions")
ax.set_ylabel("Frequency")

# COMMAND ----------

# DBTITLE 1,Percent of Dataframe remaining after eliminating outliers
print('% of data remaining:', '{:.2%}'.format(len(psdf_filtered)/len(psdf)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Selection

# COMMAND ----------

# MAGIC %md
# MAGIC ### Correlation Analysis
# MAGIC We are going to example correlations between features to search for multicolinearity (where 2+ features are highly correlated with each other).

# COMMAND ----------

# search for features with at least 80% correlation
correlations = psdf_filtered.corr().abs().unstack().sort_values(ascending = False).drop_duplicates()
correlations[correlations>=0.8]

# COMMAND ----------

# MAGIC %md
# MAGIC There is a lot of multicolinearity between viewed, added to cart, and purchased features which means that keeping all 3 in the dataset would be overcounting these features. Below we will do further analysis to determine which set of features to keep.

# COMMAND ----------

# convert back into a Spark DataFrame
df_filtered = psdf_filtered.to_spark()

# COMMAND ----------

# distribution of apple brand view binary
display(df_filtered.groupBy('brand_apple_viewed_binary').count().orderBy(desc('count')))

# COMMAND ----------

# distribution of apple brand added to cart binary
display(df_filtered.groupBy('brand_apple_added_binary').count().orderBy(desc('count')))

# COMMAND ----------

# distribution of apple brand purchased binary
display(df_filtered.groupBy('brand_apple_purchased_binary').count().orderBy(desc('count')))

# COMMAND ----------

# MAGIC %md
# MAGIC Because purchase events are most representative of customers buying intent compared to views and added to cart and because these events still have balanced classes, we are going to keep only purchased features.

# COMMAND ----------

# keep only purchased features
df_filtered = df_filtered.select(['user_id', 'year', 'month', 'growth', 'sessions_per_user_per_month', 'avg_session_duration_per_user_per_month', 
                                    'avg_conversion_rate_per_user_per_month', 'avg_order_value_per_user_per_month', 'avg_cart_abandon_rate', 
                                    'brand_apple_purchased_binary', 'brand_samsung_purchased_binary', 'brand_xiaomi_purchased_binary', 
                                    'brand_huawei_purchased_binary', 'brand_acer_purchased_binary',
                                    'subcategory_smartphone_purchased_binary', 'subcategory_audio_purchased_binary',
                                    'subcategory_clocks_purchased_binary', 'subcategory_tablet_purchased_binary',
                                    'subcategory_telephone_purchased_binary', 'product_id_1004856_purchased_binary',
                                    'product_id_1004767_purchased_binary', 'product_id_1005115_purchased_binary',
                                    'product_id_4804056_purchased_binary', 'product_id_1004833_purchased_binary'])

# COMMAND ----------

# print number of rows and columns after feature selection
print('Columns:', len(df_filtered.columns))
print('Rows:', df_filtered.count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Results to Data Lake
# MAGIC Persist the transformed data to a Delta Table on the Data Lake

# COMMAND ----------

# write transformed data to spark table
df_filtered.write.format("delta").option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/ecommerce_feature_eng").mode("overwrite").saveAsTable("ecommerce_feature_eng")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * 
# MAGIC FROM ecommerce_feature_eng
# MAGIC LIMIT 3

# COMMAND ----------

# MAGIC %md After feature engineering, we will build an ML model using Spark ML: <a href="$./04_ML_Model_Building">`ML Model Building`</a>
