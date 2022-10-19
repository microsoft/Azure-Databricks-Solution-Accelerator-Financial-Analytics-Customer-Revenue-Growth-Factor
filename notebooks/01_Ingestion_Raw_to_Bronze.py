# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Databricks notebook source
# MAGIC %md
# MAGIC Copyright (c) Microsoft Corporation. 
# MAGIC Licensed under the MIT license. 
# MAGIC # Ingestion - Raw to Bronze
# MAGIC 
# MAGIC The first step is to clean the source dataset into a version we can work with:
# MAGIC 1. Remove data missing that is missing brand and category values
# MAGIC 2. Filter to only keep brands and categories that are accurately mapped
# MAGIC 3. Write the results to a Bronze Delta Lake table

# COMMAND ----------

# MAGIC %sql
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true;
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.autoCompact = true;
# MAGIC USE growth_factors;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports

# COMMAND ----------

# python libary imports
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read in Data from Azure Data Lake

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/revenue_growth_factors/raw/"))

# COMMAND ----------

df = spark.read.csv("dbfs:/revenue_growth_factors/raw/", header = 'true')

# COMMAND ----------

# show retail dataframe
df.limit(3).display()

# COMMAND ----------

# drop rows that have no category, brand, or user session
df = df.filter((df.category_code != 'null') & (df.brand != 'null') & (df.brand != 'user_session'))

# COMMAND ----------

# split category code into category and subcategory
df = df.withColumn('category', split(col('category_code'), '\.').getItem(0))\
       .withColumn('subcategory', split(col('category_code'), '\.').getItem(1))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Cleaning
# MAGIC The retail dataset has some messy data where the brands don't align with the categories. Below is code to analyze the dataset to filter it down to the categories, subcategories, and brands that look the best.

# COMMAND ----------

df.limit(3).display()

# COMMAND ----------

# filter down to just electronics, only electronic subcategories and brands are accurate
smartphone_brands = ['samsung', 'apple', 'xiaomi', 'huawei', 'oppo', 'meizu', 'nokia', 'honor', 'sony', 'oneplus', 'lg']
audio_brands = ['lenovo', 'acer', 'apple', 'asus', 'hp', 'xiaomi', 'jbl', 'dell', 'pioneer', 'samsung', 'kicx', 'yamaha', 'sony', 'pride',
                'alphard', 'element', 'bosch', 'stagg', 'alpine', 'adagio', 'huawei', 'hertz', 'elari', 'alteco', 'msi', 'edge', 'crown', 'fender',
                'kenwood', 'conceptclub', 'harper', 'valkiria', 'cortland', 'phantom', 'makita']
clock_brands = ['casio', 'apple', 'samsung', 'xiaomi', 'garmin', 'amazfit', 'orient', 'tissot', 'huawei', 'wonlex', 'aimoto', 'armani', 'boccia', 'elari', 'fossil', 'canyon']
tablet_brands = ['samsung', 'apple', 'lenovo', 'huawei', 'prestigio', 'acer', 'xiaomi', 'wacom', 'huion', 'microsoft']
telephone_brands = ['nokia', 'texet', 'panasonic', 'maxvi', 'lorelli', 'philips', 'prestigio']

df = df.filter(
                                    (df['category'] == 'electronics') & \
                                    (
                                        (df['subcategory'] == 'smartphone') & (df['brand'].isin(smartphone_brands)) | \
                                        (df['subcategory'] == 'audio') & (df['brand'].isin(audio_brands)) | \
                                        (df['subcategory'] == 'clocks') & (df['brand'].isin(clock_brands)) | \
                                        (df['subcategory'] == 'tablet') & (df['brand'].isin(tablet_brands)) | \
                                        (df['subcategory'] == 'telephone') & (df['brand'].isin(telephone_brands))
                                    )
                                )

# add month & year, re-order columns
df = df.withColumn('month', month('event_time')) \
    .withColumn('year', year('event_time')) \
    .drop('category_code') \
    .select('user_id', 'year', 'month', 'event_type', 'product_id', 'category_id', 'category', 'subcategory', 'brand', 'price', 'user_session', 'event_time')

# convert timestamp column to timestamp format
df = df.withColumn("event_time", to_timestamp(col("event_time")))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Cleaned Data to a Delta Lake Table

# COMMAND ----------

df.write.format("delta").option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/ecommerce_bronze").mode("overwrite").saveAsTable("ecommerce_bronze")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ecommerce_bronze LIMIT 3

# COMMAND ----------

# MAGIC %md Next, we transform the data to to capture relevant metrics for ML modeling: <a href="$./02_DE_Bronze_to_Silver">`Data Engineering - Bronze to Silver`</a>
