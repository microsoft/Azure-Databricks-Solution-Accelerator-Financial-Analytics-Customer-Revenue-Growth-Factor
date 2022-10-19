# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Databricks notebook source
# MAGIC %md
# MAGIC Copyright (c) Microsoft Corporation. 
# MAGIC Licensed under the MIT license. 
# MAGIC # Power BI Data Model - Silver to Gold
# MAGIC 
# MAGIC Build a data model for reporting in Power BI. Save the resulting tables in Delta Lake.
# MAGIC 
# MAGIC The resulting data model includes four tables:
# MAGIC 
# MAGIC 1. Customer: user info, growth/no growth, & aggregated session metrics
# MAGIC 1. Activity: user clickstream activity, e.g. product views, purchases
# MAGIC 1. Products: reference table with additional product information
# MAGIC 1. Categories: reference table with additional product category information.

# COMMAND ----------

# MAGIC %sql
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite = true;
# MAGIC SET spark.databricks.delta.properties.defaults.autoOptimize.autoCompact = true;
# MAGIC USE growth_factors;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Library Imports

# COMMAND ----------

from pyspark.sql.functions import *
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.window import Window

# COMMAND ----------

# MAGIC %md
# MAGIC ### Customer Table
# MAGIC A table with the customer and key metrics about the customer

# COMMAND ----------

customers = spark.table("ecommerce_silver")

# COMMAND ----------

# filter down just to the columns needed for the customer table and add composite key of customer, year, month
customers = customers[['user_id',
                       'year',
                       'month',
                       'growth',
                       'sessions_per_user_per_month',
                       'avg_session_duration_per_user_per_month',
                       'avg_conversion_rate_per_user_per_month',
                       'avg_order_value_per_user_per_month',
                       'avg_cart_abandon_rate']]\
                     .withColumn('UID', concat(customers['user_id'], lit('-'), customers['year'], lit('-'), customers['month']))

# COMMAND ----------

customers.limit(3).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clickstream Activity Table
# MAGIC 
# MAGIC A transaction table that lists each clickstream event, including product views, add to cart, and purchases.

# COMMAND ----------

ecommerce_bronze = spark.table("ecommerce_bronze")

# filter to only rows where 'growth' is applicable, i.e. rows in customer table
activities = ecommerce_bronze.withColumn('UID', concat(ecommerce_bronze['user_id'], lit('-'), ecommerce_bronze['year'], lit('-'), ecommerce_bronze['month'])) \
    .join(customers, ['UID'], how='right') \
    .select('UID', 'event_type', 'product_id')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Products Table
# MAGIC 
# MAGIC A reference table with additional product information.

# COMMAND ----------

products = ecommerce_bronze.select('product_id', 'brand', 'price', 'category_id').dropDuplicates(['product_id'])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categories Table
# MAGIC 
# MAGIC A reference table with additional product category information.

# COMMAND ----------

categories = ecommerce_bronze.select('category_id', 'category', 'subcategory').dropDuplicates(['category_id'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Tables to Data Lake
# MAGIC 
# MAGIC Persist the four tables to Delta Lake tables for reporting.

# COMMAND ----------

customers.write.format("delta").option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/customers_gold/customers_gold").mode("overwrite").saveAsTable(f"customers_gold")

activities.write.format("delta").option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/activities_gold/activities_gold").mode("overwrite").saveAsTable(f"activities_gold")

products.write.format("delta").option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/products_gold/products_gold").mode("overwrite").saveAsTable(f"products_gold")

categories.write.format("delta").option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/categories_gold/categories_gold").mode("overwrite").saveAsTable(f"categories_gold")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM customers_gold
# MAGIC LIMIT 3;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM activities_gold
# MAGIC LIMIT 3;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM products_gold
# MAGIC LIMIT 3;

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM categories_gold
# MAGIC LIMIT 3;

# COMMAND ----------

# MAGIC %md Now, let's go back to the : <a href="https://github.com/microsoft/Azure-Databricks-Solution-Accelerator-Financial-Analytics-Customer-Revenue-Growth-Factor">`Readme`</a> to learn how to connect Databricks SQL to Power BI to visualize our metrics and ML results.
