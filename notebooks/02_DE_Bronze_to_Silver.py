# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Databricks notebook source
# MAGIC %md
# MAGIC Copyright (c) Microsoft Corporation. 
# MAGIC Licensed under the MIT license. 
# MAGIC # Data Engineering
# MAGIC 
# MAGIC After cleaning the data, we transform it in order to capture relevant metrics for ML modeling. These metrics  capture information related to:
# MAGIC * Users & sessions
# MAGIC * Buying behavior
# MAGIC * Product details - brand, category, subcategories, product
# MAGIC 
# MAGIC Results are written to a Silver Delta Lake table

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
# MAGIC ## Read in Data from Delta Lake

# COMMAND ----------

df = spark.table("ecommerce_bronze")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Transformation
# MAGIC 
# MAGIC ### Growth Indicator
# MAGIC 
# MAGIC Classify customers as growth (1) or no growth (0) based on the month-over-month change in net revenue.
# MAGIC 
# MAGIC 1. Growth if there is a >10% net revenue increase
# MAGIC 1. No growth if there is a >10% net revenue decrease

# COMMAND ----------

# get monthly revenue
growth = df.filter(col('event_type') == 'purchase') \
    .withColumn('revenue', df['price'].cast('double'))\
    .groupBy('user_id', 'year', 'month') \
    .sum('revenue') \
    .withColumnRenamed('sum(revenue)', 'total_net_revenue') \
    .orderBy('user_id', 'year', 'month')

# COMMAND ----------

# get deltas for previous month
window_specs = Window.partitionBy('user_id').orderBy('user_id', 'year', 'month')
growth_lag = growth.withColumn('last_month_revenue', lag(growth.total_net_revenue).over(window_specs).cast('double'))
growth_delta = growth_lag.withColumn('delta_net_revenue', (growth_lag.total_net_revenue - growth_lag.last_month_revenue).cast('double'))

# identify growth vs. no growth customers
# growth defined as +/-10% revenue month-over-month

df_growth_a = growth_delta.withColumn('percent_delta_revenue', growth_delta['delta_net_revenue']/growth_delta['last_month_revenue'].cast('double'))
df_growth = df_growth_a.withColumn('growth', 
        when(df_growth_a['percent_delta_revenue'] > .1, 1)
        .when(df_growth_a['percent_delta_revenue'] < -.1, 0)) \
        .drop('last_month_revenue', 'delta_net_revenue', 'total_net_revenue', 'percent_delta_revenue') \
        .filter(col('growth').isNotNull())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Aggregated Metrics
# MAGIC 
# MAGIC Transform data to produce metrics related to user sessions, buying behavior, and product categories. All features are aggregated on a per-user, per-month basis.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Session & Buying Metrics
# MAGIC 
# MAGIC * Number of sessions
# MAGIC * Average session duration
# MAGIC * Average conversion rate
# MAGIC * Average order value
# MAGIC * Average cart abandon rate

# COMMAND ----------

# DBTITLE 1,Sessions Per User
sessions_per_user_per_month = df.groupBy('user_id', 'year', 'month') \
    .agg(countDistinct('user_session').alias('sessions_per_user_per_month')) \
    .fillna({'sessions_per_user_per_month': 0}) \
    .orderBy('user_id', 'year', 'month')

# COMMAND ----------

# DBTITLE 1,Average Session Duration
# time between start & end of each session, aggregated per user per month
session_durations = df.groupBy('user_id', 'year', 'month', 'user_session') \
    .agg(
        min('event_time').alias('session_start_time'),
        max('event_time').alias('session_end_time')) \
    .withColumn('session_duration', col('session_end_time').cast("long") - col('session_start_time').cast("long")) \
    .drop('user_session', 'session_start_time', 'session_end_time')

avg_session_duration_per_user_per_month = session_durations.groupBy('user_id', 'year', 'month') \
    .agg(mean('session_duration').cast('double').alias('avg_session_duration_per_user_per_month')) \
    .orderBy('user_id', 'year', 'month')

# COMMAND ----------

# DBTITLE 1,Average Conversion Rate
# avg # purchases / # views per user per month
avg_conversion_rate_per_user_per_month = df.groupBy('user_id', 'year', 'month') \
    .agg(
        count(when(col('event_type') == 'view', True)).alias('num_views'),
        count(when(col('event_type') == 'purchase', True)).alias('num_purchases')) \
    .fillna({'num_views': 0, 'num_purchases': 0}) \
    .withColumn('avg_conversion_rate_per_user_per_month', (col('num_purchases')/col('num_views')).cast('double')) \
    .drop('num_views', 'num_purchases') \
    .orderBy('user_id', 'year', 'month')

# COMMAND ----------

# DBTITLE 1,Average Order Value
# price per user per month, for purchases only
avg_order_value_per_user_per_month = df.filter(col('event_type') == 'purchase') \
    .groupBy('user_id', 'year', 'month') \
    .agg(mean('price').cast('double').alias('avg_order_value_per_user_per_month')) \
    .orderBy('user_id', 'year', 'month')

# COMMAND ----------

# DBTITLE 1,Average Cart Abandon Rate
# items that were added to cart, but not purchased
abandon_rate_per_session = df.filter((col('event_type') == 'purchase') | (col('event_type') == 'cart')) \
    .groupBy('user_id', 'year', 'month', 'user_session', 'product_id') \
    .pivot('event_type').agg(count('product_id')) \
    .fillna({'cart':0, 'purchase':0}) \
    .withColumn('cart_abandon_rate', (col('cart')-col('purchase'))/col('cart'))

avg_cart_abandon_rate = abandon_rate_per_session.groupBy('user_id', 'year', 'month') \
    .agg(mean('cart_abandon_rate').cast('double').alias('avg_cart_abandon_rate'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Brand, Subcategory, & Product Metrics
# MAGIC 
# MAGIC For the top 5 most popular values in each product-related category (brand, subcategory, and product_id), identify the frequency of user clickstream interactions (product views, add to cart, and purchases).

# COMMAND ----------

# reusable function
## event_type = clickstream activity (view, cart, purchase)
## match_type = product-related column (brand, subcategory, product_id)

def get_top_5(df, event_type, match_type):

    # get list of top 5
    top_5_list = df.filter(col('event_type')==event_type).groupBy(match_type).pivot('event_type') \
        .agg(count('user_session')).orderBy(desc(event_type)) \
        .select(match_type).limit(5).rdd.flatMap(lambda x: x).collect()
        
    # filter df for top 5
    top_5_df = df.where(col(match_type).isin(top_5_list)) \
        .filter(col('event_type')==event_type) \
        .groupBy('user_id', 'year', 'month') \
        .pivot(match_type) \
        .agg(count('user_session'))

    # reformat types / naming convention
    if (event_type == 'view'):
        event_type = 'viewed'
    elif (event_type == 'cart'):
        event_type = 'added'
    else:
        event_type = 'purchased'

    # convert to binary & count columns
    for i in range(1, len(top_5_list)+1):
        i_name = top_5_list[i-1]
        top_5_df = top_5_df.withColumn(f'{match_type}_{i_name}_{event_type}_binary', when(col(i_name).isNotNull(), 1).otherwise(0)) \
            .withColumnRenamed(f'{i_name}', f'{match_type}_{i_name}_{event_type}_count') \
            .fillna({f'{match_type}_{i_name}_{event_type}_count': 0})

    return top_5_df

# COMMAND ----------

# brands
top_brands_viewed = get_top_5(df, 'view', 'brand')
top_brands_added = get_top_5(df, 'cart', 'brand')
top_brands_purchased = get_top_5(df, 'purchase', 'brand')

# subcategories
top_subcategories_viewed = get_top_5(df, 'view', 'subcategory')
top_subcategories_added = get_top_5(df, 'cart', 'subcategory')
top_subcategories_purchased = get_top_5(df, 'purchase', 'subcategory')

# products
top_products_viewed = get_top_5(df, 'view', 'product_id')
top_products_added = get_top_5(df, 'cart', 'product_id')
top_products_purchased = get_top_5(df, 'purchase', 'product_id')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join DataFrames into Single DataFrame

# COMMAND ----------

# join dfs
def join_dfs (df_list):
    joined_df = df_growth
    for l in df_list:
        joined_df = joined_df.join(l, ['user_id', 'year', 'month'], how='left')
    return joined_df

features_df = join_dfs([sessions_per_user_per_month, \
    avg_session_duration_per_user_per_month, \
    avg_conversion_rate_per_user_per_month, \
    avg_order_value_per_user_per_month, \
    avg_cart_abandon_rate, \
    top_brands_viewed, top_brands_added, top_brands_purchased, \
    top_subcategories_viewed, top_subcategories_added, top_subcategories_purchased, \
    top_products_viewed, top_products_added, top_products_purchased
    ]).fillna(0)

# COMMAND ----------

# write transformed data to spark table
features_df.write.format("delta").option("mergeSchema", "true").option("path", f"dbfs:/revenue_growth_factors/ecommerce_silver").mode("overwrite").saveAsTable("ecommerce_silver")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ecommerce_silver LIMIT 3

# COMMAND ----------

# MAGIC %md Next, we analyze the transformed data and select the features that will be used in the model: <a href="$./03_ML_Feature_Engineering">`ML Feature Engineering`</a>
