# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Databricks notebook source
# MAGIC %md
# MAGIC Copyright (c) Microsoft Corporation. 
# MAGIC Licensed under the MIT license. 
# MAGIC # Data Extract
# MAGIC 
# MAGIC We are using an open source eCommerce store dataset from Kaggle: [eCommerce behavior data from multi category store](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store).  By running this notebook, you will be downloading the October 2019 and November 2019 datasets from Kaggle and December 2019 - April 2020 datasets from Google Drive. Before running this notebook, make sure you have entered your own credentials for Kaggle and have agreed to the Terms and Conditions of using this dataset.
# MAGIC 
# MAGIC Alternatively, you can manually download these from [Kaggle](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) and [Google Drive](https://drive.google.com/drive/folders/1Nan8X33H8xrXS5XhCKZmSpClFTCJsSpE), upload to an ADLS Gen 2 storage account via the Storage Browser in the Azure Portal or via [Azure Storage Explorer](https://azure.microsoft.com/en-us/products/storage/storage-explorer/), and [set up access to that ADLS Gen 2 storage account](https://learn.microsoft.com/en-us/azure/databricks/data/data-sources/azure/azure-storage)
# MAGIC 
# MAGIC In this notebook, we will also be cleaning the following data anomalies from this dataset into a version we can work with:
# MAGIC 1. Remove data missing that is missing brand and category values
# MAGIC 2. Filter to only keep brands and categories that are accurately mapped

# COMMAND ----------

# MAGIC %pip install kaggle

# COMMAND ----------

import requests, os
import urllib.request
import shutil
import time

# COMMAND ----------

# Clean Up Files
dbutils.fs.rm("dbfs:/revenue_growth_factors/raw/", recurse = True)
spark.sql("DROP DATABASE IF EXISTS `growth_factors`;")

# Create raw file directory
dbutils.fs.mkdirs("dbfs:/revenue_growth_factors/raw/")

# Create Database
spark.sql("CREATE DATABASE `growth_factors`;")

# COMMAND ----------

os.environ['kaggle_username'] = 'YOUR KAGGLE USERNAME HERE' # replace with your own credential here temporarily or set up a secret scope with your credential
os.environ['kaggle_key'] = 'YOUR KAGGLE KEY HERE' # replace with your own credential here temporarily or set up a secret scope with your credential

# COMMAND ----------

# DBTITLE 1,Download October and November 2019 Data from Kaggle to the Driver Node
# MAGIC %sh
# MAGIC cd /databricks/driver
# MAGIC export KAGGLE_USERNAME=$kaggle_username
# MAGIC export KAGGLE_KEY=$kaggle_key
# MAGIC kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store --unzip

# COMMAND ----------

# Validate that October and November 2019 files were downloaded to the driver node
display(dbutils.fs.ls("file:/databricks/driver/"))

# COMMAND ----------

# DBTITLE 1,Move Oct and Nov 2013 Data to Mounted ADLS
# Move October 2019 to ADLS
print("Moving October 2019 Data to ADLS...")
start_time = time.time()
dbutils.fs.mv("file:/databricks/driver/2019-Oct.csv", "dbfs:/revenue_growth_factors/raw/2019-Oct.csv")
print("Completed moving October 2019 Data to ADLS. It took", "{:.1f}".format(time.time() - start_time), "sec to run")
print("")

# Move November 2019 to ADLS
print("Moving November 2019 Data to ADLS...")
start_time = time.time()
dbutils.fs.mv("file:/databricks/driver/2019-Nov.csv", "dbfs:/revenue_growth_factors/raw/2019-Nov.csv")
print("Completed moving November 2019 Data to ADLS. It took", "{:.1f}".format(time.time() - start_time), "sec to run")

# COMMAND ----------

# Validate that October and November 2019 files were moved to DBFS
display(dbutils.fs.ls("dbfs:/revenue_growth_factors/raw/"))

# COMMAND ----------

def google_drive_downloader(year_month, download_id, uuid, tmp_file, mnt_dir, mnt_file):
  # Download the data to temp storage
  start_time = time.time()
  print(f"Downloading {year_month} Data to tmp storage...")
  urllib.request.urlretrieve(f"https://drive.google.com/uc?export=download&id={download_id}&confirm=t&uuid={uuid}", 
                             f"/tmp/{tmp_file}")
  print(f"Completed downloading {year_month} Data to tmp storage. It took", "{:.1f}".format(time.time() - start_time), "sec to run")
  
  # Move the data from temp storage to ADLS
  start_time = time.time()
  print(f"Moving {year_month} to ADLS...")
  shutil.move(f"/tmp/{tmp_file}", f"/dbfs/revenue_growth_factors/{mnt_dir}/{mnt_file}")
  print(f"Completed moving {year_month} to ADLS. It took", "{:.1f}".format(time.time() - start_time), "sec to run")
  print("")

# COMMAND ----------

# December 2019
google_drive_downloader("December 2019", "1qZIwMbMgMmgDC5EoMdJ8aI9lQPsWA3-P", "f78922c9-4bcf-4062-8ce1-4c088e8e1708", 
                        "2019-Dec.csv.gz", "raw", "2019-Dec.csv.gz")
print("Sleep for 15 sec...")
time.sleep(15)
print("")

# January 2020
google_drive_downloader("January 2020", "1x5ohrrZNhWQN4Q-zww0RmXOwctKHH9PT", "4f1f6c8b-9a2a-421a-9832-03d14c64b92d", 
                        "2020-Jan.csv.gz", "raw", "2020-Jan.csv.gz")
print("Sleep for 15 sec...")
time.sleep(15)
print("")

# February 2020
google_drive_downloader("February 2020", "1-Rov9fFtGJqb7_ePc6qH-Rhzxn0cIcKB", "bdfbdb76-0bc5-4169-ad3c-940ec5da17e5", 
                        "2020-Feb.csv.gz", "raw", "2020-Feb.csv.gz")
print("Sleep for 15 sec...")
time.sleep(15)
print("")

# March 2020
google_drive_downloader("March 2020", "1zr_RXpGvOWN2PrWI6itWL8HnRsCpyqz8", "915589a7-1947-4476-9e65-88c659437bd2", 
                        "2020-Mar.csv.gz", "raw", "2020-Mar.csv.gz")
print("Sleep for 15 sec...")
time.sleep(15)
print("")

# April 2020
google_drive_downloader("April 2020", "1g5WoIgLe05UMdREbxAjh0bEFgVCjA1UL", "8e5822d7-defc-47c5-973a-4a9b5cc3ef36", 
                        "2020-Apr.csv.gz", "raw", "2020-Apr.csv.gz")

# COMMAND ----------

# Validate that all files (Oct 2019 - April 2020) files were moved to the mounted ADLS
display(dbutils.fs.ls("dbfs:/revenue_growth_factors/raw/"))

# COMMAND ----------

# Validate files are correct with a test CSV read
spark.read.csv("dbfs:/revenue_growth_factors/raw/", header = True).limit(3).display()
