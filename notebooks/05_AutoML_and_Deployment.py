# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Databricks notebook source
# MAGIC %md
# MAGIC Copyright (c) Microsoft Corporation. 
# MAGIC Licensed under the MIT license. 
# MAGIC # AutoML & Deployment
# MAGIC 
# MAGIC This notebook walks you through how to develop and deploy a machine learning model to predict revenue growth without writing a single line of code.

# COMMAND ----------

# MAGIC %md
# MAGIC Start by Navigating to the Experiments Tab. Click on `Create AutoML Experiment`
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/1_automl_experiment.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Next, you will need to configure the AutoML pipeline with the following parameters:
# MAGIC * Cluster: ``
# MAGIC * ML Problem Type: `Classification`
# MAGIC * Dataset: `growth_factors.ecommerce_feature_eng`
# MAGIC * Experiment Name: revenue_growth_factors_automl
# MAGIC * Uncheck the boxes for `user_id`, `year`, and `month` as these are ID columns we don't want to include in the ML model
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/3_auto_ml_configs.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Click on advanced configuration and set the following parameters:
# MAGIC * Evaluation metric: `ROC/AUC`
# MAGIC * Positive label: `1`
# MAGIC * Click on `Start AutoML`
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/4_auto_ml_advanced_configs.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Browse the mlflow Experiment UI by clicking on the Experiments tab. Search for `revenue_growth_factors_automl`
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/5_mlflow_experiments.png">

# COMMAND ----------

# MAGIC %md
# MAGIC You can now see all of the ML models that were built using AutoML with links to open up the associated notebooks.
# MAGIC 
# MAGIC Let's click on `view notebook for best model` and `view data exploration notebook`
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/6_auto_ml_results.png">

# COMMAND ----------

# MAGIC %md
# MAGIC You can clone the glassbox generated data exploration and best model notebooks to your Repos folder. Use the following names:
# MAGIC * 08_AutoML_Data_Exploration
# MAGIC * 09_AutoML_Best_Experiment
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/7_clone.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's navigate back to the Experiments tab and find that best model experiment. Click into the experiment and then click on `Register Model`. 
# MAGIC 
# MAGIC This will package up the model and its dependencies into a Docker container to be able to deploy for batch or real-time inference.
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/8_register_model.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Let's name this model `revenue_growth_factors_automl`
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/9_register_model_2.png">

# COMMAND ----------

# MAGIC %md
# MAGIC You can browse for this registered model in the mlflow Model Registry by cliking on the Models Tab on the left and searching for `revenue_growth_factors_automl`
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/10_model_registry.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Let's click into the latest version of the model and promote this to our `Production Model`
# MAGIC 
# MAGIC Once we've done this, let's click on `Use model for inference` for batch deployment
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/12_promo_to_prod.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Set up batch inference with the following parameters:
# MAGIC * Model Version: `Production`
# MAGIC * Input Table: `growth_factors.ecommerce_feature_eng`
# MAGIC * Click on `Use model for batch inference`
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/13_batch_scoring_2.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Let's clone this notebook to our Repos folder and name it the following: `10_AutoML_Batch_Scoring`
# MAGIC <img src="https://raw.githubusercontent.com/isaac-gritz/growth-factors-images/main/images/14_batch_scoring_3.png">

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's use the mlflow apis to extract the metrics from this model to use for reporting: <a href="$./06_AutoML_Metrics">`AutoML Metrics`</a>
