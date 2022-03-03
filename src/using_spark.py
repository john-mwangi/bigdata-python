# %% [markdown]
# # Objectives
# Big data processing and Machine Learning using Apache Spark.

# %% [markdown]
# # Packages

# %%
import pandas as pd
from datetime import datetime, date
import sys

# %%
# Importing module from a different location
sys.path.insert(
    0,
    "C:\\Users\\User\\AppData\\Local\\spark\\spark-3.1.1-bin-hadoop3.2\\python",
)

from pyspark import __version__ as py_ver
from py4j import __version__ as py4_ver

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import isnull, isnan, count, when
from pyspark.sql.functions import countDistinct
from pyspark.sql.functions import mean, max, stddev

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    VectorIndexer,
    IndexToString,
    VectorAssembler,
    OneHotEncoder,
)

# %% [markdown]
# # Pyspark
# Reference:
# * https://phoenixnap.com/kb/install-spark-on-windows-10
# * https://spark.apache.org/docs/latest/api/python/index.html
# * https://www.tutorialspoint.com/pyspark/pyspark_environment_setup.htm
#
# We will use a previous Spark installation: <br>
# $spark_home <br>
# [1] "C:\\Users\\User\\AppData\\Local\\spark\\spark-3.1.1-bin-hadoop3.2"

# %%
py_ver
py4_ver

# Python executable path
sys.executable

# %% [markdown]
# ## Init & Config

# %%
sc = SparkSession.builder.getOrCreate()
sc

# %%
sc.sparkContext._conf.getAll()

# %% [markdown]
# ## Importing data
# ### From CSV

# %%
sp_csv = sc.read.csv("../inputs/uap_om/task2_data1.csv", header=True)
sp_csv

# %%
sp_csv.printSchema()

# %%
sp_csv.show()

# %% [markdown]
# ### From SQL
#
# Update `conf\spark-defaults.conf` to include the setting: `spark.driver.extraClassPath` = `E:\\Softwares\\postgresql-42.2.22.jar`. This can't be set through sparkConf() at runtime, the code chunk below will fail.
#
# Reference:
# * https://spark.apache.org/docs/latest/configuration.html
# * https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html
# * https://stackoverflow.com/questions/30983982/how-to-use-jdbc-source-to-write-and-read-data-in-pyspark

# %%
sp_sql = sc.read.jdbc(
    url="jdbc:postgresql://localhost:5432/chinook",
    table="album",
    properties={"user": "postgres", "password": "john"},
)

sp_sql

# %%
sp_sql.show()

# %% [markdown]
# ### From Pandas
# When reading from a pandas dataframe, you need to manually create a schema in the event that Spark is not able to infer the correct data type. If you have numerous columns, this can become tedious. Below is a shorter way of creating a schema that maps all columns from pandas as string columns in Spark.
#
# Essentially, you need to create a schema from df.head that you can edit manually if need be.

# %%
pandas_df = pd.read_csv("../inputs/uap_om/task2_data1.csv")
pandas_df.dtypes

# %%
pd_schema = sc.createDataFrame(pandas_df.head()).schema
pd_schema

# %%
sp_pd = sc.createDataFrame(data=pandas_df, schema=pd_schema)
sp_pd

# %%
sp_pd.show()

# %% [markdown]
# ## Data wrangling
# Common data wrangling tasks... In general there is a lot of similarity with pandas.

# %%
traindemographics = sc.read.csv(
    "../inputs/lending/traindemographics.csv", header=True
)
trainperf = sc.read.csv("../inputs/lending/trainperf.csv", header=True)

# %%
# Unique records
traindemographics = traindemographics.dropDuplicates(subset=["customerid"])
trainperf = trainperf.dropDuplicates(subset=["customerid"])

# %%
# Merging tables
lending_merged = traindemographics.join(trainperf, on="customerid", how="left")
lending_merged.columns[:10]

# %%
# Info about your columns
lending_merged.printSchema()
lending_merged.dtypes
lending_merged.columns

# %%
# Summary stats
summary_stats = lending_merged.describe()

# %%
# There is no inbuilt transpose method
summ_cols = summary_stats.toPandas().transpose().iloc[0, :].tolist()
summ_df = summary_stats.toPandas().transpose().iloc[1:, :]
summ_df.columns = summ_cols
summ_df

# %%
# Selecting columns
lending_merged.select("customerid", "birthdate", "bank_account_type").show(5)

# %%
# Filtering
lending_merged.filter(
    (lending_merged.bank_account_type == "Savings")
    | (lending_merged.good_bad_flag == "Good")
).select(["customerid", "bank_account_type", "good_bad_flag"]).show()

# %%
# Counting nulls
lending_merged.filter(lending_merged.good_bad_flag.isNull()).count()

# %%
# SQL version: when() is necessary
lending_merged.select(
    [
        count(when(condition=isnull("good_bad_flag"), value="null")).alias(
            "null_count"
        )
    ]
).show()

# %%
# Using a dict
counts = {
    col: lending_merged.filter(lending_merged[col].isNull()).count()
    for col in lending_merged.columns[:3] + ["good_bad_flag"]
}
counts

# %%
# Using SQL/select
lending_merged.select(
    [
        count(when(condition=isnan(col) | isnull(col), value=col)).alias(col)
        for col in lending_merged.columns[:3] + ["good_bad_flag"]
    ]
).show()

# %%
# Counts by group
lending_merged.groupBy(["bank_account_type", "good_bad_flag"]).agg(
    countDistinct("good_bad_flag").alias("counts")
).show()

# %%
# Grouped average, max
lending_merged.groupBy(["bank_account_type", "good_bad_flag"]).agg(
    mean("loanamount").alias("avg_loanamount")
).show()

# %%
# Simple average
lending_merged.select(mean("loanamount").alias("avg_loan")).show()

# %% [markdown]
# ## Retrieving results
# After you're satisfied with your Spark results, you can convert then into a pandas dataframe. Spark runs lazy queries and `collect` executes them and retrieves them from remote to memory as a list.

# %%
sp_res = lending_merged.groupBy(["bank_account_type", "good_bad_flag"]).agg(
    mean("loanamount").alias("avg_loanamount")
)

sp_res

# %%
pd.DataFrame(sp_res.collect(), columns=sp_res.columns)

# %% [markdown]
# ## Machine Learning
# Machine Learning using Spark. MLLib was superseded by ML.
#
# References:
# * https://spark.apache.org/docs/latest/ml-guide.html
# * https://stackoverflow.com/questions/32277576/how-to-handle-categorical-features-with-spark-ml
# * https://towardsdatascience.com/a-guide-to-exploit-random-forest-classifier-in-pyspark-46d6999cb5db
# * https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier

# %% [markdown]
# ### Sampling

# %%
# Sampling
lending_merged.select(["customerid"]).sample(
    withReplacement=False, fraction=0.1
).show()

# %%
# Stratified sampling
lending_merged.select(["customerid", "good_bad_flag"]).sampleBy(
    col="good_bad_flag", fractions={"Good": 0.1, "Bad": 0.3}
).groupBy("good_bad_flag").agg(count("good_bad_flag").alias("count")).show()

# %% [markdown]
# ### Complete columns

# %%
nulls_res = lending_merged.select(
    [
        count(when(condition=isnan(col) | isnull(col), value=col)).alias(col)
        for col in lending_merged.columns
    ]
)

# %%
nulls_df = pd.DataFrame(nulls_res.collect(), columns=nulls_res.columns)
nulls_df.shape

# %%
nulls_df.melt()["value"] == 0
complete_cols = nulls_df.melt()[
    nulls_df.melt()["value"] == 0
].variable.tolist()
complete_cols.remove("customerid")
complete_cols

# %%
rf_data_raw = lending_merged.select(complete_cols + ["good_bad_flag"]).filter(
    condition=lending_merged.good_bad_flag.isNotNull()
)

# %%
# Original data with only complete columns
rf_data_raw.show(5)

# %% [markdown]
# ### Label indexing

# %%
# Set label columns
label_indexer = StringIndexer(inputCol="good_bad_flag", outputCol="label").fit(
    rf_data_raw
)
rf_data_label_trans = label_indexer.transform(rf_data_raw)

# %%
# This adds an indexed label to the dataframe
rf_data_label_trans.show(5)

# %% [markdown]
# ### Feature indexing

# %%
# Prepare column names for the columns that will be indexed
complete_idx = [f"{col}_idx" for col in complete_cols]

# %%
# Transform adds the indexed features to the dataframe
feat_indexer = StringIndexer(
    inputCols=complete_cols, outputCols=complete_idx
).fit(rf_data_label_trans)
rf_data_feat_trans = feat_indexer.transform(rf_data_label_trans)

# %%
rf_data_feat_trans.show(5)

# %% [markdown]
# ### OneHot Encoding

# %%
# Prepare column names for columns that will be dummified
complete_dums = [f"{col}_dum" for col in complete_cols]

# %%
# Perform OHE
oh_encoder = OneHotEncoder(
    inputCols=complete_idx, outputCols=complete_dums
).fit(rf_data_feat_trans)
rf_data_dum_trans = oh_encoder.transform(rf_data_feat_trans)

# %%
rf_data_dum_trans.select(complete_dums + ["label"]).show(5)

# %% [markdown]
# ### Vector Assembly

# %%
# Perform combine columns into a single index
vector_assembler = VectorAssembler(
    inputCols=complete_dums, outputCol="features"
)
rf_data_feats_trans = vector_assembler.transform(rf_data_dum_trans)

# %%
rf_data_feats_trans.select("features", "label").show(5)

# %%
# This is now the dataframe that'll be used for training
rf_data_full = rf_data_feats_trans.select("features", "label")

# %% [markdown]
# ### Train-test Split

# %%
train_data, test_data = rf_data_full.randomSplit(weights=[0.7, 0.3], seed=123)

# %% [markdown]
# ### Model Training

# %%
# Random forest model
rf = RandomForestClassifier(
    featuresCol="features", labelCol="label", numTrees=5
)
rf_model = rf.fit(train_data)
rf_model

# %% [markdown]
# ### Prediction

# %%
pred_res = rf_model.transform(train_data)
pred_res.show()

# %%
# Convert predictions to original labels
label_converter = IndexToString(
    inputCol="prediction", outputCol="pred_label", labels=label_indexer.labels
)

# %%
label_converter.transform(pred_res).show()

# %% [markdown]
# ## Exporting data
# ### To SQL

# %%
sp_csv.write.jdbc(
    url="jdbc:postgresql://localhost:5432/chinook",
    table="spark_csv",
    mode="overwrite",
    properties={"user": "postgres", "password": "john"},
)

# %% [markdown]
# ### To parquet, csv
# Spark only supports writing to HDFS file systems. Writing locally requires converting reading to memory first: `df.toPandas()` then `pd.to_parquet()`.
