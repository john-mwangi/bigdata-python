# %% [markdown]
# # Objectives
# * Reading data into Spark
# * Using data.table, vaex and lazy queries
# * Running background jobs in Jupyter
# * Run queries on Google Collab from local database
# * Run H2O.ai to create ML model
# * Save results to parquet file
# * Use dbt and Airflow to orchestrate everything
# * Using GitHub Actions with dbt
# * Convert to script with classes
# * Create front end using Django
# * Test the following tools:
#   * DVC for data versioning
#   * ML flow for model versioning
#   * ML flow vs Air flow
#   * Model monitoring

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

from pyspark.sql import SparkSession
from pyspark import __version__ as py_ver
from py4j import __version__ as py4_ver

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

# %%
sc = SparkSession.builder.getOrCreate()
sc

# %% [markdown]
# ## Importing data
# ### From Pandas

# %%
pandas_df = pd.DataFrame(
    {
        "a": [1, 2, 3],
        "b": [2.0, 3.0, 4.0],
        "c": ["string1", "string2", "string3"],
        "d": [date(2000, 1, 1), date(2000, 2, 1), date(2000, 3, 1)],
        "e": [
            datetime(2000, 1, 1, 12, 0),
            datetime(2000, 1, 2, 12, 0),
            datetime(2000, 1, 3, 12, 0),
        ],
    }
)

sp_df = sc.createDataFrame(pandas_df)
sp_df

# %%
sp_df.printSchema()
sp_df.show()

# %% [markdown]
# ### From CSV

# %%
sp_csv = sc.read.csv("../inputs/task2_data1.csv")
sp_csv

# %%
sp_csv.show()

# %% [markdown]
# ### From SQL

# %% [markdown]
# Update conf/spark-defaults.conf to include the setting: `spark.driver.extraClassPath` = `E:\\Softwares\\postgresql-42.2.22.jar`. This can't be set through sparkConf(), the code chunk below will fail.
#
# Reference:
# * https://spark.apache.org/docs/latest/configuration.html
# * https://spark.apache.org/docs/latest/sql-data-sources-jdbc.html
# * https://stackoverflow.com/questions/30983982/how-to-use-jdbc-source-to-write-and-read-data-in-pyspark

# %%
# sc.sparkContext.stop()
# confs = [("spark.driver.extraClassPath", "E:\\Softwares\\postgresql-42.2.22.jar")]
# conf = sc.sparkContext._conf.setAll(confs)
# sc.sparkContext.stop()
# sc = SparkSession.builder.config(conf=conf).getOrCreate()
# sc.sparkContext._conf.getAll()

# %%
sc.sparkContext._conf.getAll()

# %%
sp_pg = sc.read.jdbc(
    url="jdbc:postgresql://localhost:5432/chinook",
    table="album",
    properties={"user": "postgres", "password": "john"},
)

sp_pg

# %%
sp_pg.show()

# %% [markdown]
# ## Exporting data
# ### To Pandas

# %%
sp_pg.toPandas()
