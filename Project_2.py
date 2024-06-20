# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC The following dataset was obtained in kaggle, the dataset captures information about the duration of taxi trips throughout NYC.
# MAGIC
# MAGIC This dataset has a total of 11 columns and over 1 million rows
# MAGIC
# MAGIC The idea for this project is to try and fit a linear regression model into the training data in order to predict the trip_duration. As you may already know this problem is a regression problem and not a classification problem as our intent is not to predict a category.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Loading Data into Databricks

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/NYC.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
taxi_df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(taxi_df)

# COMMAND ----------

#Creating a Table NYC_taxi 

temp_table_name = "NYC_taxi"
taxi_df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Exploration

# COMMAND ----------

taxi_df.show()
#We may need to remove some records posing as outliers. The column we are trying to predict "trip_duration" is represented in seconds.

# COMMAND ----------

taxi_df.display()
#Passanger count and trip duration are perfectily linear meaining that the more passangers in the taxi the longer the trip.

# COMMAND ----------

# MAGIC %sql
# MAGIC --Here we are trying to find whether there are any duplicate features in the dataset, query returned no result therefore there is no duplication.
# MAGIC SELECT id
# MAGIC FROM NYC_taxi
# MAGIC GROUP BY id
# MAGIC HAVING COUNT(*) > 1

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT     passenger_count,
# MAGIC           AVG(trip_duration)
# MAGIC FROM       NYC_taxi
# MAGIC GROUP BY   passenger_count
# MAGIC ORDER BY passenger_count ASC
# MAGIC
# MAGIC --As you can see some of the trips are recorded with 0 passangers, in the metadata it is explained that the number of passangers are recorded by manually by the taxi driver. We will exclude these records during the Data Cleaning process. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT     passenger_count AS Number_of_Passengers, 
# MAGIC            count(id) AS Number_of_Records
# MAGIC FROM       NYC_taxi
# MAGIC GROUP BY  passenger_count
# MAGIC ORDER BY  passenger_count ASC
# MAGIC
# MAGIC -- This query is to know the number of records affected by the 0 passanger and also showing that the most common taxi ride is with 1 passanger only. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT pickup_datetime, passenger_count,
# MAGIC MONTH(pickup_datetime) AS start_month
# MAGIC FROM NYC_taxi
# MAGIC --we are trying to find the months with the lowest/highest pickup times in this case, while all are close together, the months March/April have the highest count of pickups, when digging further into this, it has been documented that the best time to visit NYC is in the spring and early summer which would correlate to these months and the lowest pickup month is january. Winter is undoubtedly the most challenging time to visit NYC, from December to February, the city experiences freezing temperatures, biting winds, and heavy snowfall, making exploring the city difficult and uncomfortable. This visualization also shows that the data only goes up to the month of June and is not for the full year.--

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT count(*) FROM NYC_taxi 
# MAGIC WHERE trip_duration = (SELECT min(trip_duration) FROM NYC_taxi)
# MAGIC
# MAGIC --If we query the min(trip_duration) we get 1 which means that there are some trips in our dataset that have 1 second trips. We wanted to see how many trips were affected so we ran the query below to find out that 33 trips were listed with 1 second taxi runs. These are not really possible in real life so we will remove them. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT round(trip_duration) AS Trip_Duration_Hours FROM NYC_taxi
# MAGIC ORDER BY trip_duration ASC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT round(max(trip_duration) / 60, 2) AS Maximun_Trip_Duration_Minutes 
# MAGIC FROM NYC_taxi
# MAGIC --The longest trip in our dataset is too long for minutes so we will show it in hours in the next query. This trip will be filtered out as it poses as an outliers. 

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT round((trip_duration / 60) / 60, 2) AS Trip_Duration_Hours 
# MAGIC FROM NYC_taxi
# MAGIC ORDER BY trip_duration DESC
# MAGIC --We can see that there are 4 trips that are too long, these trips are outliers and so we will remove them from the dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC ##Creating a Spark Session Object

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import when
spark=SparkSession.builder.appName('supervised_ml').getOrCreate()

# COMMAND ----------

#Showing the count of all columns and rows
print((taxi_df.count(), len(taxi_df.columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC ##Data Cleaning and Transformation

# COMMAND ----------

#We need to change the column store_and_fwd_flag from a string to a number in order to include it into our Model.  
taxi_df = taxi_df.withColumn('store_and_fwd_flag', when(taxi_df['store_and_fwd_flag'] == 'N', 0)
                                         .when(taxi_df['store_and_fwd_flag'] == 'Y', 1)
                                         .otherwise(taxi_df['store_and_fwd_flag']))

# COMMAND ----------

#Even thought we changed the store_and_fwd_flag column from N and Y to 1 and 0 the column remains a string. We need to change it to numeric if we want to use it for this Model. 
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

taxi_df = taxi_df.withColumn("store_and_fwd_flag", col("store_and_fwd_flag").cast(IntegerType()))

# COMMAND ----------

# Convert timestamp columns to numerical values in order to use it in our Model without this conversion we would get an error when creating the features column(e.g., seconds since epoch)
from pyspark.sql.functions import unix_timestamp
taxi_df = taxi_df.withColumn("pickup_datetime", unix_timestamp("pickup_datetime"))

# COMMAND ----------

#There are 2112 records that exeed taxi trips over 3 hours long. We are going to filter them out of the dataframe for a better distributed dataset. 
from pyspark.sql.functions import col
count_greater_than_10800 = taxi_df.filter(col("trip_duration") > 10800).count()
print("Number of rows where 'trip_duration' > 10800:", count_greater_than_10800)

# COMMAND ----------

#Saving the new filtered data into taxi_df dataframe excluding all trips with more than 3 hours.
taxi_df = taxi_df.filter(col("trip_duration") <= 10800)

# COMMAND ----------

#Im counting the amount of records that have less than 30 seconds of a taxi trip. 
count_less_than_30_seconds = taxi_df.filter(col("trip_duration") < 30).count()
print("Number of rows where 'trip_duration' < 30:", count_less_than_30_seconds)

# COMMAND ----------

#Saving the new filtered data into taxi_df
taxi_df = taxi_df.filter(col("trip_duration") > 30)

# COMMAND ----------

#Removing the unnecessary alphabetical characters 'id' before all the id numbers in the id column
from pyspark.sql.functions import col, regexp_replace
taxi_df = taxi_df.withColumn('id', regexp_replace(col('id'), '^id', ''))
taxi_df.show()

# COMMAND ----------

#This is to filter out any ride that has no passanger in the car. 
taxi_df = taxi_df.filter(col("passenger_count") > 0)

# COMMAND ----------

#In order to add the column id to our features we need to change it to an interger
from pyspark.sql.functions import col
taxi_df = taxi_df.withColumn('id', col('id').cast('integer'))
taxi_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Splitting the Data

# COMMAND ----------

train, test = taxi_df.randomSplit([.80, .20], seed=42)

print(f"""There are {train.count()} rows in the training set,
and {test.count()} in the test set""")

# COMMAND ----------

# MAGIC %md
# MAGIC ##  Feature Engineering

# COMMAND ----------

from pyspark.ml.linalg import Vector
from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# Assemble columns into a feature vector but this time including all the features from the dataset
from pyspark.ml.feature import StandardScaler

vecAssembler = VectorAssembler(inputCols=['id','vendor_id','pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag'],
                                outputCol='features')

vecTrainDF = vecAssembler.transform(train)

vecTrainDF.select('id','vendor_id', 'pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'features','trip_duration').show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Building the Model and Fitting the training Data

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="trip_duration")

lrModel = lr.fit(vecTrainDF)

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[vecAssembler, lr])

pipelineModel = pipeline.fit(train)

# COMMAND ----------

#Predicted values of the testing data are shown under the prediction column
predDF = pipelineModel.transform(test)
predDF.select("passenger_count", "features", "trip_duration", "prediction").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Evaluating the model's performance

# COMMAND ----------

#Evaluating the model using R2
from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol= 'prediction', labelCol="trip_duration", metricName="r2")
r2 = evaluator.evaluate(predDF)
print("R-squared (r2) on test data = %g" % r2)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Improving the Model

# COMMAND ----------

from pyspark.ml.feature import PCA
#Importing Principal Component Analysis 

# COMMAND ----------

# Passing our column names into the input columns list
input_cols = ['vendor_id','pickup_datetime', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']

# Creating a StandardScaler object and set the input and output columns
scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# COMMAND ----------

#Fitting the scaler to the input data
scaler_model = scaler.fit(vecTrainDF)

# Transforming the input data to obtain the scaled output
scaled_data = scaler_model.transform(vecTrainDF)

# Reviewing our newly transformed dataframe with the scaled_features column
scaled_data.show()

# COMMAND ----------

# Now we re-train and re-test our model
train,test = scaled_data.randomSplit([0.80, 0.20])

# COMMAND ----------

# Fitting the model to our training data
lr_model = lr.fit(train)

# Making predictions on our test data and viewing the new column containing the predicted values. 
predictions_df=lr_model.transform(test)
predictions_df.show()

# COMMAND ----------

# Evaluating our model after PCA
model_predictions=lr_model.evaluate(test)
model_predictions.r2

# COMMAND ----------

#Cluster has been Terminated
