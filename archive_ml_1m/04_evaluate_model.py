from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col


# ---------------------------------------------------------
# 1. CREATE SPARK SESSION
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("MovieRecommendation-ALS-Evaluation")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# 2. DATA PATHS
# ---------------------------------------------------------

data_path = "/mnt/d/sem7/BigData/Big data project/data/raw/ml-1m"

ratings_path = data_path + "/ratings.dat"

model_path = "/mnt/d/sem7/BigData/Big data project/models/als_model"


# ---------------------------------------------------------
# 3. LOAD RATINGS
# ---------------------------------------------------------

ratings = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(ratings_path)
    .toDF("UserID", "MovieID", "Rating", "Timestamp")
)


# ---------------------------------------------------------
# 4. PREPARE DATA
# ---------------------------------------------------------

ratings = ratings.select(
    col("UserID").cast("int"),
    col("MovieID").cast("int"),
    col("Rating").cast("float")
)


# ---------------------------------------------------------
# 5. RECREATE SAME TRAIN / TEST SPLIT
# ---------------------------------------------------------

training_data, test_data = ratings.randomSplit(
    [0.8, 0.2],
    seed=42
)


# ---------------------------------------------------------
# 6. LOAD TRAINED ALS MODEL
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("       ALS MODEL EVALUATION")
print("=" * 60)

print("\nLoading trained ALS model...")

model = ALSModel.load(model_path)

print("Model loaded successfully.")


# ---------------------------------------------------------
# 7. GENERATE PREDICTIONS
# ---------------------------------------------------------

print("\nGenerating predictions on test data...")

predictions = model.transform(test_data)

predictions = predictions.dropna(
    subset=["prediction"]
)

print("Predictions generated successfully.")

print("\nSample predictions:")
predictions.select(
    "UserID",
    "MovieID",
    "Rating",
    "prediction"
).show(10)


# ---------------------------------------------------------
# 8. CALCULATE RMSE
# ---------------------------------------------------------

evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="Rating",
    predictionCol="prediction"
)

rmse = evaluator.evaluate(predictions)


# ---------------------------------------------------------
# 9. CALCULATE MAE
# ---------------------------------------------------------

mae_evaluator = RegressionEvaluator(
    metricName="mae",
    labelCol="Rating",
    predictionCol="prediction"
)

mae = mae_evaluator.evaluate(predictions)


# ---------------------------------------------------------
# 10. DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("       MODEL EVALUATION RESULTS")
print("=" * 60)

print("\nRMSE :", round(rmse, 4))
print("MAE  :", round(mae, 4))

print("\n" + "=" * 60)
print("ALS MODEL EVALUATION COMPLETED")
print("=" * 60)


# ---------------------------------------------------------
# 11. STOP SPARK
# ---------------------------------------------------------

spark.stop()