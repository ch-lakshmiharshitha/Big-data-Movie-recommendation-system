from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
import time

# ============================================================
# MOVIELENS 25M - ALS MODEL EVALUATION
# HDFS -> SPARK -> SPARK MLLIB ALS
# ============================================================

spark = (
    SparkSession.builder
    .appName("MovieLens25M-ALS-Evaluation")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

# ============================================================
# PATHS
# ============================================================

RATINGS_PATH = "hdfs:///bigdata_project/ml-25m/ratings.csv"
MODEL_PATH = "hdfs:///bigdata_project/models/als_model_25m"

# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("             MOVIELENS 25M - ALS MODEL EVALUATION")
print("=" * 70)

print("\nPipeline:")
print("HDFS -> Apache Spark -> Spark MLlib ALS -> Evaluation")

# ============================================================
# LOAD RATINGS FROM HDFS
# ============================================================

print("\nLoading ratings from HDFS...")

ratings = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(RATINGS_PATH)
)

ratings = ratings.select(
    col("userId").cast("int").alias("UserID"),
    col("movieId").cast("int").alias("MovieID"),
    col("rating").cast("float").alias("Rating")
)

print("Ratings loaded successfully.")

# ============================================================
# DATASET INFORMATION
# ============================================================

print("\nDataset Statistics:")

total_ratings = ratings.count()
total_users = ratings.select("UserID").distinct().count()
total_movies = ratings.select("MovieID").distinct().count()

print("Users        :", total_users)
print("Movies       :", total_movies)
print("Ratings      :", total_ratings)

# ============================================================
# CREATE SAME TRAIN / TEST SPLIT
# ============================================================

print("\nCreating test dataset...")

start_split = time.time()

training_data, test_data = ratings.randomSplit(
    [0.8, 0.2],
    seed=42
)

test_count = test_data.count()

split_time = time.time() - start_split

print("Testing records :", test_count)
print(f"Split time      : {split_time:.2f} seconds")

# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("\nLoading trained ALS model...")

model = ALSModel.load(MODEL_PATH)

print("ALS model loaded successfully.")

# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

prediction_start = time.time()

predictions = model.transform(test_data)

# Remove rows where ALS could not generate a prediction
predictions = predictions.dropna(
    subset=["prediction"]
)

prediction_count = predictions.count()

prediction_time = time.time() - prediction_start

print("Predictions generated successfully.")
print("Prediction records :", prediction_count)
print(f"Prediction time    : {prediction_time:.2f} seconds")

# ============================================================
# RMSE
# ============================================================

print("\nCalculating RMSE...")

rmse_evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="Rating",
    predictionCol="prediction"
)

rmse = rmse_evaluator.evaluate(predictions)

# ============================================================
# MAE
# ============================================================

print("Calculating MAE...")

mae_evaluator = RegressionEvaluator(
    metricName="mae",
    labelCol="Rating",
    predictionCol="prediction"
)

mae = mae_evaluator.evaluate(predictions)

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("                    SAMPLE PREDICTIONS")
print("=" * 70)

predictions.select(
    "UserID",
    "MovieID",
    "Rating",
    "prediction"
).show(10, truncate=False)

# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("                  EVALUATION RESULTS")
print("=" * 70)

print("\nDataset:")
print("  Users              :", total_users)
print("  Movies             :", total_movies)
print("  Total Ratings      :", total_ratings)
print("  Test Records       :", test_count)
print("  Valid Predictions  :", prediction_count)

print("\nModel:")
print("  Rank               : 10")
print("  MaxIter            : 10")
print("  RegParam           : 0.1")
print("  User Blocks        : 20")
print("  Item Blocks        : 20")

print("\nEvaluation Metrics:")
print(f"  RMSE               : {rmse:.4f}")
print(f"  MAE                : {mae:.4f}")

print("\nPipeline:")
print("  HDFS")
print("    ↓")
print("  Apache Spark")
print("    ↓")
print("  Spark MLlib ALS")
print("    ↓")
print("  RMSE / MAE")

print("\n" + "=" * 70)
print("             MODEL EVALUATION COMPLETED")
print("=" * 70)

# ============================================================
# STOP SPARK
# ============================================================

spark.stop()