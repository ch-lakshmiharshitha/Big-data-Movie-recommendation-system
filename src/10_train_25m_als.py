from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALS
import time

# ============================================================
# MOVIELENS 25M - ALS MODEL TRAINING
# HDFS -> SPARK -> SPARK MLLIB
# ============================================================

spark = (
    SparkSession.builder
    .appName("MovieLens25M-ALS-Training")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

RATINGS_PATH = "hdfs:///bigdata_project/ml-25m/ratings.csv"
# MODEL_PATH = "hdfs:///bigdata_project/models/als_model_25m"
MODEL_PATH = "hdfs:///bigdata_project/models/als_model_25m_training_capture"

# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("           MOVIELENS 25M - ALS MODEL TRAINING")
print("=" * 70)

print("\nPipeline:")
print("HDFS -> Apache Spark -> Spark MLlib ALS")

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

print("\nCounting dataset records...")

total_ratings = ratings.count()
total_users = ratings.select("UserID").distinct().count()
total_movies = ratings.select("MovieID").distinct().count()

print("\nDataset Statistics:")
print("Users        :", total_users)
print("Movies       :", total_movies)
print("Ratings      :", total_ratings)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\nCreating training and testing datasets...")

start_split = time.time()

training_data, test_data = ratings.randomSplit(
    [0.8, 0.2],
    seed=42
)

training_count = training_data.count()
test_count = test_data.count()

split_time = time.time() - start_split

print("\nTraining records:", training_count)
print("Testing records :", test_count)
print(f"Split time      : {split_time:.2f} seconds")

# ============================================================
# ALS CONFIGURATION
# ============================================================

print("\n" + "=" * 70)
print("                    ALS CONFIGURATION")
print("=" * 70)

rank = 10
max_iter = 10
reg_param = 0.1
num_user_blocks = 20
num_item_blocks = 20

print("\nRank          :", rank)
print("MaxIter       :", max_iter)
print("RegParam      :", reg_param)
print("User Blocks   :", num_user_blocks)
print("Item Blocks   :", num_item_blocks)
print("Spark Master  : local[2]")

als = ALS(
    userCol="UserID",
    itemCol="MovieID",
    ratingCol="Rating",
    rank=rank,
    maxIter=max_iter,
    regParam=reg_param,
    numUserBlocks=num_user_blocks,
    numItemBlocks=num_item_blocks,
    coldStartStrategy="drop",
    nonnegative=True
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("                  TRAINING ALS MODEL")
print("=" * 70)

print("\nTraining started...")

training_start = time.time()

model = als.fit(training_data)

training_time = time.time() - training_start

print("\nALS model training completed.")
print(f"Training time: {training_time:.2f} seconds")
print(f"Training time: {training_time / 60:.2f} minutes")

# ============================================================
# SAVE MODEL
# ============================================================

model.write().overwrite().save(MODEL_PATH)

print("Model saved successfully.")
print("Model path:", MODEL_PATH)

# # Verify that the model files were actually created
# import os

# if os.path.exists(MODEL_PATH):
#     print("Model directory verified successfully.")
#     print("Model contents:")
#     for item in os.listdir(MODEL_PATH):
#         print("  -", item)
# else:
#     print("ERROR: Model directory was not found after saving.")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("                    TRAINING SUMMARY")
print("=" * 70)

print("\nDataset:")
print("  Users          :", total_users)
print("  Movies         :", total_movies)
print("  Ratings        :", total_ratings)

print("\nTraining:")
print("  Training rows  :", training_count)
print("  Testing rows   :", test_count)

print("\nALS Parameters:")
print("  Rank           :", rank)
print("  MaxIter        :", max_iter)
print("  RegParam       :", reg_param)
print("  User Blocks    :", num_user_blocks)
print("  Item Blocks    :", num_item_blocks)

print("\nPipeline:")
print("  HDFS")
print("    ↓")
print("  Apache Spark")
print("    ↓")
print("  Spark MLlib ALS")

print("\nModel:")
print(" ", MODEL_PATH)

print("\n" + "=" * 70)
print("             ALS TRAINING COMPLETED")
print("=" * 70)

# ============================================================
# STOP SPARK
# ============================================================

spark.stop()