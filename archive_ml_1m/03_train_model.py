from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col


# ---------------------------------------------------------
# 1. CREATE SPARK SESSION
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("MovieRecommendation-ALS-Training")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# 2. DATA PATH
# ---------------------------------------------------------

data_path = "/mnt/d/sem7/BigData/Big data project/data/raw/ml-1m"

ratings_path = data_path + "/ratings.dat"


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
# 4. SELECT DATA REQUIRED BY ALS
# ---------------------------------------------------------

ratings = ratings.select(
    col("UserID").cast("int"),
    col("MovieID").cast("int"),
    col("Rating").cast("float")
)


print("\n" + "=" * 60)
print("       ALS MOVIE RECOMMENDATION MODEL")
print("=" * 60)

print("\nTotal ratings:", ratings.count())


# ---------------------------------------------------------
# 5. TRAIN / TEST SPLIT
# ---------------------------------------------------------

training_data, test_data = ratings.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\nTraining records:", training_data.count())
print("Testing records :", test_data.count())


# ---------------------------------------------------------
# 6. CREATE ALS MODEL
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING ALS MODEL")
print("=" * 60)

print("\nALS Parameters:")
print("Rank       : 10")
print("MaxIter    : 10")
print("RegParam   : 0.1")

als = ALS(
    userCol="UserID",
    itemCol="MovieID",
    ratingCol="Rating",
    rank=10,
    maxIter=10,
    regParam=0.1,
    coldStartStrategy="drop",
    nonnegative=True
)


# ---------------------------------------------------------
# 7. TRAIN MODEL
# ---------------------------------------------------------

print("\nTraining model...")

model = als.fit(training_data)

print("ALS model training completed.")


# ---------------------------------------------------------
# 8. SAVE MODEL
# ---------------------------------------------------------

model_path = "/mnt/d/sem7/BigData/Big data project/models/als_model"

print("\nSaving trained model...")

model.write().overwrite().save(model_path)

print("Model saved successfully.")
print("Location:", model_path)


# ---------------------------------------------------------
# 9. STOP SPARK
# ---------------------------------------------------------

spark.stop()

print("\n" + "=" * 60)
print("ALS TRAINING COMPLETED")
print("=" * 60)