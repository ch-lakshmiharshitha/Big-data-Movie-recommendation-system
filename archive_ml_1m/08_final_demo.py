from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator

# ============================================================
# MOVIE RECOMMENDATION SYSTEM - FINAL DEMO
# ============================================================

PROJECT_PATH = "/mnt/d/sem7/BigData/Big data project"
DATA_PATH = PROJECT_PATH + "/data/raw/ml-1m"
MODEL_PATH = PROJECT_PATH + "/models/als_model"

spark = (
    SparkSession.builder
    .appName("MovieRecommendation-FinalDemo")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


def heading(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


# ============================================================
# 1. DATA INGESTION
# ============================================================

heading("MOVIE RECOMMENDATION SYSTEM")

print("\nLoading MovieLens 1M dataset...")

ratings = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(DATA_PATH + "/ratings.dat")
    .toDF("UserID", "MovieID", "Rating", "Timestamp")
)

ratings = ratings.select(
    col("UserID").cast("int"),
    col("MovieID").cast("int"),
    col("Rating").cast("float")
)

movies = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(DATA_PATH + "/movies.dat")
    .toDF("MovieID", "Title", "Genres")
)

print("Ratings loaded successfully.")
print("Movies loaded successfully.")

total_ratings = ratings.count()
total_users = ratings.select("UserID").distinct().count()
total_movies = movies.count()

print("\nDataset Statistics:")
print("Users   :", total_users)
print("Movies  :", total_movies)
print("Ratings :", total_ratings)


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

heading("DATA PREPARATION")

training_data, test_data = ratings.randomSplit(
    [0.8, 0.2],
    seed=42
)

print("\nTraining records:", training_data.count())
print("Testing records :", test_data.count())


# ============================================================
# 3. LOAD TRAINED ALS MODEL
# ============================================================

heading("ALS RECOMMENDATION MODEL")

print("\nLoading trained ALS model...")

model = ALSModel.load(MODEL_PATH)

print("Model loaded successfully.")

print("\nALS Parameters:")
print("Algorithm  : Alternating Least Squares (ALS)")
print("Rank       : 10")
print("MaxIter    : 10")
print("RegParam   : 0.1")


# ============================================================
# 4. MODEL EVALUATION
# ============================================================

heading("MODEL EVALUATION")

print("\nGenerating predictions on test data...")

predictions = model.transform(test_data)

predictions = predictions.dropna(
    subset=["prediction"]
)

rmse_evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="Rating",
    predictionCol="prediction"
)

mae_evaluator = RegressionEvaluator(
    metricName="mae",
    labelCol="Rating",
    predictionCol="prediction"
)

rmse = rmse_evaluator.evaluate(predictions)
mae = mae_evaluator.evaluate(predictions)

print("\nEvaluation Metrics:")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")


# ============================================================
# 5. SAMPLE PREDICTIONS
# ============================================================

print("\nSample Predictions:")

predictions.select(
    "UserID",
    "MovieID",
    "Rating",
    "prediction"
).show(5, truncate=False)


# ============================================================
# 6. GENERATE RECOMMENDATIONS
# ============================================================

heading("PERSONALIZED MOVIE RECOMMENDATIONS")

user_id = 1

print(f"\nGenerating recommendations for User ID: {user_id}")

user_df = ratings.filter(
    col("UserID") == user_id
).select("UserID").distinct()

recommendations = model.recommendForUserSubset(
    user_df,
    10
)

recommendations = (
    recommendations
    .select(
        col("UserID"),
        col("recommendations")
    )
    .withColumn(
        "recommendation",
        __import__("pyspark").sql.functions.explode(
            "recommendations"
        )
    )
    .select(
        "UserID",
        col("recommendation.MovieID").alias("MovieID"),
        col("recommendation.rating").alias("PredictedRating")
    )
)

final_recommendations = (
    recommendations
    .join(movies, on="MovieID", how="left")
    .select(
        "MovieID",
        "Title",
        "Genres",
        "PredictedRating"
    )
    .orderBy(
        col("PredictedRating").desc()
    )
)

print("\nTop 10 Recommended Movies:")

final_recommendations.show(
    10,
    truncate=False
)


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

heading("PROJECT SUMMARY")

print("\nDataset:")
print("MovieLens 1M")

print("\nBig Data Technology:")
print("Apache Spark")

print("\nRecommendation Algorithm:")
print("ALS - Alternating Least Squares")

print("\nModel Performance:")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")

print("\nRecommendation Example:")
print(f"User ID: {user_id}")
print("Top 10 personalized movie recommendations generated.")

print("\n" + "=" * 65)
print("       MOVIE RECOMMENDATION SYSTEM COMPLETED")
print("=" * 65)

spark.stop()