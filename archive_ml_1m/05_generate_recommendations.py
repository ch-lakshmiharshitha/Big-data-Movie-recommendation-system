from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import col


# ---------------------------------------------------------
# 1. CREATE SPARK SESSION
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("MovieRecommendation-GenerateRecommendations")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# 2. DATA AND MODEL PATHS
# ---------------------------------------------------------

data_path = "/mnt/d/sem7/BigData/Big data project/data/raw/ml-1m"

movies_path = data_path + "/movies.dat"

model_path = "/mnt/d/sem7/BigData/Big data project/models/als_model"


# ---------------------------------------------------------
# 3. USER ID
# ---------------------------------------------------------

user_id = 1


# ---------------------------------------------------------
# 4. LOAD MOVIE DATA
# ---------------------------------------------------------

movies = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(movies_path)
    .toDF("MovieID", "Title", "Genres")
)

movies = movies.select(
    col("MovieID").cast("int"),
    col("Title"),
    col("Genres")
)


# ---------------------------------------------------------
# 5. LOAD TRAINED ALS MODEL
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("       MOVIE RECOMMENDATION SYSTEM")
print("=" * 60)

print("\nLoading trained ALS model...")

model = ALSModel.load(model_path)

print("Model loaded successfully.")


# ---------------------------------------------------------
# 6. CREATE USER DATAFRAME
# ---------------------------------------------------------

user_df = spark.createDataFrame(
    [(user_id,)],
    ["UserID"]
)


# ---------------------------------------------------------
# 7. GENERATE RECOMMENDATIONS
# ---------------------------------------------------------

print("\nGenerating recommendations for User ID:", user_id)

recommendations = model.recommendForUserSubset(
    user_df,
    10
)


# ---------------------------------------------------------
# 8. EXPLODE RECOMMENDATIONS
# ---------------------------------------------------------

recommendations = recommendations.selectExpr(
    "UserID",
    "explode(recommendations) as recommendation"
)

recommendations = recommendations.select(
    col("UserID"),
    col("recommendation.MovieID").alias("MovieID"),
    col("recommendation.rating").alias("PredictedRating")
)


# ---------------------------------------------------------
# 9. JOIN WITH MOVIE TITLES
# ---------------------------------------------------------

final_recommendations = (
    recommendations
    .join(movies, on="MovieID", how="left")
    .select(
        "MovieID",
        "Title",
        "Genres",
        "PredictedRating"
    )
    .orderBy(col("PredictedRating").desc())
)


# ---------------------------------------------------------
# 10. DISPLAY RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("       TOP 10 RECOMMENDED MOVIES")
print("=" * 60)

final_recommendations.show(
    10,
    truncate=False
)


# ---------------------------------------------------------
# 11. STOP SPARK
# ---------------------------------------------------------

spark.stop()

print("\n" + "=" * 60)
print("RECOMMENDATIONS GENERATED SUCCESSFULLY")
print("=" * 60)