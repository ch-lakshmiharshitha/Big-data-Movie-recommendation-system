from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark.ml.recommendation import ALSModel


# ============================================================
# CONFIGURATION
# ============================================================

USER_ID = 100
TOP_N = 10

RATINGS_PATH = "hdfs:///bigdata_project/ml-25m/ratings.csv"
MOVIES_PATH = "hdfs:///bigdata_project/ml-25m/movies.csv"
MODEL_PATH = "hdfs:///bigdata_project/models/als_model_25m"


# ============================================================
# START SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("MovieLens25M-GenerateRecommendations")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


print("\n" + "=" * 65)
print("       MOVIELENS 25M MOVIE RECOMMENDATION SYSTEM")
print("=" * 65)


# ============================================================
# LOAD ALS MODEL
# ============================================================

print("\nLoading ALS model from HDFS...")

model = ALSModel.load(MODEL_PATH)

print("ALS model loaded successfully.")


# ============================================================
# LOAD MOVIES
# ============================================================

print("\nLoading movie information from HDFS...")

movies = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(MOVIES_PATH)
)

movies = movies.select(
    col("movieId").cast("int").alias("MovieID"),
    col("title").alias("Title"),
    col("genres").alias("Genres")
)

print("Movie information loaded successfully.")


# ============================================================
# CREATE USER DATAFRAME
# ============================================================

user_df = spark.createDataFrame(
    [(USER_ID,)],
    ["UserID"]
)


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

print(f"\nGenerating recommendations for User ID: {USER_ID}...")

# Generate more than TOP_N so that already-rated movies
# can be removed before selecting the final TOP_N.
recommendations = model.recommendForUserSubset(
    user_df,
    TOP_N + 20
)


# ============================================================
# EXPLODE RECOMMENDATIONS
# ============================================================

recommendations = recommendations.select(
    col("UserID"),
    explode(col("recommendations")).alias("recommendation")
)

recommendations = recommendations.select(
    col("UserID"),
    col("recommendation.MovieID").alias("MovieID"),
    col("recommendation.rating").alias("PredictedRating")
)


# ============================================================
# LOAD MOVIES ALREADY RATED BY USER
# ============================================================

print("Removing movies already rated by the user...")

ratings = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(RATINGS_PATH)
)

rated_movies = (
    ratings
    .filter(col("userId") == USER_ID)
    .select(
        col("movieId").cast("int").alias("MovieID")
    )
    .distinct()
)


# ============================================================
# FILTER ALREADY-RATED MOVIES
# ============================================================

new_recommendations = (
    recommendations
    .join(
        rated_movies,
        on="MovieID",
        how="left_anti"
    )
)


# ============================================================
# JOIN WITH MOVIE DETAILS
# ============================================================

final_recommendations = (
    new_recommendations
    .join(
        movies,
        on="MovieID",
        how="left"
    )
    .select(
        "MovieID",
        "Title",
        "Genres",
        "PredictedRating"
    )
    .orderBy(
        col("PredictedRating").desc()
    )
    .limit(TOP_N)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print(f"        TOP {TOP_N} MOVIE RECOMMENDATIONS")
print(f"        USER ID: {USER_ID}")
print("=" * 65)

final_recommendations.show(
    TOP_N,
    truncate=False
)


# ============================================================
# SAVE RECOMMENDATIONS TO HDFS
# ============================================================

OUTPUT_PATH = (
    f"hdfs:///bigdata_project/output/"
    f"recommendations_user_{USER_ID}"
)

print("\nSaving recommendations to HDFS...")

(
    final_recommendations
    .coalesce(1)
    .write
    .mode("overwrite")
    .option("header", "true")
    .csv(OUTPUT_PATH)
)

print("Recommendations saved successfully.")
print("HDFS output path:", OUTPUT_PATH)


# ============================================================
# END
# ============================================================

print("\n" + "=" * 65)
print("       RECOMMENDATION GENERATION COMPLETED")
print("=" * 65)

spark.stop()