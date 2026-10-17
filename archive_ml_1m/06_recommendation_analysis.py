from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    avg,
    min,
    max
)


# ---------------------------------------------------------
# 1. CREATE SPARK SESSION
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("MovieRecommendation-Analysis")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# 2. DATA PATHS
# ---------------------------------------------------------

data_path = "/mnt/d/sem7/BigData/Big data project/data/raw/ml-1m"

ratings_path = data_path + "/ratings.dat"
movies_path = data_path + "/movies.dat"

output_path = "/mnt/d/sem7/BigData/Big data project/output"


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

ratings = ratings.select(
    col("UserID").cast("int"),
    col("MovieID").cast("int"),
    col("Rating").cast("float"),
    col("Timestamp").cast("long")
)


# ---------------------------------------------------------
# 4. LOAD MOVIES
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
# 5. BASIC DATASET STATISTICS
# ---------------------------------------------------------

total_ratings = ratings.count()
total_users = ratings.select("UserID").distinct().count()
total_movies = ratings.select("MovieID").distinct().count()

average_rating = ratings.select(
    avg("Rating")
).first()[0]

minimum_rating = ratings.select(
    min("Rating")
).first()[0]

maximum_rating = ratings.select(
    max("Rating")
).first()[0]


# ---------------------------------------------------------
# 6. DISPLAY DATASET SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("             MOVIELENS DATASET ANALYSIS")
print("=" * 65)

print("\nDataset Summary")
print("-" * 65)

print("Total users       :", total_users)
print("Total movies      :", total_movies)
print("Total ratings     :", total_ratings)
print("Average rating    :", round(average_rating, 4))
print("Minimum rating    :", minimum_rating)
print("Maximum rating    :", maximum_rating)


# ---------------------------------------------------------
# 7. RATING DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("             RATING DISTRIBUTION")
print("=" * 65)

rating_distribution = (
    ratings
    .groupBy("Rating")
    .agg(count("*").alias("RatingCount"))
    .orderBy("Rating")
)

rating_distribution.show(10)


# ---------------------------------------------------------
# 8. MOST RATED MOVIES
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("             MOST RATED MOVIES")
print("=" * 65)

most_rated_movies = (
    ratings
    .groupBy("MovieID")
    .agg(
        count("*").alias("RatingCount"),
        avg("Rating").alias("AverageRating")
    )
    .join(movies, on="MovieID", how="left")
    .select(
        "MovieID",
        "Title",
        "Genres",
        "RatingCount",
        "AverageRating"
    )
    .orderBy(col("RatingCount").desc())
)

most_rated_movies.show(
    10,
    truncate=False
)


# ---------------------------------------------------------
# 9. HIGHEST-RATED MOVIES
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("             HIGHEST-RATED MOVIES")
print("=" * 65)

highest_rated_movies = (
    ratings
    .groupBy("MovieID")
    .agg(
        count("*").alias("RatingCount"),
        avg("Rating").alias("AverageRating")
    )
    .filter(col("RatingCount") >= 100)
    .join(movies, on="MovieID", how="left")
    .select(
        "MovieID",
        "Title",
        "Genres",
        "RatingCount",
        "AverageRating"
    )
    .orderBy(col("AverageRating").desc())
)

highest_rated_movies.show(
    10,
    truncate=False
)


# ---------------------------------------------------------
# 10. SAVE ANALYSIS RESULTS
# ---------------------------------------------------------

rating_distribution.write.mode("overwrite").option(
    "header", "true"
).csv(output_path + "/rating_distribution")

most_rated_movies.write.mode("overwrite").option(
    "header", "true"
).csv(output_path + "/most_rated_movies")

highest_rated_movies.write.mode("overwrite").option(
    "header", "true"
).csv(output_path + "/highest_rated_movies")


# ---------------------------------------------------------
# 11. COMPLETION
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("             ANALYSIS COMPLETED")
print("=" * 65)

print("\nResults saved to:")
print(output_path)

print("\nGenerated outputs:")
print(" - rating_distribution")
print(" - most_rated_movies")
print(" - highest_rated_movies")


# ---------------------------------------------------------
# 12. STOP SPARK
# ---------------------------------------------------------

spark.stop()