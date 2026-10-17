from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, min, max


# ---------------------------------------------------------
# 1. CREATE SPARK SESSION
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("MovieRecommendation-DataPreprocessing")
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


print("\n" + "=" * 60)
print("       DATA PREPROCESSING")
print("=" * 60)

print("\nRatings data loaded successfully.")


# ---------------------------------------------------------
# 4. CHECK NULL VALUES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("NULL VALUE CHECK")
print("=" * 60)

null_counts = ratings.select(
    count(col("UserID")).alias("UserID_Total"),
    count(col("MovieID")).alias("MovieID_Total"),
    count(col("Rating")).alias("Rating_Total")
)

null_counts.show()


# ---------------------------------------------------------
# 5. CHECK DUPLICATES
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

total_records = ratings.count()

unique_records = ratings.dropDuplicates(
    ["UserID", "MovieID"]
).count()

duplicate_count = total_records - unique_records

print("Total rating records :", total_records)
print("Unique user-movie pairs:", unique_records)
print("Duplicate records    :", duplicate_count)


# ---------------------------------------------------------
# 6. CHECK RATING RANGE
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RATING RANGE CHECK")
print("=" * 60)

ratings.select(
    min("Rating").alias("Minimum Rating"),
    max("Rating").alias("Maximum Rating"),
    avg("Rating").alias("Average Rating")
).show()


# ---------------------------------------------------------
# 7. CHECK INVALID RATINGS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("INVALID RATING CHECK")
print("=" * 60)

invalid_ratings = ratings.filter(
    (col("Rating") < 1) | (col("Rating") > 5)
)

print("Invalid rating count:", invalid_ratings.count())


# ---------------------------------------------------------
# 8. RATING DISTRIBUTION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("RATING DISTRIBUTION")
print("=" * 60)

ratings.groupBy("Rating") \
    .count() \
    .orderBy("Rating") \
    .show()


# ---------------------------------------------------------
# 9. CREATE MODEL DATA
# ---------------------------------------------------------

model_data = ratings.select(
    "UserID",
    "MovieID",
    "Rating"
)

print("\n" + "=" * 60)
print("MODEL DATA")
print("=" * 60)

model_data.printSchema()

model_data.show(5)


# ---------------------------------------------------------
# 10. FINAL RECORD COUNT
# ---------------------------------------------------------

print("\nTotal records prepared for ALS:", model_data.count())


# ---------------------------------------------------------
# 11. STOP SPARK
# ---------------------------------------------------------

spark.stop()

print("\n" + "=" * 60)
print("DATA PREPROCESSING COMPLETED")
print("=" * 60)