from pyspark.sql import SparkSession


# ---------------------------------------------------------
# 1. CREATE SPARK SESSION
# ---------------------------------------------------------

spark = (
    SparkSession.builder
    .appName("MovieRecommendationSystem")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")


# ---------------------------------------------------------
# 2. DATASET PATHS
# ---------------------------------------------------------

data_path = "/mnt/d/sem7/BigData/Big data project/data/raw/ml-1m"

movies_path = data_path + "/movies.dat"
ratings_path = data_path + "/ratings.dat"
users_path = data_path + "/users.dat"

# ---------------------------------------------------------
# 3. LOAD MOVIES
# ---------------------------------------------------------

movies = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(movies_path)
    .toDF("MovieID", "Title", "Genres")
)


# ---------------------------------------------------------
# 4. LOAD RATINGS
# ---------------------------------------------------------

ratings = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(ratings_path)
    .toDF("UserID", "MovieID", "Rating", "Timestamp")
)


# ---------------------------------------------------------
# 5. LOAD USERS
# ---------------------------------------------------------

users = (
    spark.read
    .option("sep", "::")
    .option("inferSchema", "true")
    .csv(users_path)
    .toDF("UserID", "Gender", "Age", "Occupation", "ZipCode")
)


# ---------------------------------------------------------
# 6. DISPLAY DATASET INFORMATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("       LARGE-SCALE MOVIE RECOMMENDATION SYSTEM")
print("=" * 60)

print("\nDataset successfully loaded using Apache Spark.\n")

print("Number of Movies :", movies.count())
print("Number of Ratings:", ratings.count())
print("Number of Users  :", users.count())


# ---------------------------------------------------------
# 7. DISPLAY SCHEMAS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("MOVIES DATAFRAME SCHEMA")
print("=" * 60)

movies.printSchema()


print("\n" + "=" * 60)
print("RATINGS DATAFRAME SCHEMA")
print("=" * 60)

ratings.printSchema()


print("\n" + "=" * 60)
print("USERS DATAFRAME SCHEMA")
print("=" * 60)

users.printSchema()


# ---------------------------------------------------------
# 8. DISPLAY SAMPLE RECORDS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SAMPLE MOVIES")
print("=" * 60)

movies.show(5, truncate=False)


print("\n" + "=" * 60)
print("SAMPLE RATINGS")
print("=" * 60)

ratings.show(5)


print("\n" + "=" * 60)
print("SAMPLE USERS")
print("=" * 60)

users.show(5)


# ---------------------------------------------------------
# 9. STOP SPARK
# ---------------------------------------------------------

spark.stop()

print("\nSpark session stopped successfully.")
print("Data loading test completed.")