from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# ============================================================
# MOVIELENS 25M - HDFS DATA LOADING
# ============================================================

spark = (
    SparkSession.builder
    .appName("MovieLens25M-HDFS-Loading")
    .master("local[*]")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

RATINGS_PATH = "hdfs:///bigdata_project/ml-25m/ratings.csv"
MOVIES_PATH = "hdfs:///bigdata_project/ml-25m/movies.csv"

print("\n" + "=" * 65)
print("        MOVIELENS 25M - HDFS DATA LOADING")
print("=" * 65)

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
    col("rating").cast("float").alias("Rating"),
    col("timestamp").cast("long").alias("Timestamp")
)

print("Ratings loaded successfully.")

# ============================================================
# LOAD MOVIES FROM HDFS
# ============================================================

print("\nLoading movies from HDFS...")

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

print("Movies loaded successfully.")

# ============================================================
# DATASET STATISTICS
# ============================================================

print("\n" + "=" * 65)
print("             DATASET STATISTICS")
print("=" * 65)

total_ratings = ratings.count()
total_users = ratings.select("UserID").distinct().count()
total_movies = movies.count()

print("\nTotal Users   :", total_users)
print("Total Movies  :", total_movies)
print("Total Ratings :", total_ratings)

# ============================================================
# SAMPLE DATA
# ============================================================

print("\n" + "=" * 65)
print("             SAMPLE RATINGS")
print("=" * 65)

ratings.show(5, truncate=False)

print("\n" + "=" * 65)
print("             SAMPLE MOVIES")
print("=" * 65)

movies.show(5, truncate=False)

print("\n" + "=" * 65)
print("       HDFS → SPARK LOADING COMPLETED")
print("=" * 65)

spark.stop()