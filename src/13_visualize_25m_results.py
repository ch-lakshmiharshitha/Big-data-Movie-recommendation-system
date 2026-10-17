
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count
from pyspark.ml.recommendation import ALSModel


# ============================================================
# CONFIGURATION
# ============================================================

RATINGS_PATH = "hdfs:///bigdata_project/ml-25m/ratings.csv"
MOVIES_PATH = "hdfs:///bigdata_project/ml-25m/movies.csv"
MODEL_PATH = "hdfs:///bigdata_project/models/als_model_25m"

OUTPUT_DIR = "output/visualizations_25m"


# ============================================================
# START SPARK
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("MovieLens25M-Visualizations")
    .master("local[2]")
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("=" * 65)
print("       MOVIELENS 25M - DATA VISUALIZATION")
print("=" * 65)
print("No model training will be performed.")


try:
    # --------------------------------------------------------
    # LOAD RATINGS AND MOVIE METADATA FROM HDFS
    # --------------------------------------------------------

    print("\nLoading ratings from HDFS...")

    ratings = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(RATINGS_PATH)
        .select(
            col("userId").cast("int").alias("UserID"),
            col("movieId").cast("int").alias("MovieID"),
            col("rating").cast("float").alias("Rating")
        )
    )

    print("Loading movie metadata from HDFS...")

    movies = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(MOVIES_PATH)
        .select(
            col("movieId").cast("int").alias("MovieID"),
            col("title").alias("Title"),
            col("genres").alias("Genres")
        )
    )

    # --------------------------------------------------------
    # VISUALIZATION 1: RATING DISTRIBUTION
    # --------------------------------------------------------

    print("\n[1/3] Creating rating distribution chart...")

    rating_counts = (
        ratings
        .groupBy("Rating")
        .agg(count("*").alias("Count"))
        .orderBy("Rating")
        .collect()
    )

    rating_values = [row["Rating"] for row in rating_counts]
    rating_frequencies = [row["Count"] for row in rating_counts]

    plt.figure(figsize=(10, 6))
    plt.bar(
        [str(r) for r in rating_values],
        rating_frequencies
    )
    plt.title("MovieLens 25M - Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Number of Ratings")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path1 = os.path.join(
        OUTPUT_DIR, "rating_distribution_25m.png"
    )
    plt.savefig(path1, dpi=200)
    plt.close()

    print("Saved:", path1)

    # --------------------------------------------------------
    # VISUALIZATION 2: TOP 10 MOST-RATED MOVIES
    # --------------------------------------------------------

    print("\n[2/3] Creating most-rated movies chart...")

    movie_counts = (
        ratings
        .groupBy("MovieID")
        .agg(count("*").alias("RatingCount"))
        .join(movies, on="MovieID", how="inner")
        .orderBy(col("RatingCount").desc(), col("Title").asc())
        .limit(10)
        .collect()
    )

    movie_titles = [row["Title"] for row in movie_counts][::-1]
    movie_frequencies = [
        row["RatingCount"] for row in movie_counts
    ][::-1]

    plt.figure(figsize=(12, 7))
    plt.barh(movie_titles, movie_frequencies)
    plt.title("MovieLens 25M - Top 10 Most-Rated Movies")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Movie")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path2 = os.path.join(
        OUTPUT_DIR, "top_10_most_rated_movies.png"
    )
    plt.savefig(path2, dpi=200)
    plt.close()

    print("Saved:", path2)

    # --------------------------------------------------------
    # VISUALIZATION 3: ACTUAL VS PREDICTED RATINGS
    # --------------------------------------------------------

    print("\n[3/3] Loading the existing ALS model from HDFS...")

    model = ALSModel.load(MODEL_PATH)

    print("Existing model loaded. No training performed.")

    print("Recreating the test split using seed 42...")

    training_data, test_data = ratings.randomSplit(
        [0.8, 0.2], seed=42
    )

    print("Generating predictions from the existing model...")

    predictions = (
        model.transform(test_data)
        .dropna(subset=["prediction"])
        .select("Rating", "prediction")
        .sample(False, 0.002, seed=42)
        .limit(5000)
        .collect()
    )

    if not predictions:
        raise RuntimeError(
            "No valid predictions were collected for the chart."
        )

    actual = [row["Rating"] for row in predictions]
    predicted = [row["prediction"] for row in predictions]

    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.25, s=12)

    plt.plot(
        [0.5, 5.0],
        [0.5, 5.0],
        linestyle="--",
        label="Ideal prediction"
    )

    plt.title("MovieLens 25M - Actual vs Predicted Ratings")
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.xlim(0.5, 5.0)
    plt.ylim(0.5, 5.5)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    path3 = os.path.join(
        OUTPUT_DIR, "actual_vs_predicted_25m.png"
    )
    plt.savefig(path3, dpi=200)
    plt.close()

    print("Saved:", path3)
    print("Predictions plotted:", len(predictions))

    print("\n" + "=" * 65)
    print("       VISUALIZATION COMPLETED")
    print("=" * 65)
    print("Output directory:", os.path.abspath(OUTPUT_DIR))
    print("1. rating_distribution_25m.png")
    print("2. top_10_most_rated_movies.png")
    print("3. actual_vs_predicted_25m.png")

finally:
    spark.stop()
