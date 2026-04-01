import argparse
from pyspark.sql import SparkSession, functions as F
def clean_data(df):
    NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92
    NYC_LON_MIN, NYC_LON_MAX = -74.27, -73.68
    crit_cols = ["fare_amount", "trip_distance", "pickup_latitude", "pickup_longitude"]
    df = df.dropna(subset=crit_cols)
    df = df.filter(
        (F.col("pickup_latitude") >= NYC_LAT_MIN) & (F.col("pickup_latitude") <= NYC_LAT_MAX) &
        (F.col("pickup_longitude") >= NYC_LON_MIN) & (F.col("pickup_longitude") <= NYC_LON_MAX)
    )
    df = df.filter((F.col("fare_amount") > 0) & (F.col("trip_distance") > 0))
    df = df.filter((F.col("fare_amount") <= 200) & (F.col("trip_distance") <= 50))
    return df


FILES = {
    "2016-02": "yellow_tripdata_2016-02.csv",
    "2016-03": "yellow_tripdata_2016-03.csv",
}

NUMERIC_COLS = [
    "fare_amount",
    "trip_distance",
    "passenger_count",
    "RateCodeID",
    "pickup_longitude",
    "pickup_latitude",
]

CATEGORICAL_COLS = [
    "RateCodeID",
    "passenger_count",
]

SAMPLE_ROWS = 5


def build_spark():
    return SparkSession.builder.master("local[*]").appName("yellowTaxi-EDA-pyspark").getOrCreate()


def load_month(spark, fname, nrows=0):
    df = spark.read.format("csv").option("header", True).option("inferSchema", True).load(fname)
    if nrows > 0:
        df = df.limit(nrows)
    df = df.withColumn("tpep_pickup_datetime", F.to_timestamp(F.col("tpep_pickup_datetime")))
    df = df.withColumn("tpep_dropoff_datetime", F.to_timestamp(F.col("tpep_dropoff_datetime")))
    return df



def eda(df, name="DataFrame"):
    print(f"\n--- EDA for {name} ---\n")
    print(f"Row count: {df.count()}")
    print(f"Columns: {df.columns}")
    print("Schema:")
    df.printSchema()
    print("\nSample rows:")
    df.show(SAMPLE_ROWS)
    print("\nSummary statistics (numeric columns):")
    df.select(NUMERIC_COLS).describe().show()
    for col in NUMERIC_COLS:
        if col in df.columns:
            print(f"Percentiles for {col}:")
            quantiles = df.approxQuantile(col, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], 0.01)
            print(f"  1%: {quantiles[0]:.2f}, 5%: {quantiles[1]:.2f}, 25%: {quantiles[2]:.2f}, 50%: {quantiles[3]:.2f}, 75%: {quantiles[4]:.2f}, 95%: {quantiles[5]:.2f}, 99%: {quantiles[6]:.2f}")
    print("\nValue counts (categorical columns):")
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            print(f"Top values for {col}:")
            df.groupBy(col).count().orderBy(F.desc("count")).show(10)

    # 1. Data quality and cleaning checks
    print("\n Data Quality Checks ")
    # NYC bounds
    NYC_LAT_MIN, NYC_LAT_MAX = 40.49, 40.92
    NYC_LON_MIN, NYC_LON_MAX = -74.27, -73.68
    if "pickup_latitude" in df.columns and "pickup_longitude" in df.columns:
        invalid_coords = df.filter(
            (F.col("pickup_latitude") < NYC_LAT_MIN) | (F.col("pickup_latitude") > NYC_LAT_MAX) |
            (F.col("pickup_longitude") < NYC_LON_MIN) | (F.col("pickup_longitude") > NYC_LON_MAX)
        )
        print(f"Invalid pickup coordinates: {invalid_coords.count()} out of {df.count()}")
    if "fare_amount" in df.columns:
        neg_fare = df.filter(F.col("fare_amount") <= 0).count()
        print(f"Negative or zero fares: {neg_fare}")
    if "trip_distance" in df.columns:
        neg_dist = df.filter(F.col("trip_distance") <= 0).count()
        print(f"Negative or zero trip distances: {neg_dist}")
    print("Proportion of missing/null values per column:")
    total = df.count()
    for c in df.columns:
        nulls = df.filter(F.col(c).isNull()).count()
        print(f"  {c}: {nulls/total:.4f}")

    # 2. Temporal analysis
    print("\n Temporal Analysis ")
    if "tpep_pickup_datetime" in df.columns:
        df = df.withColumn("pickup_hour", F.hour(F.col("tpep_pickup_datetime")))
        df = df.withColumn("pickup_dayofweek", F.dayofweek(F.col("tpep_pickup_datetime")))
        print("Trips per hour:")
        df.groupBy("pickup_hour").count().orderBy("pickup_hour").show(24)
        print("Mean fare per hour:")
        if "fare_amount" in df.columns:
            df.groupBy("pickup_hour").agg(F.mean("fare_amount").alias("mean_fare")).orderBy("pickup_hour").show(24)
        print("Trips per day of week:")
        df.groupBy("pickup_dayofweek").count().orderBy("pickup_dayofweek").show()

    # 3. Outlier exploration
    print("\nOutlier")
    if "trip_distance" in df.columns:
        large_dist = df.filter(F.col("trip_distance") > 50).count()
        print(f"Trips with distance > 50 miles: {large_dist}")
    if "fare_amount" in df.columns:
        large_fare = df.filter(F.col("fare_amount") > 200).count()
        print(f"Trips with fare > $200: {large_fare}")

    # 4. Spatial sanity check
    print("\n--- Spatial Sanity Check ---")
    if "pickup_latitude" in df.columns and "pickup_longitude" in df.columns:
        out_nyc = df.filter(
            (F.col("pickup_latitude") < NYC_LAT_MIN) | (F.col("pickup_latitude") > NYC_LAT_MAX) |
            (F.col("pickup_longitude") < NYC_LON_MIN) | (F.col("pickup_longitude") > NYC_LON_MAX)
        ).count()
        print(f"Pickups outside NYC bounds: {out_nyc}")

    # 5. Feature relationships
    print("\nFeature Relationships")
    if set(["fare_amount", "trip_distance"]).issubset(df.columns):
        corr = df.stat.corr("fare_amount", "trip_distance")
        print(f"Correlation fare_amount vs. trip_distance: {corr:.3f}")
    if set(["tip_amount", "fare_amount"]).issubset(df.columns):
        corr = df.stat.corr("tip_amount", "fare_amount")
        print(f"Correlation tip_amount vs. fare_amount: {corr:.3f}")
    if set(["total_amount", "fare_amount", "tip_amount"]).issubset(df.columns):
        print("Sample total_amount vs. sum(fare+tip):")
        df_sample = df.select("total_amount", "fare_amount", "tip_amount").dropna().limit(5)
        for row in df_sample.collect():
            print(f"  total: {row['total_amount']}, fare+tip: {row['fare_amount']+row['tip_amount']}")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=str, default=None, help="Month key (2016-02 2016-03) or 'all'")
    parser.add_argument("--nrows", type=int, default=0, help="Limit number of rows (0 = all)")
    parser.add_argument("--showcols", action="store_true", help="Show all columns and exit")
    args = parser.parse_args()
    spark = build_spark()
    months = list(FILES.keys()) if args.month is None or args.month == "all" else [args.month]
    eda_results = {}
    for m in months:
        if m not in FILES:
            print(f"Unknown month: {m}")
            continue
        print(f"\nLoading {m}...")
        df = load_month(spark, FILES[m], args.nrows)
        if args.showcols:
            print(f"Columns in {m}:\n{df.columns}")
            continue
        print("\nCleaning data starts")
        df_clean = clean_data(df)
        print(f"Rows after cleaning: {df_clean.count()} (removed {df.count() - df_clean.count()})")
        eda(df_clean, name=m+" (CLEAN)")
        # Save cleaned data to CSV
        out_path = f"cleaned_{m}.csv"
        print(f"Saving cleaned data for {m} to {out_path} ...")
        df_clean.write.mode("overwrite").option("header", True).csv(out_path)
        eda_results[m] = df_clean
    # 6. Between-month comparison
    if len(eda_results) == 2:
        print("\nBetween-Month Comparison (2016-02 vs 2016-03) ")
        df1 = eda_results.get("2016-02")
        df2 = eda_results.get("2016-03")
        if df1 and df2:
            for col in ["fare_amount", "trip_distance", "passenger_count"]:
                if col in df1.columns and col in df2.columns:
                    mean1 = df1.select(F.mean(col)).first()[0]
                    mean2 = df2.select(F.mean(col)).first()[0]
                    std1 = df1.select(F.stddev(col)).first()[0]
                    std2 = df2.select(F.stddev(col)).first()[0]
                    print(f"{col}: Feb mean={mean1:.2f}, Mar mean={mean2:.2f} | Feb std={std1:.2f}, Mar std={std2:.2f}")
    spark.stop()

if __name__ == "__main__":
    main()
