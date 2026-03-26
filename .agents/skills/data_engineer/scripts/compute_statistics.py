from pyspark.sql import DataFrame
from pyspark.sql.functions import mean, stddev, min as spark_min, max as spark_max
import json

def compute_dataset_statistics(df: DataFrame, numerical_columns: list) -> dict:
    """
    Computes summary statistics for a given PySpark DataFrame.
    This is used by the Data Engineer Agent to detect Data Drift.
    
    Returns:
        A dictionary containing mean, stddev, min, max for each specified column.
    """
    stats = {}
    for col_name in numerical_columns:
        agg_df = df.select(
            mean(col_name).alias("mean"),
            stddev(col_name).alias("stddev"),
            spark_min(col_name).alias("min"),
            spark_max(col_name).alias("max")
        ).collect()[0]
        
        stats[col_name] = {
            "mean": agg_df["mean"],
            "stddev": agg_df["stddev"],
            "min": agg_df["min"],
            "max": agg_df["max"]
        }
    return stats

def main():
    print("This script provides the compute_dataset_statistics utility.")
    print("Use this to generate the metadata state for the LangGraph Data Engineer Agent.")

if __name__ == "__main__":
    main()
