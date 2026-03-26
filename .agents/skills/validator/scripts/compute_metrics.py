from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator

def evaluate_regression_model(predictions_df: DataFrame, label_col: str = "target", prediction_col: str = "prediction") -> dict:
    """
    Evaluates a regression model against a PySpark DataFrame with labels and predictions.
    This creates the metadata needed for the Validator Agent.
    
    Returns:
        A dictionary containing RMSE, R2, and MAE.
    """
    metrics = {}
    
    rmse_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="rmse")
    metrics["RMSE"] = rmse_evaluator.evaluate(predictions_df)
    
    r2_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="r2")
    metrics["R2"] = r2_evaluator.evaluate(predictions_df)
    
    mae_evaluator = RegressionEvaluator(labelCol=label_col, predictionCol=prediction_col, metricName="mae")
    metrics["MAE"] = mae_evaluator.evaluate(predictions_df)
    
    return metrics

def main():
    print("This script provides the evaluate_regression_model utility.")
    print("Use this to generate the evaluation_metrics for the LangGraph Validator Agent.")

if __name__ == "__main__":
    main()
