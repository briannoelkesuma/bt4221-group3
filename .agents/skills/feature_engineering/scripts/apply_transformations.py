from pyspark.sql import DataFrame
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler, Imputer
from pyspark.ml import Pipeline

def apply_feature_engineering_plan(df: DataFrame, plan: dict) -> DataFrame:
    """
    Executes the PySpark MLlib operations decided by the Feature Engineering Agent.
    
    Args:
        df: PySpark DataFrame
        plan: A parsed JSON dictionary matching the FeatureEngineeringPlan structure.
    """
    stages = []
    
    # 1. Drop features
    if plan.get("features_to_drop"):
        df = df.drop(*plan["features_to_drop"])
        
    # 2. Imputation
    num_imputations = plan.get("numerical_imputation", {})
    if num_imputations.get("columns"):
        strategy = num_imputations.get("strategy", "mean")
        cols = num_imputations["columns"]
        out_cols = [f"{c}_imputed" for c in cols]
        imputer = Imputer(inputCols=cols, outputCols=out_cols).setStrategy(strategy)
        stages.append(imputer)
        
    # 3. Categorical Encoding
    cat_encodings = plan.get("categorical_encoding", {})
    if cat_encodings.get("columns"):
        for c in cat_encodings["columns"]:
            indexer = StringIndexer(inputCol=c, outputCol=f"{c}_indexed", handleInvalid="keep")
            stages.append(indexer)
            
    # 4. Vector Assembler & Scaling (Requires vector input)
    scaling = plan.get("scaling", {})
    if scaling.get("columns"):
        assembler = VectorAssembler(inputCols=scaling["columns"], outputCol="unscaled_features")
        stages.append(assembler)
        
        if scaling.get("strategy") == "StandardScaler":
            scaler = StandardScaler(inputCol="unscaled_features", outputCol="scaled_features", withStd=True, withMean=True)
            stages.append(scaler)
            
    # Fit the pipeline
    if stages:
        pipeline = Pipeline(stages=stages)
        pipeline_model = pipeline.fit(df)
        df_transformed = pipeline_model.transform(df)
        return df_transformed
        
    return df
