import pandas as pd
import numpy as np
import joblib
import os

# --- 1. Load Dependencies & 2. Load Trained Model ---
# This assumes setup_inference_demo.py has been run to generate best_model.pkl
if not os.path.exists("best_model.pkl"):
    print("Error: best_model.pkl not found. Run setup_inference_demo.py first.")
else:
    model = joblib.load("best_model.pkl")

# --- 3. Load Unseen Data (March 2016) ---
# Mocked from February taxi data using setup_inference_demo.py
if not os.path.exists("march_2016_with_lags.csv"):
    print("Error: march_2016_with_lags.csv not found. Run setup_inference_demo.py first.")
else:
    df = pd.read_csv("march_2016_with_lags.csv")

# --- 4. Define Feature Columns ---
FEATURES = [
    "hour",
    "day_of_week",
    "pickup_cluster",
    "is_peak_hour",
    "lag_1",
    "lag_24"
]

# --- 5. Generate Predictions ---
if 'model' in locals() and 'df' in locals():
    df["predicted_demand"] = model.predict(df[FEATURES])

# --- 6. Aggregate Predictions per Cluster (Next Hour) ---
    cluster_predictions = (
        df.groupby("pickup_cluster")["predicted_demand"]
        .mean()
        .reset_index()
    )

# --- 7. Rank Clusters (Top K Recommendation) ---
    TOP_K = 3
    top_k_clusters = cluster_predictions.sort_values(
        by="predicted_demand", ascending=False
    ).head(TOP_K)

# --- 8. Output Results ---
    print("🚕 Top Recommended Clusters:\n")
    print(top_k_clusters)
    print("\n" + "="*30 + "\n")

# --- 9. Wrap into Function (FOR PRESENTATION) ---
def recommend_clusters(df, model, top_k=3):
    FEATURES = [
        "hour",
        "day_of_week",
        "pickup_cluster",
        "is_peak_hour",
        "lag_1",
        "lag_24"
    ]
    
    # Predict demand
    df_copy = df.copy()
    df_copy["predicted_demand"] = model.predict(df_copy[FEATURES])
    
    # Aggregate by cluster
    cluster_preds = (
        df_copy.groupby("pickup_cluster")["predicted_demand"]
        .mean()
        .reset_index()
    )
    
    # Rank clusters
    top_clusters = cluster_preds.sort_values(
        by="predicted_demand", ascending=False
    ).head(top_k)
    
    return top_clusters

# --- 10. (A+ BONUS) Simulate Real-Time Input ---
def predict_single_input(model, input_dict):
    FEATURES = [
        "hour",
        "day_of_week",
        "pickup_cluster",
        "is_peak_hour",
        "lag_1",
        "lag_24"
    ]
    
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df[FEATURES])[0]
    return prediction

# --- Run Demo ---
if 'model' in locals() and 'df' in locals():
    print("Running Function Demo:")
    top_clusters = recommend_clusters(df, model)
    print(top_clusters)
    print("\nReal-Time Input Demo:")
    sample_input = {
        "hour": 10,
        "day_of_week": 2,
        "pickup_cluster": 3,
        "is_peak_hour": 1,
        "lag_1": 120,
        "lag_24": 95
    }
    print("Predicted Demand for Sample Input:", predict_single_input(model, sample_input))
