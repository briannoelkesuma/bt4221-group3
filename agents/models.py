from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field

# --- Pydantic Models for Agent Outputs ---

class FeatureEngineeringPlan(BaseModel):
    numerical_imputation: Dict[str, Any] = Field(description="Imputation strategy for numerical columns")
    categorical_encoding: Dict[str, Any] = Field(description="Encoding strategy for categorical columns")
    scaling: Dict[str, Any] = Field(description="Scaling strategy (e.g., StandardScaler)")
    features_to_drop: List[str] = Field(description="List of columns to drop")

class DriftReport(BaseModel):
    drift_detected: bool = Field(description="True if significant data drift is detected")
    drifted_features: List[str] = Field(description="List of features that drifted")
    drift_explanation: str = Field(description="Reasoning for drift detection")

class ValidationReport(BaseModel):
    model_health_status: str = Field(description="e.g., 'Degraded' or 'Healthy'")
    requires_retraining: bool = Field(description="True if model requires retraining")
    evaluation_summary: str = Field(description="Reasoning based on evaluation metrics")

class CoordinatorDecision(BaseModel):
    next_node: str = Field(description="The next node to route to: 'feature_engineering', 'data_engineer', 'validator', or 'end'")
    reasoning: str = Field(description="Why this node was chosen")

# --- LangGraph State ---

class AgentState(TypedDict):
    scenario: str # 'initial_training' or 'march_monitoring'
    dataset_stats: Dict[str, Any] # e.g. Schema, Mean, StdDev provided by PySpark
    evaluation_metrics: Optional[Dict[str, float]] # e.g. RMSE, R2 provided by PySpark
    
    # Outputs populated by agents making decisions
    feature_engineering_plan: Optional[FeatureEngineeringPlan]
    drift_report: Optional[DriftReport]
    validation_report: Optional[ValidationReport]
    
    # Internal routing determined by Coordinator
    next_node: str 
