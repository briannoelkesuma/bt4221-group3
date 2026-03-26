import os
import json
from typing import Dict, Any, List, Optional, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# --- 1. Load Environment & Models ---
load_dotenv()
# Make sure to set OPENAI_API_KEY in your .env file
# llm = ChatOpenAI(model="gpt-4o", temperature=0) # Uncomment when running with API Key!

# --- 2. Define Pydantic Models for Agent Outputs ---

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
    next_node: str = Field(description="The next node to route to: 'feature_engineer', 'data_engineer', 'validator', or 'end'")
    reasoning: str = Field(description="Why this node was chosen")

# --- 3. Define LangGraph State ---

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

# --- 4. Helper Function to Read Skills ---
def read_skill(skill_name: str) -> str:
    path = os.path.join(".agents", "skills", skill_name, "SKILL.md")
    with open(path, "r") as f:
        return f.read()

# --- 5. LangGraph Nodes ---

def coordinator_node(state: AgentState):
    print("--- COORDINATOR AGENT ---")
    sys_prompt = read_skill("coordinator")
    
    prompt = f"Current Scenario: {state['scenario']}\n"
    prompt += f"Has Feature Plan: {state.get('feature_engineering_plan') is not None}\n"
    prompt += f"Has Drift Report: {state.get('drift_report') is not None}\n"
    prompt += f"Has Validation Report: {state.get('validation_report') is not None}\n"
    
    # Uncomment when LLM is active:
    # structured_llm = llm.with_structured_output(CoordinatorDecision)
    # decision = structured_llm.invoke([
    #     SystemMessage(content=sys_prompt),
    #     HumanMessage(content=prompt)
    # ])
    
    # Dummy mock for now assuming LLM is not hooked up yet
    next_node = "end"
    if state["scenario"] == "initial_training" and not state.get("feature_engineering_plan"):
        next_node = "feature_engineer"
    elif state["scenario"] == "march_monitoring":
        if not state.get("drift_report"):
            next_node = "data_engineer"
        elif not state.get("validation_report"):
            next_node = "validator"
            
    print(f"Coordinator decided next node: {next_node}")
    return {"next_node": next_node}

def feature_engineer_node(state: AgentState):
    print("--- FEATURE ENGINEERING AGENT ---")
    sys_prompt = read_skill("feature_engineering")
    prompt = f"Dataset Stats & Schema:\n{json.dumps(state['dataset_stats'], indent=2)}"
    
    # structured_llm = llm.with_structured_output(FeatureEngineeringPlan)
    # plan = structured_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
    
    # Mock return
    plan = FeatureEngineeringPlan(
        numerical_imputation={"strategy": "mean", "columns": ["age", "income"]},
        categorical_encoding={"strategy": "StringIndexer", "columns": []},
        scaling={"strategy": "StandardScaler", "columns": ["age", "income"]},
        features_to_drop=["id_column"]
    )
    return {"feature_engineering_plan": plan}

def data_engineer_node(state: AgentState):
    print("--- DATA ENGINEER AGENT (Drift Detection) ---")
    sys_prompt = read_skill("data_engineer")
    prompt = f"Dataset Stats (Compare Jan to Feb):\n{json.dumps(state['dataset_stats'], indent=2)}"
    
    # structured_llm = llm.with_structured_output(DriftReport)
    # report = structured_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
    
    # Mock return
    report = DriftReport(
        drift_detected=True,
        drifted_features=["income"],
        drift_explanation="Income mean decreased by 25%."
    )
    return {"drift_report": report}

def validator_node(state: AgentState):
    print("--- VALIDATOR AGENT ---")
    sys_prompt = read_skill("validator")
    prompt = f"Evaluation Metrics on new data:\n{json.dumps(state['evaluation_metrics'], indent=2)}"
    
    # structured_llm = llm.with_structured_output(ValidationReport)
    # validation = structured_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
    
    # Mock return
    validation = ValidationReport(
        model_health_status="Degraded",
        requires_retraining=True,
        evaluation_summary="R2 is below 0.50 threshold, retraining is necessary."
    )
    return {"validation_report": validation}

# --- 6. Routing Function ---
def route_next(state: AgentState) -> str:
    # Dynamically route based on Coordinator's output
    if state["next_node"] == "end":
        return END
    return state["next_node"]

# --- 7. Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("coordinator", coordinator_node)
workflow.add_node("feature_engineer", feature_engineer_node)
workflow.add_node("data_engineer", data_engineer_node)
workflow.add_node("validator", validator_node)

# Coordinator is the entry point
workflow.add_edge(START, "coordinator")

# The coordinator decides where to go
workflow.add_conditional_edges("coordinator", route_next, {
    "feature_engineer": "feature_engineer",
    "data_engineer": "data_engineer",
    "validator": "validator",
    END: END
})

# After any specific agent makes a decision, return to coordinator to check what's next
workflow.add_edge("feature_engineer", "coordinator")
workflow.add_edge("data_engineer", "coordinator")
workflow.add_edge("validator", "coordinator")

app = workflow.compile()

# --- 8. Dummy Execution ---
if __name__ == "__main__":
    
    # In reality, you'd use PySpark first to gather these subset stats.
    # We pass the concise stats to the agents instead of the big dataframes.
    
    print("\n========== SCENARIO 1: INITIAL TRAINING (JAN) ==========")
    initial_state = {
        "scenario": "initial_training",
        "dataset_stats": {
            "columns": ["age", "income", "target"],
            "jan_stats": {"age_mean": 35, "income_mean": 60000}
        },
        "evaluation_metrics": None
    }
    
    final_state = app.invoke(initial_state)
    print("\nFinal Pipeline State For Jan:")
    print(final_state["feature_engineering_plan"].model_dump() if final_state.get("feature_engineering_plan") else "No Plan")
    
    print("\n\n========== SCENARIO 2: MARCH MONITORING (FEB DATA) ==========")
    monitoring_state = {
        "scenario": "march_monitoring",
        "dataset_stats": {
            "columns": ["age", "income", "target"],
            "jan_stats": {"age_mean": 35, "income_mean": 60000},
            "feb_stats": {"age_mean": 36, "income_mean": 45000} # Simulated drift
        },
        "evaluation_metrics": {
            "RMSE": 150.5,
            "R2": 0.45,  # Too low
            "MAE": 110.2
        } 
    }
    
    final_monitoring_state = app.invoke(monitoring_state)
    print("\nFinal Pipeline State For Feb:")
    print("Drift Report: ", final_monitoring_state["drift_report"].model_dump() if final_monitoring_state.get("drift_report") else None)
    print("Validation Report: ", final_monitoring_state["validation_report"].model_dump() if final_monitoring_state.get("validation_report") else None)
    print("Retrain decision given to PySpark: ", final_monitoring_state.get("validation_report").requires_retraining if final_monitoring_state.get("validation_report") else False)
