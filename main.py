import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

# Import the modular agent nodes and State
from agents import (
    coordinator_node,
    data_engineer_node,
    feature_engineer_node,
    validator_node,
    AgentState
)

# Load Environment Variables (e.g. OPENAI_API_KEY)
load_dotenv()

# --- 1. Routing Function ---
def route_next(state: AgentState) -> str:
    # Dynamically route based on Coordinator's output decision
    if state["next_node"] == "end":
        return END
    return state["next_node"]

# --- 2. Build the LangGraph Workflow ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("coordinator", coordinator_node)
workflow.add_node("feature_engineering", feature_engineer_node)
workflow.add_node("data_engineer", data_engineer_node)
workflow.add_node("validator", validator_node)

# Coordinator is the entry point
workflow.add_edge(START, "coordinator")

# The coordinator decides where to go based on AgentState
workflow.add_conditional_edges("coordinator", route_next, {
    "feature_engineering": "feature_engineering",
    "data_engineer": "data_engineer",
    "validator": "validator",
    END: END
})

# After any specific agent makes a decision, return to coordinator to navigate next steps
workflow.add_edge("feature_engineering", "coordinator")
workflow.add_edge("data_engineer", "coordinator")
workflow.add_edge("validator", "coordinator")

app = workflow.compile()

# --- 3. Example PySpark Pipeline Orchestration ---
if __name__ == "__main__":
    
    # [SCENARIO 1] Initial Training (e.g., January)
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
    if final_state.get("feature_engineering_plan"):
        print(final_state["feature_engineering_plan"].model_dump())
    else:
        print("No Plan Generated")
    
    
    # [SCENARIO 2] March Monitoring (Testing Feb inference data against Jan)
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
            "R2": 0.45,  # Too low -> Retrain
            "MAE": 110.2
        } 
    }
    
    final_monitoring_state = app.invoke(monitoring_state)
    print("\nFinal Pipeline State For Feb:")
    if final_monitoring_state.get("drift_report"):
        print("Drift Report: ", final_monitoring_state["drift_report"].model_dump())
    
    if final_monitoring_state.get("validation_report"):
        print("Validation Report: ", final_monitoring_state["validation_report"].model_dump())
        print("Retrain decision given to PySpark: ", final_monitoring_state["validation_report"].requires_retraining)
