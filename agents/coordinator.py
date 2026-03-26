import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import AgentState, CoordinatorDecision

def read_skill(skill_name: str) -> str:
    path = os.path.join(".agents", "skills", skill_name, "SKILL.md")
    with open(path, "r") as f:
        return f.read()

def coordinator_node(state: AgentState):
    print("--- COORDINATOR AGENT ---")
    sys_prompt = read_skill("coordinator")
    
    prompt = f"Current Scenario: {state['scenario']}\n"
    prompt += f"Has Feature Plan: {state.get('feature_engineering_plan') is not None}\n"
    prompt += f"Has Drift Report: {state.get('drift_report') is not None}\n"
    prompt += f"Has Validation Report: {state.get('validation_report') is not None}\n"
    
    # Check if OPENAI_API_KEY is available (to fall back to dummy mock if running without key)
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = llm.with_structured_output(CoordinatorDecision)
        decision = structured_llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=prompt)
        ])
        next_node = decision.next_node
        reasoning = decision.reasoning
    else:
        # Dummy mock
        next_node = "end"
        if state["scenario"] == "initial_training" and not state.get("feature_engineering_plan"):
            next_node = "feature_engineering"
        elif state["scenario"] == "march_monitoring":
            if not state.get("drift_report"):
                next_node = "data_engineer"
            elif not state.get("validation_report"):
                next_node = "validator"
        reasoning = "Mock reasoning (No API key found)"
            
    print(f"Coordinator decided next node: {next_node} ({reasoning})")
    return {"next_node": next_node}
