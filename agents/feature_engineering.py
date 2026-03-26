import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import AgentState, FeatureEngineeringPlan

def read_skill(skill_name: str) -> str:
    path = os.path.join(".agents", "skills", skill_name, "SKILL.md")
    with open(path, "r") as f:
        return f.read()

def feature_engineer_node(state: AgentState):
    print("--- FEATURE ENGINEERING AGENT ---")
    sys_prompt = read_skill("feature_engineering")
    prompt = f"Dataset Stats & Schema:\n{json.dumps(state['dataset_stats'], indent=2)}"
    
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = llm.with_structured_output(FeatureEngineeringPlan)
        plan = structured_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
    else:
        # Mock payload
        plan = FeatureEngineeringPlan(
            numerical_imputation={"strategy": "mean", "columns": ["age", "income"]},
            categorical_encoding={"strategy": "StringIndexer", "columns": []},
            scaling={"strategy": "StandardScaler", "columns": ["age", "income"]},
            features_to_drop=["id_column"]
        )
        
    print(f"Generated Feature Plan: {plan.model_dump()}")
    return {"feature_engineering_plan": plan}
