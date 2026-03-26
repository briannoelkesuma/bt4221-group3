import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import AgentState, DriftReport

def read_skill(skill_name: str) -> str:
    path = os.path.join(".agents", "skills", skill_name, "SKILL.md")
    with open(path, "r") as f:
        return f.read()

def data_engineer_node(state: AgentState):
    print("--- DATA ENGINEER AGENT (Drift Detection) ---")
    sys_prompt = read_skill("data_engineer")
    prompt = f"Dataset Stats (Compare Jan to Feb):\n{json.dumps(state['dataset_stats'], indent=2)}"
    
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        structured_llm = llm.with_structured_output(DriftReport)
        report = structured_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
    else:
        # Mock return
        report = DriftReport(
            drift_detected=True,
            drifted_features=["income"],
            drift_explanation="Income mean decreased by 25%."
        )
        
    print(f"Drift Report: {report.model_dump()}")
    return {"drift_report": report}
