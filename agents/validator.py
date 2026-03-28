import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import AgentState, ValidationReport

def read_skill(skill_name: str) -> str:
    path = os.path.join(".agents", "skills", skill_name, "SKILL.md")
    with open(path, "r") as f:
        return f.read()

def validator_node(state: AgentState):
    print("--- VALIDATOR AGENT ---")
    sys_prompt = read_skill("validator")
    prompt = f"Evaluation Metrics on new data:\n{json.dumps(state['evaluation_metrics'], indent=2)}"
    
    if os.getenv("OPENAI_API_KEY"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        structured_llm = llm.with_structured_output(ValidationReport)
        validation = structured_llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=prompt)])
    else:
        # Mock return
        validation = ValidationReport(
            model_health_status="Degraded",
            requires_retraining=True,
            evaluation_summary="R2 is below 0.50 threshold, retraining is necessary."
        )
        
    print(f"Validation Report: {validation.model_dump()}")
    return {"validation_report": validation}
