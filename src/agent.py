from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from src.llm import ask_llm

class AgentState(TypedDict):
    predictions: dict
    peak_hours: dict
    peak_days: dict
    stats: dict
    demand_summary: str
    guidelines: str
    recommendations: str
    review_feedback: str
    is_plan_ok: bool
    report: dict
    error: str

def validate_input(state):
    errs = []
    
    if not state.get("stats"):
        errs.append("missing stats")
    if not state.get("peak_hours"):
        errs.append("missing peak hours") 
    if not state.get("peak_days"):
        errs.append("missing peak days")
        
    s = state.get("stats", {})
    if s.get("total_records", 0) < 10:
        errs.append("not enough data")
    if s.get("avg_volume", 0) == 0:
        errs.append("volume is 0")
        
    if len(errs) > 0:
        state["error"] = ", ".join(errs)
    else:
        state["error"] = ""
        
    return state

def analyze_demand(state):
    s = state["stats"]
    
    h_str = ""
    for h, v in state["peak_hours"].items():
        h_str += f"{h}:00({v:.1f}), "
        
    d_str = ""
    for d, v in state["peak_days"].items():
        d_str += f"{d}({v:.1f}), "

    p = f"""look at this EV charging data and summarize it in 4-5 lines:
records: {s.get('total_records')}
date range: {s.get('date_range_start')} to {s.get('date_range_end')}
avg vol: {s.get('avg_volume', 0):.2f}
max vol: {s.get('max_volume', 0):.2f}
peak hours: {h_str}
peak days: {d_str}
"""
    if state.get("error"):
        p += f"\nwarning: {state['error']}"

    state["demand_summary"] = ask_llm(p)
    return state

from src.rag import retrieve_best_guidelines

def retrieve_guidelines(state):
    q = "ev charging station placement grid integration load management"
    state["guidelines"] = retrieve_best_guidelines(q)
    return state

def generate_plan(state):
    p = f'''you are an ev planning expert.
    
demand summary: {state["demand_summary"]}

guidelines: {state["guidelines"]}

based on the data, write recommendations for:
1. station placement
2. handling peak loads
3. capacity upgrades
4. grid stuff

make sure claims are supported by the demand data.'''

    state["recommendations"] = ask_llm(p)
    return state

def review_plan(state):
    p = f'''look at this plan: {state["recommendations"]}
    
based on this data: {state["demand_summary"]}

is the plan realistic and backed by data? is anything missing?
start your response with "APPROVED" if it looks good, otherwise "NEEDS REVISION"'''

    res = ask_llm(p)
    state["review_feedback"] = res
    if "APPROVED" in res.upper():
        state["is_plan_ok"] = True
    else:
        state["is_plan_ok"] = False
    return state

def revise_plan(state):
    p = f'''fix this plan based on the feedback:
    
plan: {state["recommendations"]}
feedback: {state["review_feedback"]}

write a better version.'''
    state["recommendations"] = ask_llm(p)
    state["is_plan_ok"] = True
    return state

def check_review(state):
    if state.get("is_plan_ok") == True:
        return "format_report"
    return "revise_plan"

def format_report(state):
    sched = ask_llm(f"based on peak hours {state['peak_hours']} and days {state['peak_days']}, suggest 3-4 scheduling strategies.")

    state["report"] = {
        "demand_summary": state["demand_summary"],
        "high_load_analysis": f"Peak Hours: {state['peak_hours']}\nPeak Days: {state['peak_days']}",
        "infrastructure_recommendations": state["recommendations"],
        "scheduling_insights": sched,
        "references": state["guidelines"],
        "review_status": state.get("review_feedback", ""),
        "data_warnings": state.get("error", "")
    }
    return state

def create_agent():
    g = StateGraph(AgentState)

    g.add_node("validate_input", validate_input)
    g.add_node("analyze_demand", analyze_demand)
    g.add_node("retrieve_guidelines", retrieve_guidelines)
    g.add_node("generate_plan", generate_plan)
    g.add_node("review_plan", review_plan)
    g.add_node("revise_plan", revise_plan)
    g.add_node("format_report", format_report)

    g.add_edge(START, "validate_input")
    g.add_edge("validate_input", "analyze_demand")
    g.add_edge("analyze_demand", "retrieve_guidelines")
    g.add_edge("retrieve_guidelines", "generate_plan")
    g.add_edge("generate_plan", "review_plan")
    g.add_conditional_edges("review_plan", check_review)
    g.add_edge("revise_plan", "format_report")
    g.add_edge("format_report", END)

    return g.compile()

def run_agent(predictions, peak_hours, peak_days, stats):
    agent = create_agent()
    
    # init state
    init = {
        "predictions": predictions,
        "peak_hours": dict(peak_hours) if peak_hours is not None else {},
        "peak_days": dict(peak_days) if peak_days is not None else {},
        "stats": stats if stats else {},
        "demand_summary": "",
        "guidelines": "",
        "recommendations": "",
        "review_feedback": "",
        "is_plan_ok": False,
        "report": {},
        "error": ""
    }
    
    res = agent.invoke(init)
    return res["report"]
