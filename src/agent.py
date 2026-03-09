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
    report: dict


def analyze_demand(state):
    s = state["stats"]
    ph = state["peak_hours"]
    pd = state["peak_days"]

    hours_str = ", ".join([f"{h}:00 ({v:.1f} kWh)" for h, v in ph.items()])
    days_str = ", ".join([f"{d} ({v:.1f} kWh)" for d, v in pd.items()])

    prompt = f"""Analyze this EV charging data and give a short summary (4-5 lines):
- Total records: {s['total_records']}
- Date range: {s['date_range_start']} to {s['date_range_end']}
- Avg volume: {s['avg_volume']:.2f} kWh
- Max volume: {s['max_volume']:.2f} kWh
- Peak hours: {hours_str}
- Peak days: {days_str}

Focus on demand patterns and trends."""

    state["demand_summary"] = ask_llm(prompt)
    return state


def retrieve_guidelines(state):
    prompt = """List 5-6 key EV charging infrastructure planning guidelines covering:
- Station placement
- Capacity planning  
- Load management
- Grid integration
Keep each point brief. Include references if possible."""

    state["guidelines"] = ask_llm(prompt)
    return state


def generate_plan(state):
    prompt = f"""You are an EV infrastructure planner.

Demand analysis:
{state['demand_summary']}

Planning guidelines:
{state['guidelines']}

Give specific recommendations for:
1. Where to add new charging stations
2. How to handle peak load
3. Capacity expansion
4. Grid integration
Reference the data in your answer."""

    state["recommendations"] = ask_llm(prompt)
    return state


def format_report(state):
    # get scheduling tips
    sched_prompt = f"""Based on peak hours {state['peak_hours']} and peak days {state['peak_days']},
give 3-4 scheduling and load-balancing strategies. Be brief."""

    scheduling = ask_llm(sched_prompt)

    state["report"] = {
        "demand_summary": state["demand_summary"],
        "high_load_analysis": f"Peak Hours: {state['peak_hours']}\nPeak Days: {state['peak_days']}",
        "infrastructure_recommendations": state["recommendations"],
        "scheduling_insights": scheduling,
        "references": state["guidelines"]
    }
    return state


def create_agent():
    g = StateGraph(AgentState)
    g.add_node("analyze_demand", analyze_demand)
    g.add_node("retrieve_guidelines", retrieve_guidelines)
    g.add_node("generate_plan", generate_plan)
    g.add_node("format_report", format_report)

    g.add_edge(START, "analyze_demand")
    g.add_edge("analyze_demand", "retrieve_guidelines")
    g.add_edge("retrieve_guidelines", "generate_plan")
    g.add_edge("generate_plan", "format_report")
    g.add_edge("format_report", END)

    return g.compile()


def run_agent(predictions, peak_hours, peak_days, stats):
    agent = create_agent()
    init = {
        "predictions": predictions,
        "peak_hours": dict(peak_hours),
        "peak_days": dict(peak_days),
        "stats": stats,
        "demand_summary": "",
        "guidelines": "",
        "recommendations": "",
        "report": {}
    }
    result = agent.invoke(init)
    return result["report"]
