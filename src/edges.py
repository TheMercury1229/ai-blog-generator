from src.states import BlogState
from langgraph.types import Send


def route_next(state: BlogState) -> str:
    return "research" if state['needs_research'] else "orchestrator"


def fanout(state: BlogState):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])]
            }
        )
        for task in state["plan"].tasks
    ]
