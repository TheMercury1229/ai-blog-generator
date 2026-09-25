from __future__ import annotations
from langgraph.graph import StateGraph, START, END
from src.nodes import router_node, research_node, orchestrator, worker_node, merge_content
from src.edges import route_next, fanout
from src.states import BlogState

from dotenv import load_dotenv
load_dotenv()


reducer_graph = StateGraph(BlogState)
reducer_graph.add_node("merge_content", merge_content)

reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", END)
reducer_subgraph = reducer_graph.compile()
g = StateGraph(BlogState)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {
                        "research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
