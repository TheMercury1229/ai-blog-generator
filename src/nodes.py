from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List
from langchain_core.messages import SystemMessage, HumanMessage
from src.models import llm, llm2
from src.prompts import ROUTER_SYSTEM, RESEARCH_SYSTEM, ORCH_SYSTEM, WORKER_SYSTEM
from src.tools import tavily_search_tool
from src.states import BlogState, RouterDecision, EvidenceItem, EvidencePack, Plan, Task


def router_node(state: BlogState) -> dict:
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)

    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}")
        ]
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries
    }


def research_node(state: BlogState) -> dict:
    queries = state["queries"]
    max_results_per_query = 5

    raw_results: List[dict] = []

    for q in queries:
        raw_results.extend(tavily_search_tool(
            q, max_results=max_results_per_query))

    if not raw_results:
        return {"evidence": []}

    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f"Raw search results:\n\n{raw_results}")
        ]
    )
    # Deduplicate by URL
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}


def orchestrator(state: BlogState) -> dict:
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(content=(
                f"Topic: {state['topic']}\n"
                f"Mode: {mode}\n"
                f"Evidence (ONLY use for fresh claims:may be empty):\n\n"
                f"{[e.model_dump() for e in evidence] [:16]}"
            ))
        ]
    )

    return {"plan": plan}


def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )
    section_md = llm2.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).text.strip()

    return {"sections": [(task.id, section_md)]}
# -----------------------------
# 8) Reducer (merge + save)
# -----------------------------


def sanitize_filename(title: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    name = re.sub(r'[\\/:*?"<>|]', '', title)
    name = name.strip().lower().replace(" ", "_")
    return output_dir / (name + ".md")


def merge_content(state: BlogState) -> dict:
    plan = state["plan"]
    assert plan is not None

    ordered_sections = [md for _, md in sorted(
        state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    filename = sanitize_filename(plan.blog_title)
    filename.write_text(final_md, encoding="utf-8")

    return {"merged_md": final_md, "final": final_md}

