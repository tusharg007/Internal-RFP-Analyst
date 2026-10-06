"""Explicit retrieval plans: a finite ontology, never an executable query."""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import Field, model_validator

from rfp_analyst.graph.extraction import (
    FRAMEWORKS,
    TECHNOLOGIES,
    StrictModel,
    alias_spans,
    normalized,
)

RetrievalMode = Literal["vector_only", "graph_only", "hybrid"]
RetrievalPolicy = Literal["auto", "vector_only", "graph_only", "hybrid"]
QueryType = Literal[
    "unsupported",
    "project_constraints",
    "shared_technologies",
    "requirement_candidates",
    "project_fields",
]
ShortName = Annotated[str, Field(min_length=1, max_length=256)]

INDUSTRIES = {
    "healthcare": ("healthcare", "healthcare & life sciences"),
    "banking": ("banking & financial services", "banking"),
    "retail": ("retail & consumer goods", "retail"),
    "insurance": ("insurance", "insurance & risk management"),
    "telecom": ("telecommunications", "telecom"),
    "government": ("government & public sector", "government"),
    "pharma": ("pharmaceutical & life sciences", "pharma", "pharmaceuticals"),
    "energy": ("energy & utilities", "energy"),
    "fintech": ("financial services & fintech", "fintech"),
    "manufacturing": (
        "manufacturing & industrial",
        "manufacturing & industry 4.0",
        "manufacturing",
    ),
}


class GraphPlan(StrictModel):
    query_type: QueryType = "unsupported"
    technology_groups: Annotated[tuple[tuple[ShortName, ...], ...], Field(max_length=5)] = ()
    frameworks: Annotated[tuple[ShortName, ...], Field(max_length=5)] = ()
    industries: Annotated[tuple[ShortName, ...], Field(max_length=5)] = ()
    project_refs: Annotated[tuple[ShortName, ...], Field(max_length=2)] = ()
    fields: Annotated[
        tuple[Literal["timeline", "budget", "outcome"], ...], Field(max_length=3)
    ] = ()
    has_framework: bool = False
    delivered_keyword: Annotated[str, Field(max_length=80)] = ""

    @model_validator(mode="after")
    def controlled(self):
        for group in self.technology_groups:
            if not 1 <= len(group) <= 32 or any(name not in TECHNOLOGIES for name in group):
                raise ValueError("Unknown or oversized technology seed group")
        if any(name not in FRAMEWORKS for name in self.frameworks):
            raise ValueError("Unknown framework seed")
        if self.query_type == "shared_technologies" and len(self.project_refs) != 2:
            raise ValueError("Shared technology requires exactly two project references")
        if self.query_type == "project_constraints" and not (
            self.technology_groups
            or self.frameworks
            or self.industries
            or self.has_framework
            or self.delivered_keyword
        ):
            raise ValueError("A graph intersection needs explicit constraints")
        if self.query_type == "project_fields" and not (self.project_refs and self.fields):
            raise ValueError("Field lookup requires a project reference and typed field")
        return self


class RetrievalDecision(StrictModel):
    mode: RetrievalMode
    plan: GraphPlan = GraphPlan()
    reason: Literal[
        "policy_override",
        "narrative_evidence",
        "connected_constraints",
        "exact_lookup",
        "unsupported_ontology",
    ]


def technology_seed_group(name):
    """Curated provider expansion for candidates, never an entity merge."""
    if name == "Microsoft Azure":
        return tuple(n for n in TECHNOLOGIES if n == name or n.startswith("Azure "))
    return (name,)


def matches_project_reference(reference: str, name: str) -> bool:
    """Ordered title words support shortened descriptions; callers require uniqueness."""
    value = " ".join(re.sub(r"\b(?:the|projects?)\b", " ", normalized(reference)).split())
    title = normalized(name.replace("_", " "))
    if value and value in title:
        return True
    words = value.split()
    return len(words) >= 2 and bool(
        re.search(r"\b" + r"\b.*?\b".join(map(re.escape, words)) + r"\b", title)
    )


def _technology_groups(query):
    groups = []
    for _, _, name in alias_spans(query, TECHNOLOGIES):
        group = technology_seed_group(name)
        if group not in groups:
            groups.append(group)
    return tuple(groups)


def plan_retrieval(query: str, *, policy: RetrievalPolicy = "auto") -> RetrievalDecision:
    if policy not in {"auto", "vector_only", "graph_only", "hybrid"}:
        raise ValueError("Unsupported retrieval policy")
    q = normalized(query)
    narrative = bool(
        re.search(r"\bsummariz|\bpassages?\b|\bwhat (?:does|did)\b.*\bsay|\bresume\b", q)
    )
    groups = _technology_groups(query)
    frameworks = tuple(dict.fromkeys(name for _, _, name in alias_spans(query, FRAMEWORKS)))
    industries = tuple(
        dict.fromkeys(
            alias
            for key, aliases in INDUSTRIES.items()
            if re.search(r"\b" + re.escape(key) + r"\b", q)
            for alias in aliases
        )
    )
    has_framework = bool(re.search(r"\bcompliance\b|\bregulatory\b", q))
    shared = bool(
        re.search(r"\bshared?\b|\bin common\b", q) and re.search(r"\btechnolog|\bstacks?\b", q)
    )
    refs = ()
    if shared:
        match = re.search(r"(?:between|by)\s+(.+?)\s+and\s+(.+?)[?.!]*$", query, re.I)
        if match:
            refs = tuple(part.strip(" .?!\"'") for part in match.groups())
        else:
            quoted = re.findall(r'["\']([^"\']+)["\']', query)
            refs = tuple(quoted[:2])
    fields = tuple(field for field in ("timeline", "budget", "outcome") if field in q)
    if re.search(r"\bachieved\b", q) and "outcome" not in fields:
        fields = (*fields, "outcome")
    field_ref = re.search(r"(?:timeline|budget|outcomes?)\s+(?:for|of)\s+(.+?)[?.!]*$", query, re.I)
    comparison_refs = ()
    if fields and re.search(
        r"\bcompare\s+(?:the\s+)?(?:timelines?|budgets?|budget ranges?|outcomes?)\b", q
    ):
        comparison = re.search(
            r"\b(?:of|for|between)\s+(.+?)\s+(?:and|versus|vs\.?|with)\s+(.+?)[?.!]*$",
            query,
            re.I,
        )
        if comparison:
            comparison_refs = tuple(part.strip(" .?!\"'") for part in comparison.groups())
    delivered = ""
    if re.search(r"\b(?:projects?|case studies)\s+(?:that\s+)?delivered\b", q):
        match = re.search(r"delivered\s+(.+?)(?:\s+for\s+|\s+in\s+|[?.!]|$)", q)
        delivered = match.group(1) if match else ""
    too_long = any(not ref or len(ref) > 256 for ref in (*refs, *comparison_refs))
    too_long = too_long or bool(field_ref and len(field_ref.group(1).strip()) > 256)
    if (
        len(groups) > 5
        or len(frameworks) > 5
        or len(industries) > 5
        or len(delivered) > 80
        or too_long
    ):
        return RetrievalDecision(
            mode=policy if policy != "auto" else "vector_only",
            plan=GraphPlan(),
            reason="unsupported_ontology",
        )
    if shared and len(refs) == 2:
        plan = GraphPlan(query_type="shared_technologies", project_refs=refs)
        mode, reason = "hybrid", "connected_constraints"
    elif comparison_refs and not narrative:
        plan = GraphPlan(query_type="project_fields", project_refs=comparison_refs, fields=fields)
        mode, reason = "hybrid", "exact_lookup"
    elif (
        re.search(r"\b(?:target|uploaded) rfp\b", q)
        and re.search(r"\bmatch|\bcase studies|\bcapabilit|\brequirement", q)
        and not narrative
    ):
        plan = GraphPlan(
            query_type="requirement_candidates",
            technology_groups=groups,
            frameworks=frameworks,
            industries=industries,
            has_framework=has_framework,
            delivered_keyword=delivered,
            fields=fields,
        )
        mode, reason = "hybrid", "connected_constraints"
    elif field_ref and fields and not narrative:
        plan = GraphPlan(
            query_type="project_fields",
            project_refs=(field_ref.group(1).strip(" .?!\"'"),),
            fields=fields,
        )
        mode, reason = "graph_only", "exact_lookup"
    elif (
        (groups or frameworks or industries or has_framework or delivered)
        and re.search(r"\bprojects?\b|\bcase studies\b|\bmust\b|\bshall\b|\bshould\b", q)
        and not narrative
    ):
        plan = GraphPlan(
            query_type="project_constraints",
            technology_groups=groups,
            frameworks=frameworks,
            industries=industries,
            has_framework=has_framework,
            delivered_keyword=delivered,
            fields=fields,
        )
        constraints = (
            len(groups) + len(frameworks) + bool(industries) + bool(has_framework) + bool(delivered)
        )
        mode, reason = (
            ("hybrid", "connected_constraints")
            if constraints > 1
            else ("graph_only", "exact_lookup")
        )
    else:
        plan = GraphPlan()
        mode, reason = "vector_only", "narrative_evidence" if narrative else "unsupported_ontology"
    if policy != "auto":
        mode, reason = policy, "policy_override"
    return RetrievalDecision(mode=mode, plan=plan, reason=reason)
