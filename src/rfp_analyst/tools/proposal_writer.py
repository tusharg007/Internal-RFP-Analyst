"""Proposal drafting helpers."""

from __future__ import annotations


def generate_proposal_outline(user_query: str, case_studies: dict, requirements: list[dict] | None = None) -> dict:
    """Create a grounded proposal outline from retrieved evidence."""
    requirements = requirements or []
    matches = case_studies.get("matches", [])

    requirement_lines = "\n".join(
        f"- {requirement['id']}: {requirement['text']}" for requirement in requirements
    ) or "- No structured requirements were extracted."

    evidence_lines = "\n".join(
        f"- {match['source']} [Source: {match['source']}, Page {match['pages'][0] + 1}]"
        for match in matches[:5]
    ) or "- No matching case studies found."

    outline = f"""## Executive Summary
Address the request: {user_query}

## Client Requirements
{requirement_lines}

## Relevant Case Studies
{evidence_lines}

## Proposed Approach
- Reuse proven delivery patterns from the cited projects.
- Map each workstream to the structured client requirements.
- Highlight measurable outcomes supported by prior engagements.

## Delivery Plan
- Discovery and requirements confirmation
- Solution design and implementation
- Validation, rollout, and stakeholder enablement

## Risks and Mitigations
- Call out delivery risks only when supported by cited prior work.
- Add compliance and implementation assumptions explicitly.
"""
    return {"outline": outline, "requirements": requirements, "matches": matches}
