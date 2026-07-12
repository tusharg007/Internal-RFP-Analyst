"""Proposal drafting helpers."""

from __future__ import annotations


def generate_proposal_outline(user_query: str, case_studies: dict, requirements: list[dict] | None = None) -> dict:
    """Create a grounded proposal outline from retrieved evidence."""
    requirements = requirements or []
    matches = case_studies.get("matches", [])

    requirement_lines = "\n".join(
        f"- {requirement['id']}: {requirement['text']}"
        + (
            f" [Source: {requirement['source_file']}, Page {int(requirement['page']) + 1}]"
            if requirement.get("source_file") and requirement.get("page") is not None
            else ""
        )
        for requirement in requirements
    ) or "- No structured requirements were extracted."

    gap_lines = []
    for requirement in requirements:
        for gap in requirement.get("inferred_gaps", [])[:3]:
            gap_lines.append(
                f"- {requirement['id']}: {gap['detail']}"
            )
    gap_block = "\n".join(gap_lines[:8]) or "- No inferred gaps were identified from the available target evidence."

    evidence_lines = "\n".join(
        (
            f"- {match['source']}: fit score {match.get('fit_score', 0)}/5; "
            f"matched {', '.join(match.get('matched_requirements', [])) or 'no explicit requirements'}; "
            f"missing {', '.join(match.get('missing_coverage', [])) or 'no scored coverage gaps'} "
            f"[Source: {match['source']}, Page {match['pages'][0] + 1}]"
        )
        for match in matches[:3]
    ) or "- No matching case studies found."

    outline = f"""## Executive Summary
- Address the request with a proposal grounded in the uploaded target requirements and the strongest cited case-study evidence.
- Treat unprovided target details as assumptions to confirm before pricing or delivery commitment.

## Requirement Understanding and Gaps
{requirement_lines}
{gap_block}

## Proposed Architecture and Security
- Map Azure, integration, and control decisions directly to the confirmed requirements before final architecture selection.
- Confirm absent or ambiguous security/compliance controls before representing them as delivery facts.

## Dashboard and Analytics Workstream
- Define dashboard users, KPI definitions, refresh cadence, and acceptance criteria before build planning.
- Use cited analytics case-study patterns only where the supporting evidence matches the target requirement.

## Phased Delivery and Risk Management
- Convert confirmed requirements into discovery, design, implementation, validation, and rollout milestones.
- Track missing timelines, dependencies, SLAs, staffing, and risk ownership as proposal assumptions.

## Case Studies, Outcomes and Success Metrics
{evidence_lines}
"""
    return {"outline": outline, "requirements": requirements, "matches": matches}
