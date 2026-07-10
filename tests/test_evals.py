from evals.metrics import build_metrics_summary


def test_build_metrics_summary():
    results = [
        {
            "expected_sources": ["A.pdf"],
            "retrieved_sources": ["A.pdf"],
            "expect_citations": True,
            "answer": "Fact [Source: A.pdf, Page 1]",
            "grounded": True,
            "latency_ms": 10,
            "tool_call_count": 2,
            "passed": True,
        },
        {
            "expected_sources": ["B.pdf"],
            "retrieved_sources": [],
            "expect_citations": False,
            "answer": "No answer",
            "grounded": False,
            "latency_ms": 30,
            "tool_call_count": 1,
            "passed": False,
        },
    ]

    metrics = build_metrics_summary(results)

    assert metrics["retrieval_hit_rate"] == 0.5
    assert metrics["citation_coverage"] == 1.0
    assert metrics["grounded_answer_score"] == 0.5
    assert metrics["average_latency"] == 20.0
    assert metrics["average_latency_ms"] == 20.0
    assert metrics["tool_call_count"] == 1.5
    assert metrics["failure_rate"] == 0.5
