import copy

from rfp_analyst.ui.explainability import query_explanation, trace_card_html


def test_trace_html_escapes_query_source_and_summary():
    card = trace_card_html('<img src=x onerror=alert(1)>', '<script>x</script>', 'a & b')
    assert '<img' not in card and '<script>' not in card
    assert '&lt;script&gt;' in card and 'a &amp; b' in card


def test_explanation_preserves_recorded_provenance_without_claim_links():
    payload = {
        'requested_retrieval_mode': 'hybrid', 'retrieval_mode': 'vector_only',
        'graph_fallback_reason': 'unavailable', 'grounded': True,
        'generation_kind': 'llm_kb', 'generation_prompt_hash': 'hash',
        'generation_evidence': [{'source': 'public.pdf', 'page': 0, 'chunk_id': 'c1', 'content': 'private content'}],
        'graph_paths': [{'path_id': 'p1', 'evidence_ids': ['e1'], 'secret': 'password'}],
        'traces': [{'tool': 'final_grounding_verifier', 'verification_status': 'grounded'}],
        'llm': object(), 'prompt': 'private prompt', 'api_key': 'password',
    }
    original = copy.copy(payload)
    explanation = query_explanation(payload)
    assert explanation['requested_mode'] == 'hybrid'
    assert explanation['effective_mode'] == 'vector_only'
    assert explanation['graph_fallback_reason'] == 'unavailable'
    assert explanation['generation_evidence'] == [{'source': 'public.pdf', 'page': 0, 'chunk_id': 'c1'}]
    assert explanation['grounding_checks'][0]['status'] == 'grounded'
    assert explanation['claim_to_chunk_links'] == 'not_recorded'
    assert 'password' not in str(explanation) and 'private' not in str(explanation)
    assert payload == original


def test_unknown_execution_is_not_reported_as_grounded():
    explanation = query_explanation({})
    assert explanation['grounded'] is None
    assert explanation['generation_kind'] == 'not_generated'
    assert explanation['requested_mode'] == 'unknown'
    assert explanation['graph_paths'] == []


def test_explanation_bounds_graph_paths_and_generation_evidence():
    explanation = query_explanation({'graph_paths': [{'path_id': str(i)} for i in range(30)],
                                     'generation_evidence': [{'chunk_id': str(i)} for i in range(30)]})
    assert explanation['graph_path_count'] == 30
    assert len(explanation['graph_paths']) == 20
    assert len(explanation['generation_evidence']) == 20


def test_explanation_exposes_only_recorded_relationship_and_entity_metadata():
    relation = {'subject_id': 'p1', 'predicate': 'uses_technology', 'object_id': 't1',
                'modality': 'proposed', 'evidence_id': 'c1', 'secret': 'do not export'}
    result = query_explanation({'graph_paths': [{'relationships': [relation] * 40}],
                                'graph_entities': [{'entity_id': 't1', 'kind': 'Technology',
                                                    'content': 'do not export'}]})
    assert len(result['graph_paths'][0]['relationships']) == 30
    assert result['graph_paths'][0]['relationships'][0]['modality'] == 'proposed'
    assert result['graph_entities'] == [{'entity_id': 't1', 'kind': 'Technology'}]
    assert 'do not export' not in str(result)
