"""Public post-hoc probe validation, with synthetic records and no API calls."""

import copy

import pytest

from evals.matched_environment import PUBLIC_HASH
from evals.ragas_adapter import fingerprint
from tools.judge_public_smoke import select_public_capture


def fixture():
    cases = [{'id': 'semantic_modernization', 'question': 'Synthetic question',
              'reference': 'Synthetic independent reference'}]
    lock = {'signature': {'corpus_hash': PUBLIC_HASH, 'question_set_hash': fingerprint(cases)}}
    row = {'id': cases[0]['id'], 'status': 'completed', 'generation_kind': 'llm_kb',
           'sample': {'user_input': cases[0]['question'], 'response': 'Synthetic output',
                      'reference': cases[0]['reference'], 'retrieved_contexts': ['Original context']},
           'provenance': {'generation_prompt_hash': 'original'}, 'deterministic': {}}
    capture = {'freeze': lock, 'reports': [{'retrieval_mode': 'vector_only',
                                          'corpus_hash': PUBLIC_HASH, 'cases': [row]}]}
    return capture, cases, lock


def test_public_judge_probe_preserves_exact_saved_context_and_answer():
    capture, cases, lock = fixture()
    original = copy.deepcopy(capture)
    _, row, execution = select_public_capture(capture, cases, lock)
    assert execution.sample.retrieved_contexts == row['sample']['retrieved_contexts']
    assert execution.sample.response == row['sample']['response']
    assert capture == original


@pytest.mark.parametrize('change', ['corpus', 'contexts', 'question', 'reference', 'prompt_hash'])
def test_public_judge_probe_refuses_mismatch_or_missing_boundary_capture(change):
    capture, cases, lock = fixture()
    report = capture['reports'][0]
    row = report['cases'][0]
    if change == 'corpus':
        report['corpus_hash'] = 'not-public'
    elif change == 'contexts':
        row['sample']['retrieved_contexts'] = []
    elif change == 'question':
        row['sample']['user_input'] = 'altered'
    elif change == 'reference':
        row['sample']['reference'] = 'altered'
    else:
        row['provenance']['generation_prompt_hash'] = ''
    with pytest.raises(ValueError):
        select_public_capture(capture, cases, lock)
