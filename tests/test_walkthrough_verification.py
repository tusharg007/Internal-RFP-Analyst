import copy

import pytest

from evals.matched_environment import CORPUS_ID, PUBLIC_HASH
from tools.verify_walkthrough import public_verification


def reports():
    manifest = {'indexed_digest': PUBLIC_HASH, 'graph_corpus_id': CORPUS_ID,
                'document_count': 11, 'chunk_count': 54, 'eval_target_rfp_present': True,
                'graph_version': 'version', 'documents': [{'source_file': 'public.pdf'}],
                'index': 'private local path - omit from receipt'}
    protected = {'node_sha256': 'n', 'relationship_sha256': 'r'}
    comparison = {'parity': True, 'failures': [], 'expected_document_sha256': 'd',
                  'actual_document_sha256': 'd', 'expected_chunk_sha256': 'c',
                  'actual_chunk_sha256': 'c'}
    report = {'manifest': manifest, 'comparison': comparison, 'internal_rfp_before': protected,
              'internal_rfp_after_publication': copy.deepcopy(protected),
              'internal_rfp_untouched': True, 'frozen_index_unchanged_after_publication': True,
              'publication': 'reused_without_writes'}
    return report, copy.deepcopy(report)


def test_public_verification_receipt_is_path_free_and_does_not_mutate_inputs():
    before, after = reports()
    original = copy.deepcopy(before)
    result = public_verification(before, after)
    assert result['parity_before'] and result['parity_after'] and result['internal_rfp_unchanged']
    assert 'private local path' not in str(result)
    assert before == original


@pytest.mark.parametrize('mismatch', ['protected', 'manifest', 'parity', 'record_hash', 'publication'])
def test_verification_refuses_any_integrity_mismatch(mismatch):
    before, after = reports()
    if mismatch == 'protected':
        after['internal_rfp_after_publication']['node_sha256'] = 'changed'
    elif mismatch == 'manifest':
        after['manifest']['graph_version'] = 'changed'
    elif mismatch == 'parity':
        after['comparison']['parity'] = False
    elif mismatch == 'record_hash':
        after['comparison']['actual_chunk_sha256'] = 'changed'
    else:
        after['publication'] = 'published'
    with pytest.raises(ValueError):
        public_verification(before, after)
