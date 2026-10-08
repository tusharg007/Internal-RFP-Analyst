"""Publish a secret-free, machine-checkable before/after corpus-parity receipt."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.matched_environment import CORPUS_ID, PUBLIC_HASH  # noqa: E402
from evals.ragas_reports import write_report  # noqa: E402


def public_verification(before, after):
    manifests = [r['manifest'] for r in (before, after)]
    if any(m['indexed_digest'] != PUBLIC_HASH or m['graph_corpus_id'] != CORPUS_ID
           or m['document_count'] != 11 or m['chunk_count'] != 54
           or not m['eval_target_rfp_present'] for m in manifests):
        raise ValueError('Not the approved public evaluation environment')
    if manifests[0] != manifests[1]:
        raise ValueError('Public corpus manifest changed during walkthrough')
    protected = [r[key] for r in (before, after)
                 for key in ('internal_rfp_before', 'internal_rfp_after_publication')]
    if any(p != protected[0] for p in protected):
        raise ValueError('Protected live corpus changed')
    comparisons = [r['comparison'] for r in (before, after)]
    for r, comparison in zip((before, after), comparisons):
        if (not comparison['parity'] or comparison['failures']
                or not r['internal_rfp_untouched']
                or not r['frozen_index_unchanged_after_publication']
                or r['publication'] != 'reused_without_writes'
                or comparison['expected_document_sha256'] != comparison['actual_document_sha256']
                or comparison['expected_chunk_sha256'] != comparison['actual_chunk_sha256']):
            raise ValueError('Failed corpus parity/immutability verification')
    manifest = manifests[0]
    return {
        'kind': 'public_walkthrough_integrity_receipt', 'parity_before': True,
        'parity_after': True, 'internal_rfp_unchanged': True,
        'graph_publication': 'none:reused_without_writes',
        'document_count': manifest['document_count'], 'chunk_count': manifest['chunk_count'],
        'document_names': sorted(d['source_file'] for d in manifest['documents']),
        'indexed_digest': PUBLIC_HASH, 'graph_corpus_id': CORPUS_ID,
        'graph_version': manifest['graph_version'], 'record_comparisons': comparisons,
        # Hashes only, never private records, service addresses or local paths.
        'protected_namespace_fingerprints': {
            k: protected[0][k] for k in ('node_sha256', 'relationship_sha256')
        },
        'meaning': 'manifest parity does not prove runtime graph retrieval or answer quality',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = public_verification(
        json.loads(args.before.read_text(encoding='utf-8')),
        json.loads(args.after.read_text(encoding='utf-8')),
    )
    write_report(args.output, result)
    print(json.dumps({k: result[k] for k in ('parity_before', 'parity_after', 'internal_rfp_unchanged')}))


if __name__ == '__main__':
    main()
