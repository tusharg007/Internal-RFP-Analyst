from types import SimpleNamespace

import pytest

from tools import public_demo_app


def test_public_demo_rejects_non_public_corpus_before_app_run(monkeypatch, tmp_path):
    monkeypatch.setattr(public_demo_app, 'load_indexed_snapshot', lambda _: SimpleNamespace(fingerprint='private'))
    with pytest.raises(ValueError, match='private/changed corpora refused'):
        public_demo_app.validate_public_index(tmp_path)


def test_public_demo_accepts_only_exact_approved_digest(monkeypatch, tmp_path):
    snapshot = SimpleNamespace(fingerprint=public_demo_app.PUBLIC_HASH)
    monkeypatch.setattr(public_demo_app, 'load_indexed_snapshot', lambda _: snapshot)
    assert public_demo_app.validate_public_index(tmp_path) is snapshot


def test_public_demo_requires_explicit_index(monkeypatch):
    monkeypatch.delenv('RFP_PUBLIC_DEMO_INDEX', raising=False)
    with pytest.raises(ValueError, match='Set RFP_PUBLIC_DEMO_INDEX'):
        public_demo_app.main()
