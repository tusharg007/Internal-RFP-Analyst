from chromadb import PersistentClient
from chromadb.config import Settings

from rfp_analyst.retrieval.hybrid import load_indexed_snapshot
from rfp_analyst.retrieval.vector_store import VectorStoreManager


def test_graph_snapshot_then_vector_client_uses_same_chroma_settings(tmp_path):
    client = PersistentClient(path=str(tmp_path), settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection('rfp_kb_v2', embedding_function=None)
    collection.add(ids=['public-chunk'], embeddings=[[0.1, 0.2]], documents=['Public Azure example'],
                   metadatas=[{'chunk_id': 'public-chunk', 'source_file': 'public.pdf',
                               'document_origin': 'sample', 'file_hash': 'a' * 64,
                               'page': 0, 'chunk_index': 0}])
    snapshot = load_indexed_snapshot(tmp_path, 'rfp_kb_v2')
    vector = VectorStoreManager(tmp_path, embedding_function=object()).load(create_if_missing=False)
    assert vector._collection.count() == 1
    assert load_indexed_snapshot(tmp_path, 'rfp_kb_v2').fingerprint == snapshot.fingerprint
