"""Fixed Neo4j 5.26+ provenance schema for the current Python driver.

Only document/chunk anchors are initialized in this phase. Domain entities and
assertions will have separate migrations when extraction is implemented.
Composite uniqueness works without Enterprise-only node-key constraints.
"""

SCHEMA_CONSTRAINTS = {
    "rfp_document_identity": ("RFPDocument", ["corpus_id", "corpus_version", "document_id"]),
    "rfp_chunk_evidence_identity": ("RFPChunk", ["corpus_id", "corpus_version", "evidence_id"]),
    "rfp_chunk_legacy_identity": ("RFPChunk", ["corpus_id", "corpus_version", "chunk_id"]),
}
SCHEMA_INDEXES = {
    "rfp_document_origin": ("RFPDocument", ["corpus_id", "corpus_version", "document_origin"]),
    "rfp_document_file_hash": ("RFPDocument", ["corpus_id", "corpus_version", "file_hash"]),
    "rfp_chunk_document_page": (
        "RFPChunk",
        ["corpus_id", "corpus_version", "document_id", "page_index"],
    ),
}

CHECK_CONSTRAINTS = """
SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties
WHERE name IN $names
RETURN name, type, entityType, labelsOrTypes, properties
"""
CHECK_INDEXES = """
SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties
WHERE name IN $names
RETURN name, type, entityType, labelsOrTypes, properties
"""

SCHEMA_STATEMENTS = (
    "CREATE CONSTRAINT rfp_document_identity IF NOT EXISTS "
    "FOR (d:RFPDocument) REQUIRE (d.corpus_id, d.corpus_version, d.document_id) IS UNIQUE",
    "CREATE CONSTRAINT rfp_chunk_evidence_identity IF NOT EXISTS "
    "FOR (c:RFPChunk) REQUIRE (c.corpus_id, c.corpus_version, c.evidence_id) IS UNIQUE",
    "CREATE CONSTRAINT rfp_chunk_legacy_identity IF NOT EXISTS "
    "FOR (c:RFPChunk) REQUIRE (c.corpus_id, c.corpus_version, c.chunk_id) IS UNIQUE",
    "CREATE INDEX rfp_document_origin IF NOT EXISTS "
    "FOR (d:RFPDocument) ON (d.corpus_id, d.corpus_version, d.document_origin)",
    "CREATE INDEX rfp_document_file_hash IF NOT EXISTS "
    "FOR (d:RFPDocument) ON (d.corpus_id, d.corpus_version, d.file_hash)",
    "CREATE INDEX rfp_chunk_document_page IF NOT EXISTS "
    "FOR (c:RFPChunk) ON (c.corpus_id, c.corpus_version, c.document_id, c.page_index)",
)

UPSERT_DOCUMENTS = """
UNWIND $rows AS row
MERGE (d:RFPDocument {
    corpus_id: row.corpus_id, corpus_version: row.corpus_version, document_id: row.document_id
})
ON CREATE SET d += row
RETURN properties(d) AS properties
"""

UPSERT_CHUNKS = """
UNWIND $rows AS row
MATCH (d:RFPDocument {
    corpus_id: row.corpus_id, corpus_version: row.corpus_version, document_id: row.document_id
})
WHERE d.file_hash = row.file_hash AND d.source_file = row.source_file
      AND d.document_origin = row.document_origin
MERGE (c:RFPChunk {
    corpus_id: row.corpus_id, corpus_version: row.corpus_version, evidence_id: row.evidence_id
})
ON CREATE SET c += row
MERGE (c)-[:IN_DOCUMENT]->(d)
RETURN properties(c) AS properties
"""
