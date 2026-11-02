ALTER TABLE chunks ADD COLUMN embedding vector(1024);
ALTER TABLE chunks ADD COLUMN bm25_length integer CHECK (bm25_length >= 0);

CREATE TABLE chunk_terms (
    term text NOT NULL,
    chunk_id bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    tf integer NOT NULL CHECK (tf > 0),
    PRIMARY KEY (term, chunk_id)
);

CREATE INDEX chunk_terms_chunk_id_idx ON chunk_terms (chunk_id);
CREATE INDEX chunks_embedding_hnsw_idx ON chunks USING hnsw (embedding vector_cosine_ops);
