CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE groups (
    id text PRIMARY KEY CHECK (id ~ '^[a-z][a-z0-9_]{1,62}$'),
    description text NOT NULL
);

CREATE TABLE documents (
    id text PRIMARY KEY CHECK (id ~ '^[a-z0-9][a-z0-9-]{1,79}$'),
    title text NOT NULL,
    language text NOT NULL CHECK (language IN ('ur', 'en')),
    file_type text NOT NULL CHECK (file_type IN ('pdf', 'txt')),
    source_url text,
    source_sha256 text NOT NULL CHECK (source_sha256 ~ '^[0-9a-f]{64}$'),
    translation_of text,
    license_note text NOT NULL,
    page_count integer NOT NULL CHECK (page_count > 0),
    pipeline_fingerprint text NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE document_acl (
    document_id text REFERENCES documents ON DELETE CASCADE,
    group_id text REFERENCES groups ON DELETE RESTRICT,
    PRIMARY KEY (document_id, group_id)
);

CREATE INDEX document_acl_group_id_idx ON document_acl (group_id);

CREATE TABLE pages (
    document_id text REFERENCES documents ON DELETE CASCADE,
    page_number integer CHECK (page_number >= 1),
    text text NOT NULL,
    included boolean NOT NULL,
    quality jsonb NOT NULL,
    PRIMARY KEY (document_id, page_number)
);

CREATE TABLE chunks (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id text NOT NULL REFERENCES documents ON DELETE CASCADE,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    sentences text[] NOT NULL CHECK (cardinality(sentences) > 0),
    text text NOT NULL,
    section_heading text,
    language text NOT NULL CHECK (language IN ('ur', 'en', 'mixed')),
    page_start integer NOT NULL,
    page_end integer NOT NULL,
    token_count integer NOT NULL CHECK (token_count > 0),
    content_sha256 text NOT NULL,
    UNIQUE (document_id, ordinal),
    CHECK (page_start >= 1 AND page_end >= page_start)
);
