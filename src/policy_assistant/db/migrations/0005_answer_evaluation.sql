CREATE TABLE eval_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz NOT NULL DEFAULT now(),
    actor text NOT NULL,
    dataset_name text NOT NULL,
    dataset_sha256 text NOT NULL CHECK (dataset_sha256 ~ '^[0-9a-f]{64}$'),
    split text NOT NULL CHECK (split IN ('sample', 'dev', 'test')),
    item_count integer NOT NULL CHECK (item_count > 0),
    settings jsonb NOT NULL
);

CREATE TABLE eval_results (
    run_id uuid NOT NULL REFERENCES eval_runs(id),
    item_id text NOT NULL,
    audit_id uuid NOT NULL UNIQUE REFERENCES answer_audit(id),
    language text NOT NULL CHECK (language IN ('ur', 'en')),
    category text NOT NULL CHECK (category IN ('answerable', 'unanswerable', 'restricted')),
    tags text[] NOT NULL DEFAULT '{}',
    reference_answer text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, item_id)
);

CREATE TRIGGER eval_runs_no_update_delete
BEFORE UPDATE OR DELETE ON eval_runs
FOR EACH ROW EXECUTE FUNCTION reject_modification();

CREATE TRIGGER eval_runs_no_truncate
BEFORE TRUNCATE ON eval_runs
FOR EACH STATEMENT EXECUTE FUNCTION reject_modification();

CREATE TRIGGER eval_results_no_update_delete
BEFORE UPDATE OR DELETE ON eval_results
FOR EACH ROW EXECUTE FUNCTION reject_modification();

CREATE TRIGGER eval_results_no_truncate
BEFORE TRUNCATE ON eval_results
FOR EACH STATEMENT EXECUTE FUNCTION reject_modification();
