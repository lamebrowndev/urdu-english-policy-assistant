CREATE TABLE answer_audit (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at timestamptz NOT NULL DEFAULT now(),
    actor text NOT NULL,
    principal_groups text[] NOT NULL,
    question text NOT NULL,
    question_language text NOT NULL CHECK (question_language IN ('ur', 'en')),
    outcome text NOT NULL CHECK (
        outcome IN (
            'answered',
            'abstained_retrieval',
            'abstained_model',
            'abstained_uncited',
            'refused',
            'error'
        )
    ),
    answer_text text NOT NULL,
    answer jsonb,
    retrieval jsonb NOT NULL,
    llm jsonb,
    flags text[] NOT NULL DEFAULT '{}',
    prompt_version text NOT NULL,
    pipeline_fingerprint text NOT NULL,
    latency_ms jsonb NOT NULL
);

CREATE INDEX answer_audit_created_at_idx ON answer_audit (created_at DESC);
CREATE INDEX answer_audit_outcome_idx ON answer_audit (outcome);

CREATE FUNCTION answer_audit_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'answer_audit is append-only';
END;
$$;

CREATE TRIGGER answer_audit_no_update_delete
BEFORE UPDATE OR DELETE ON answer_audit
FOR EACH ROW EXECUTE FUNCTION answer_audit_append_only();

CREATE TRIGGER answer_audit_no_truncate
BEFORE TRUNCATE ON answer_audit
FOR EACH STATEMENT EXECUTE FUNCTION answer_audit_append_only();
