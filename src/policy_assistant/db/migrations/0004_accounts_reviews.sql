CREATE TABLE users (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    username text NOT NULL UNIQUE CHECK (username ~ '^[a-z0-9][a-z0-9._-]{1,62}$'),
    display_name text NOT NULL CHECK (length(display_name) BETWEEN 1 AND 100),
    password_hash text NOT NULL,
    role text NOT NULL CHECK (role IN ('user', 'reviewer', 'admin')),
    is_active boolean NOT NULL DEFAULT true,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE user_groups (
    user_id bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    group_id text NOT NULL REFERENCES groups(id) ON DELETE RESTRICT,
    PRIMARY KEY (user_id, group_id)
);

CREATE INDEX user_groups_group_id_idx ON user_groups (group_id);

CREATE TABLE sessions (
    token_hash bytea PRIMARY KEY,
    user_id bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    csrf_token text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL
);

CREATE INDEX sessions_user_id_idx ON sessions (user_id);
CREATE INDEX sessions_expires_at_idx ON sessions (expires_at);

CREATE TABLE answer_reviews (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    audit_id uuid NOT NULL REFERENCES answer_audit(id),
    reviewer_id bigint NOT NULL REFERENCES users(id),
    reviewer_username text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    verdict text NOT NULL CHECK (
        verdict IN (
            'correct',
            'partially_correct',
            'incorrect',
            'correct_abstention',
            'missed_answer'
        )
    ),
    citation_ok boolean,
    notes text NOT NULL DEFAULT '' CHECK (length(notes) <= 4000)
);

CREATE INDEX answer_reviews_audit_id_created_at_idx
    ON answer_reviews (audit_id, created_at DESC);

CREATE FUNCTION reject_modification() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
END;
$$;

CREATE TRIGGER answer_reviews_no_update_delete
BEFORE UPDATE OR DELETE ON answer_reviews
FOR EACH ROW EXECUTE FUNCTION reject_modification();

CREATE TRIGGER answer_reviews_no_truncate
BEFORE TRUNCATE ON answer_reviews
FOR EACH STATEMENT EXECUTE FUNCTION reject_modification();

CREATE INDEX answer_audit_actor_created_at_idx
    ON answer_audit (actor, created_at DESC);
