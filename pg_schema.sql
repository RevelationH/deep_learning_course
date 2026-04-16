CREATE TABLE IF NOT EXISTS deep_learning_users (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS deep_learning_user_wrong_questions (
    wrong_id BIGSERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    keypoint TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    std_answer TEXT NOT NULL DEFAULT '',
    user_answer TEXT NOT NULL DEFAULT '',
    source_doc_id TEXT,
    asked_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (username, keypoint, source_doc_id)
);

CREATE INDEX IF NOT EXISTS idx_dl_wrong_questions_user_keypoint
    ON deep_learning_user_wrong_questions (username, keypoint, asked_at DESC);

CREATE TABLE IF NOT EXISTS deep_learning_progress_users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    account_name TEXT NOT NULL DEFAULT '',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_attempt_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS deep_learning_attempts (
    attempt_pk BIGSERIAL PRIMARY KEY,
    user_id TEXT NOT NULL,
    source_attempt_id TEXT,
    timestamp TIMESTAMPTZ NOT NULL,
    kp_id TEXT NOT NULL DEFAULT '',
    kp_name TEXT NOT NULL DEFAULT '',
    question_id TEXT NOT NULL DEFAULT '',
    question_type TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    submitted_answer TEXT NOT NULL DEFAULT '',
    reference_answer TEXT NOT NULL DEFAULT '',
    is_correct BOOLEAN NOT NULL DEFAULT FALSE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (user_id, source_attempt_id)
);

CREATE INDEX IF NOT EXISTS idx_dl_attempts_user_timestamp
    ON deep_learning_attempts (user_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_dl_attempts_user_question
    ON deep_learning_attempts (user_id, question_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_dl_attempts_user_kp
    ON deep_learning_attempts (user_id, kp_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS deep_learning_latest_attempts (
    user_id TEXT NOT NULL,
    question_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    kp_id TEXT NOT NULL DEFAULT '',
    kp_name TEXT NOT NULL DEFAULT '',
    question_type TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    submitted_answer TEXT NOT NULL DEFAULT '',
    reference_answer TEXT NOT NULL DEFAULT '',
    is_correct BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (user_id, question_id)
);

CREATE INDEX IF NOT EXISTS idx_dl_latest_attempts_user_timestamp
    ON deep_learning_latest_attempts (user_id, timestamp DESC);

CREATE TABLE IF NOT EXISTS deep_learning_followup_cache (
    user_id TEXT PRIMARY KEY,
    cache_key TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    items JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE TABLE IF NOT EXISTS deep_learning_chat_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '',
    title_generated BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_preview TEXT NOT NULL DEFAULT '',
    message_count INTEGER NOT NULL DEFAULT 0,
    course TEXT NOT NULL DEFAULT 'deep_learning',
    session_summary TEXT NOT NULL DEFAULT '',
    active_topic TEXT NOT NULL DEFAULT '',
    summary_updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_dl_chat_sessions_user_updated
    ON deep_learning_chat_sessions (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS deep_learning_chat_messages (
    message_pk BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES deep_learning_chat_sessions(session_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    source_message_id TEXT,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL DEFAULT '',
    citations JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    order_index INTEGER NOT NULL,
    mode TEXT NOT NULL DEFAULT '',
    UNIQUE (session_id, order_index),
    UNIQUE (session_id, source_message_id)
);

CREATE INDEX IF NOT EXISTS idx_dl_chat_messages_session_order
    ON deep_learning_chat_messages (session_id, order_index ASC);

CREATE INDEX IF NOT EXISTS idx_dl_chat_messages_user_created
    ON deep_learning_chat_messages (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS deep_learning_chat_jobs (
    job_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    result JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    worker_id TEXT NOT NULL DEFAULT '',
    request_meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_dl_chat_jobs_status_created
    ON deep_learning_chat_jobs (status, created_at ASC);

CREATE INDEX IF NOT EXISTS idx_dl_chat_jobs_user_status
    ON deep_learning_chat_jobs (user_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS deep_learning_learning_report_snapshots (
    user_id TEXT PRIMARY KEY,
    signature_key TEXT NOT NULL DEFAULT '',
    target_signature_key TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'empty'
        CHECK (status IN ('empty', 'queued', 'running', 'ready', 'failed')),
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    generated_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_message TEXT NOT NULL DEFAULT '',
    worker_id TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_dl_report_snapshots_status_updated
    ON deep_learning_learning_report_snapshots (status, updated_at ASC);

CREATE TABLE IF NOT EXISTS deep_learning_migration_meta (
    migration_key TEXT PRIMARY KEY,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
