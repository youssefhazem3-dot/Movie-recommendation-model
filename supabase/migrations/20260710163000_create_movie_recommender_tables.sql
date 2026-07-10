CREATE TABLE IF NOT EXISTS public.users (
    id BIGSERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS public.feedback (
    id BIGSERIAL PRIMARY KEY,
    app_user_id BIGINT NOT NULL REFERENCES public.users(id),
    netflix_user_id BIGINT NOT NULL,
    movie_id BIGINT NOT NULL,
    title TEXT NOT NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    review_text TEXT NOT NULL,
    sentiment_label TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS public.chat_messages (
    id BIGSERIAL PRIMARY KEY,
    app_user_id BIGINT NOT NULL REFERENCES public.users(id),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    message TEXT NOT NULL,
    movie_id BIGINT,
    title TEXT,
    predicted_rating DOUBLE PRECISION,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS feedback_app_user_id_created_at_idx
    ON public.feedback (app_user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS chat_messages_app_user_id_created_at_idx
    ON public.chat_messages (app_user_id, created_at ASC);