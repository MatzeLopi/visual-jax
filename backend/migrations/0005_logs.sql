CREATE TABLE IF NOT EXISTS "logs" (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    origin_uid UUID,
    logs TEXT,
    created_at timestamptz not null default now()
    
);