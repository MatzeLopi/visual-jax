CREATE TABLE IF NOT EXISTS "datasets" (
    dataset_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES "users"(user_id) ON DELETE CASCADE, 
    name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);