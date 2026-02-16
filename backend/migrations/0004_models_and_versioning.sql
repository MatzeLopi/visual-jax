-- Add versioning and visibility to datasets
ALTER TABLE datasets ADD COLUMN version INT NOT NULL DEFAULT 1;
ALTER TABLE datasets ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT FALSE;

-- Update existing datasets to be public if user_id is NULL
UPDATE datasets SET is_public = TRUE WHERE user_id IS NULL;

-- Create models table
CREATE TABLE IF NOT EXISTS "models" (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES "users"(user_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    version INT NOT NULL DEFAULT 1,
    is_public BOOLEAN NOT NULL DEFAULT FALSE,
    graph_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

-- Add updated_at trigger for models
SELECT trigger_updated_at('"models"');
