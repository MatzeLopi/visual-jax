CREATE TABLE IF NOT EXISTS "models" (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id UUID NOT NULL,
    user_id UUID REFERENCES "users"(user_id) ON DELETE CASCADE,
    version_ INT NOT NULL DEFAULT 1,
    model_name TEXT,
    model_description TEXT,
    model_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE models ADD CONSTRAINT unique_uid_version UNIQUE (model_id, version_);

CREATE OR REPLACE FUNCTION set_model_version()
RETURNS TRIGGER AS $$
BEGIN

    SELECT COALESCE(MAX(version_), 0) + 1
    INTO NEW.version_
    FROM models
    WHERE model_id = NEW.model_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_model_version
BEFORE INSERT ON models
FOR EACH ROW
EXECUTE FUNCTION set_model_version();

CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_timestamp
BEFORE UPDATE ON models
FOR EACH ROW
EXECUTE FUNCTION update_timestamp();