CREATE TABLE IF NOT EXISTS "models" (
    model_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES "users"(user_id) ON DELETE RESTRICT,
    model_name TEXT,
    model_description TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS "model_versions" (
    id INT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id UUID NOT NULL REFERENCES "models"(model_id) ON DELETE CASCADE,
    version_ INT NOT NULL DEFAULT 1,
    model_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_model_version UNIQUE (model_id, version_)
);

CREATE OR REPLACE FUNCTION set_model_version()
RETURNS TRIGGER AS $$
BEGIN
    SELECT COALESCE(MAX(version_), 0) + 1
    INTO NEW.version_
    FROM model_versions
    WHERE model_id = NEW.model_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_set_model_version
BEFORE INSERT ON model_versions
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