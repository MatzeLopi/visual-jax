CREATE TYPE log_severity AS ENUM ('INFO', 'ERROR', 'WARN', 'DEBUG');

ALTER TABLE logs ADD COLUMN severity log_severity NOT NULL DEFAULT 'INFO';