-- Enable pgvector extension on startup
CREATE EXTENSION IF NOT EXISTS vector;

-- Confirm
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
