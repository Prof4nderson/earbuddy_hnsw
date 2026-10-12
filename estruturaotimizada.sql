-- 1. Habilita a extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Tabela principal de eventos do EarBuddy
CREATE TABLE IF NOT EXISTS earbuddy_audio_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    raw_label VARCHAR(100) NOT NULL,       -- Rótulo do YAMNet (ex: "Speech", "Train", "Alarm")
    refined_label VARCHAR(100),            -- Rótulo refinado (ex: "Voz do Anderson")
    confidence REAL,                       -- Confiança da detecção
    audio_embedding vector(512)            -- Vetor de biometria/características do som
);

-- 3. Índice HNSW ultrarrápido para busca por similaridade de cosseno
CREATE INDEX IF NOT EXISTS idx_earbuddy_hnsw 
ON earbuddy_audio_events 
USING hnsw (audio_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);