-- 1. Habilita a extensão de vetores
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Tabela de eventos de áudio com vetor de biometria/áudio (ex: 512 dimensões)
CREATE TABLE audio_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detected_at TIMESTAMPTZ DEFAULT NOW(),
    raw_label VARCHAR(100) NOT NULL,       -- Ex: "Speech", "Train", "Alarm" (do YAMNet/Edge)
    refined_label VARCHAR(100),            -- Ex: "Voz do Anderson" (após processamento HNSW)
    confidence REAL,
    audio_embedding vector(512)           -- Vetor de embedding da voz/som
);

-- 3. Criação do Índice HNSW para busca por similaridade de cosseno
CREATE INDEX ON audio_events 
USING hnsw (audio_embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);