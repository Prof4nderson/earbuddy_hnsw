import os
import sys

# Força a libpq do Postgres e a saída do terminal a usarem UTF-8
os.environ["PGCLIENTENCODING"] = "utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

import psycopg2
import numpy as np

DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "secret",
    "host": "127.0.0.1",
    "port": "5433"  # <--- MUDOU PARA 5433
}

def init_db(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS earbuddy_audio_events (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                detected_at TIMESTAMPTZ DEFAULT NOW(),
                raw_label VARCHAR(100) NOT NULL,
                refined_label VARCHAR(100),
                confidence REAL,
                audio_embedding vector(512)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_earbuddy_hnsw 
            ON earbuddy_audio_events 
            USING hnsw (audio_embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64);
        """)
    conn.commit()
    print("⚡ Banco de Dados e Índice HNSW inicializados com sucesso!", flush=True)

def register_voice_sample(conn, label, embedding):
    with conn.cursor() as cur:
        sql = """
            INSERT INTO earbuddy_audio_events (raw_label, refined_label, confidence, audio_embedding)
            VALUES (%s, %s, %s, %s::vector);
        """
        cur.execute(sql, ("Speech", label, 1.0, embedding.tolist()))
    conn.commit()
    print(f"✅ Assinatura de voz cadastrada: '{label}'", flush=True)

def match_voice_hnsw(conn, raw_label, current_embedding, threshold=0.85):
    if raw_label.lower() not in ["speech", "fala"]:
        return raw_label, 1.0

    query_sql = """
        SELECT refined_label, (1 - (audio_embedding <=> %s::vector)) AS similarity
        FROM earbuddy_audio_events
        WHERE refined_label IS NOT NULL
        ORDER BY audio_embedding <=> %s::vector
        LIMIT 1;
    """
    
    vec_list = current_embedding.tolist()
    with conn.cursor() as cur:
        cur.execute(query_sql, (vec_list, vec_list))
        result = cur.fetchone()

    if result:
        matched_label, similarity = result
        if similarity >= threshold:
            return matched_label, similarity

    return "Fala Não Identificada", 0.0

if __name__ == "__main__":
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.set_client_encoding('UTF8')
        
        # 1. Prepara o banco
        init_db(conn)

        # 2. Gera o vetor 'DNA' da Voz do Anderson (512 dimensões)
        np.random.seed(42)
        anderson_voice_vector = np.random.rand(512).astype(np.float32)

        # 3. Cadastra a voz do Anderson no banco
        register_voice_sample(conn, "Voz do Anderson", anderson_voice_vector)

        # 4. TESTE A: Simula captura com pequeno ruído (Sua voz)
        noise = np.random.normal(0, 0.03, 512).astype(np.float32)
        incoming_audio_vector = anderson_voice_vector + noise

        label_a, similarity_a = match_voice_hnsw(conn, "Speech", incoming_audio_vector)

        print("\n--- RESULTADO DA BUSCA HNSW ---", flush=True)
        print(f"🎧 [Captura 1 - Voz com Ruído] YAMNet: Speech", flush=True)
        print(f"🔍 [Refinamento HNSW] Rótulo: '{label_a}' (Similaridade: {similarity_a:.2%})", flush=True)

        # 5. TESTE B: Simula voz de outra pessoa
        unknown_voice_vector = np.random.rand(512).astype(np.float32)
        label_b, similarity_b = match_voice_hnsw(conn, "Speech", unknown_voice_vector)

        print(f"\n🎧 [Captura 2 - Outra Pessoa] YAMNet: Speech", flush=True)
        print(f"🔍 [Refinamento HNSW] Rótulo: '{label_b}' (Similaridade: {similarity_b:.2%})", flush=True)

        conn.close()

    except Exception as e:
        msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
        print(f"❌ Erro na execução: {msg}", flush=True)