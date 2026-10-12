import psycopg2
import numpy as np

# Conexão com o banco PostgreSQL
conn = psycopg2.connect("dbname=earbuddy user=postgres password=secret host=localhost")
cur = conn.cursor()

def refine_speech_label(raw_label, input_embedding):
    """
    Se o rótulo base for "Speech" ou "Fala", consulta o HNSW para 
    verificar se o embedding pertence à voz do Anderson.
    """
    if raw_label.lower() not in ["speech", "fala"]:
        return raw_label  # Retorna o rótulo original (ex: "Alarm", "Train")

    # Query HNSW buscando o vetor mais próximo na base de vozes conhecidas
    # Usa a distância de cosseno (<=>)
    query_sql = """
        SELECT refined_label, (1 - (audio_embedding <=> %s::vector)) AS similarity
        FROM audio_events
        WHERE refined_label IS NOT NULL
        ORDER BY audio_embedding <=> %s::vector
        LIMIT 1;
    """
    
    embedding_list = input_embedding.tolist() if isinstance(input_embedding, np.ndarray) else input_embedding
    cur.execute(query_sql, (embedding_list, embedding_list))
    result = cur.fetchone()

    if result:
        matched_label, similarity = result
        # Limiar de corte (threshold) para confirmar que é a sua voz
        if similarity >= 0.85:
            return f"{matched_label} (Match: {similarity:.2%})"

    return "Fala Não Identificada"

# Exemplo de teste simples:
# vetor_teste_capturado = np.random.rand(512)
# print(refine_speech_label("Speech", vetor_teste_capturado))