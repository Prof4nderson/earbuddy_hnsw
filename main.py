import os
import json
import numpy as np
import psycopg2
import requests
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="EarBuddy Cold Path & Reasoning API",
    version="1.1.0",
    description="API para refinamento vetorial HNSW (PostgreSQL) e síntese de contexto por LLM."
)

DB_PARAMS = {
    "dbname": "earbuddy_db",
    "user": "earbuddy_user",
    "password": "earbuddy_password",
    "host": "localhost",
    "port": "5433"  # Porta mapeada no Docker
}

# Configurações para a Camada 3 (LLM)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        conn.set_client_encoding('UTF8')
        return conn
    except Exception as e:
        raw_bytes = bytes(str(e), encoding='utf-8', errors='ignore')
        clean_msg = raw_bytes.decode('utf-8', errors='ignore') or repr(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro no banco de dados: {clean_msg}"
        )

# --- MODELOS PYDANTIC ---

class VoiceRegistrationRequest(BaseModel):
    label: str = Field(..., example="Voz do Anderson")
    embedding: List[float] = Field(..., min_items=512, max_items=512)

class AudioEventRequest(BaseModel):
    raw_label: str = Field(..., example="Speech")
    confidence: float = Field(..., example=0.95)
    embedding: List[float] = Field(..., min_items=512, max_items=512)

class AudioEventResponse(BaseModel):
    raw_label: str
    confidence: float
    refined_label: str
    similarity: float
    is_known_speaker: bool
    context_summary: str
    llm_synthesis: str  # <--- Nova propriedade com a análise contextual gerada pela LLM


# --- CAMADA 3: RACIOCÍNIO CONTEXTUAL POR LLM ---

def generate_llm_context_synthesis(raw_label: str, refined_label: str, similarity: float, is_known: bool) -> str:
    """
    Sintetiza um resumo de áudio amigável, seguro e acionável utilizando uma LLM local (Ollama)
    ou um gerador contextual resiliente (fallback).
    """
    prompt = f"""
Você é o motor de raciocínio contextual do EarBuddy, um sistema vestível/assistivo de acessibilidade acústica.
Analise os seguintes dados do evento sonoro capturado e gere UMA ÚNICA frase curta, clara e informativa em Português para a interface gráfica/tátil do usuário.

Dados do Evento:
- Classificação Bruta (Hot Path): {raw_label}
- Identificação Específica (Cold Path HNSW): {refined_label}
- Precisão/Similaridade Vetorial: {similarity:.1%}
- Emissor Conhecido: {'Sim' if is_known else 'Não'}

Exemplo de resposta esperada:
"Identificado um assobio no ambiente com 98% de precisão. Pode ser um sinal de chamamento ou atenção próxima."
"""
    try:
        # Tenta invocar o Ollama local caso esteja rodando
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "max_tokens": 100}
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=2.5)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
    except Exception:
        # Fallback inteligente (execução rápida em tempo real sem dependência externa obrigatoria)
        pass

    if is_known:
        return f"Acoustic Context: Som confirmado como '{refined_label}' ({similarity:.1%} de similaridade). Emissor cadastrado reconhecido com sucesso."
    else:
        return f"Acoustic Context: Evento bruto '{raw_label}' detectado. Perfil não coincide totalmente com assinaturas cadastradas."


# --- ENDPOINTS ---

@app.post("/api/v1/voices/register", status_code=status.HTTP_201_CREATED)
def register_voice(payload: VoiceRegistrationRequest):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        vec_str = "[" + ",".join(map(str, payload.embedding)) + "]"
        
        # Checa colisão
        cursor.execute("""
            SELECT label, 1 - (audio_embedding <=> %s::vector) AS similarity
            FROM voice_signatures
            ORDER BY audio_embedding <=> %s::vector
            LIMIT 1;
        """, (vec_str, vec_str))
        top_match = cursor.fetchone()

        if top_match and top_match["similarity"] > 0.985 and top_match["label"] != payload.label:
            cursor.close()
            conn.close()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Assinatura idêntica já cadastrada como '{top_match['label']}' ({top_match['similarity']:.2%})."
            )

        cursor.execute("""
            INSERT INTO voice_signatures (label, audio_embedding)
            VALUES (%s, %s::vector)
            ON CONFLICT (label) 
            DO UPDATE SET audio_embedding = EXCLUDED.audio_embedding;
        """, (payload.label, vec_str))
        
        conn.commit()
        cursor.close()
        conn.close()
        return {"message": f"Assinatura '{payload.label}' registrada com sucesso no HNSW!"}

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        if conn:
            conn.rollback()
            conn.close()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/audio/process", response_model=AudioEventResponse)
def process_audio_event(payload: AudioEventRequest):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        vec_str = "[" + ",".join(map(str, payload.embedding)) + "]"

        # Busca HNSW via similaridade de cosseno
        cursor.execute("""
            SELECT id, label, 1 - (audio_embedding <=> %s::vector) AS similarity
            FROM voice_signatures
            ORDER BY audio_embedding <=> %s::vector
            LIMIT 1;
        """, (vec_str, vec_str))

        result = cursor.fetchone()
        cursor.close()
        conn.close()

        SIMILARITY_THRESHOLD = 0.80

        if result and result["similarity"] >= SIMILARITY_THRESHOLD:
            refined_label = result["label"]
            similarity = float(result["similarity"])
            is_known = True
            context_summary = f"Emissor identificado como '{refined_label}' com {similarity:.2%} de precisão."
        elif result:
            refined_label = f"Desconhecido (Próximo de '{result['label']}')"
            similarity = float(result["similarity"])
            is_known = False
            context_summary = f"Áudio não atingiu o limiar de {SIMILARITY_THRESHOLD:.0%} de confiança."
        else:
            refined_label = payload.raw_label
            similarity = 0.0
            is_known = False
            context_summary = "Nenhuma assinatura cadastrada no banco de dados."

        # Invocação da Camada 3 (Síntese LLM)
        llm_synthesis = generate_llm_context_synthesis(
            raw_label=payload.raw_label,
            refined_label=refined_label,
            similarity=similarity,
            is_known=is_known
        )

        return AudioEventResponse(
            raw_label=payload.raw_label,
            confidence=payload.confidence,
            refined_label=refined_label,
            similarity=similarity,
            is_known_speaker=is_known,
            context_summary=context_summary,
            llm_synthesis=llm_synthesis
        )

    except Exception as e:
        if conn:
            conn.close()
        raise HTTPException(status_code=500, detail=str(e))