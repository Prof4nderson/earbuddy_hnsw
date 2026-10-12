import os
import time
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

API_URL = "http://localhost:8000/api/v1"
SAMPLE_RATE = 16000
DURATION = 3.0  # Duração em segundos
WAV_DIR = "recordings"

# Garante que a pasta de gravações exista
os.makedirs(WAV_DIR, exist_ok=True)

def audio_to_embedding(audio_data: np.ndarray, sr: int = 16000, target_dim: int = 512) -> list[float]:
    """Converte o sinal de áudio mono gravado num vetor de 512 dimensões."""
    y = audio_data.flatten().astype(np.float32)
    
    # Extração de características acústicas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral, axis=1)
    ])
    
    # Ajusta para exatamente 512 dimensões
    embedding = np.interp(np.linspace(0, len(features), target_dim), np.arange(len(features)), features)
    
    # Normalização L2
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()

def record_and_save_audio(label: str, duration: float = 3.0, sr: int = 16000) -> tuple[np.ndarray, str]:
    """Grava o áudio do microfone e salva uma cópia em formato .wav na pasta recordings/"""
    print(f"\n🔴 [GRAVANDO] Fale ou faça o som AGORA ({duration}s)...", flush=True)
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("⏹️ [GRAVAÇÃO CONCLUÍDA]", flush=True)
    
    # Nome do arquivo limpo com data/hora
    timestamp = int(time.time())
    safe_label = "".join([c if c.isalnum() else "_" for c in label])
    filename = os.path.join(WAV_DIR, f"{safe_label}_{timestamp}.wav")
    
    # Salva o arquivo .wav físico no disco
    sf.write(filename, audio, sr)
    print(f"📁 Arquivo .wav gravado com sucesso em: {filename}")
    
    return audio, filename

def register_sound_mode():
    """Modo de Cadastro: Grava o som, salva o .wav e registra no HNSW/Postgres."""
    label = input("\n👉 Digite o rótulo do som (ex: 'Latido do Bob', 'Voz do Anderson'): ").strip()
    if not label:
        print("❌ Rótulo inválido.")
        return

    audio_data, wav_path = record_and_save_audio(label, DURATION, SAMPLE_RATE)
    vector = audio_to_embedding(audio_data, SAMPLE_RATE)

    payload = {
        "label": label,
        "embedding": vector
    }

    try:
        response = requests.post(f"{API_URL}/voices/register", json=payload)
        if response.status_code == 201:
            print(f"🎉 SUCESSO: '{label}' cadastrado no índice HNSW do Postgres!")
        else:
            print(f"⚠️ Resposta da API ({response.status_code}): {response.json().get('detail', response.text)}")
    except Exception as e:
        print(f"❌ Erro ao conectar na API: {e}")

def test_sound_mode():
    """Modo de Match: Grava o som, salva o .wav e consulta o HNSW/Postgres."""
    raw_label = input("\n👉 Informe o rótulo do Hot Path / YAMNet (padrão: 'Speech' ou 'Bark'): ").strip() or "Speech"
    
    audio_data, wav_path = record_and_save_audio("teste_match", DURATION, SAMPLE_RATE)
    vector = audio_to_embedding(audio_data, SAMPLE_RATE)

    payload = {
        "raw_label": raw_label,
        "confidence": 0.95,
        "embedding": vector
    }

    try:
        response = requests.post(f"{API_URL}/audio/process", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("\n--- RESULTADO DA BUSCA VETORIAL HNSW ---")
            print(f"🎧 Classificação Bruta (Hot Path): {data['raw_label']}")
            print(f"🔍 Rótulo Refinado (Cold Path):  {data['refined_label']}")
            print(f"📊 Similaridade de Cosseno:     {data['similarity']:.2%}")
            print(f"✅ Emissor Conhecido?            {'SIM' if data['is_known_speaker'] else 'NÃO'}")
            print(f"📝 Resumo de Contexto:           {data['context_summary']}")
            # Na função test_sound_mode() do earbuddy_recorder.py, adicione:
            print(f"🧠 Análise LLM (Camada 3):       {data.get('llm_synthesis')}")
        else:
            print(f"❌ Erro na API ({response.status_code}): {response.text}")
    except Exception as e:
        print(f"❌ Erro ao conectar na API: {e}")

if __name__ == "__main__":
    while True:
        print("\n==========================================")
        print(" 🎙️ EARBUDDY - GRAVADOR & TESTADOR AUTOMÁTICO")
        print("==========================================")
        print("1. Cadastrar novo som (Gravar + Salvar .wav + Registrar no PG)")
        print("2. Testar áudio ao vivo (Gravar + Salvar .wav + Buscar no HNSW)")
        print("3. Sair")
        
        choice = input("Escolha uma opção (1/2/3): ").strip()
        
        if choice == "1":
            register_sound_mode()
        elif choice == "2":
            test_sound_mode()
        elif choice == "3":
            print("Encerrando gravador...")
            break
        else:
            print("Opção inválida!")