import numpy as np
import librosa

def wav_to_embedding(wav_file_path: str, target_dim: int = 512) -> list[float]:
    """
    Lê um arquivo .wav real e extrai um vetor numérico de características (512d)
    baseado na assinatura espectral e frequência do áudio.
    """
    # 1. Carrega o áudio (taxa de amostragem padrão de 16kHz)
    y, sr = librosa.load(wav_file_path, sr=16000, duration=3.0)
    
    # 2. Extrai características acústicas (MFCCs, Chroma, Espectro)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # 3. Agrupa e tira a média ao longo do tempo (resumo acústico do som)
    features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(spectral, axis=1)
    ])
    
    # 4. Ajusta e normaliza o tamanho exato para 512 dimensões
    if len(features) < target_dim:
        # Repete/Completa com interpolação se for menor
        embedding = np.pad(features, (0, target_dim - len(features)), mode='constant')
    else:
        # Interpola para exatamente 512 dimensões
        embedding = np.interp(np.linspace(0, len(features), target_dim), np.arange(len(features)), features)
        
    # Normalização L2 (para a busca por Similaridade de Cosseno funcionar com precisão)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.tolist()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        wav_path = sys.argv[1]
        vec = wav_to_embedding(wav_path)
        print(f"✅ Sucesso! Áudio '{wav_path}' convertido em vetor de {len(vec)} posições.")
        print(f"Primeiros 5 valores do vetor: {vec[:5]}")
    else:
        print("Uso: python extract_embedding.py caminho_do_audio.wav")