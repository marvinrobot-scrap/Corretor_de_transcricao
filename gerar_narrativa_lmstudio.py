import os
import sys

# --- SOLUÇÃO PARA ERRO DE DLL NVIDIA ---
def carregar_dlls_nvidia():
    """Adiciona as pastas das bibliotecas NVIDIA ao PATH do sistema."""
    try:
        import nvidia.cublas
        import nvidia.cudnn
        cublas_path = os.path.join(os.path.dirname(nvidia.cublas.__file__), "bin")
        cudnn_path = os.path.join(os.path.dirname(nvidia.cudnn.__file__), "bin")
        os.environ["PATH"] += os.pathsep + cublas_path + os.pathsep + cudnn_path
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(cublas_path)
            os.add_dll_directory(cudnn_path)
    except ImportError:
        pass

carregar_dlls_nvidia()
# ---------------------------------------

import argparse
import requests

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Erro: O módulo 'faster-whisper' não está instalado.")
    sys.exit(1)

# ============== CONFIGURAÇÕES ==============
MODELO_LM_STUDIO = "qwen2.5-7b-instruct-1m@q8_0" 
URL_LM_STUDIO_PADRAO = "http://localhost:1234/v1/chat/completions"
MODELO_WHISPER = "large-v3" 
# ===========================================

PAPEIS_VALIDOS = ["vitima", "vítima", "testemunha", "informante", "acusado"]

def criar_pastas():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp")
    result_dir = os.path.join(script_dir, "result")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    return temp_dir, result_dir

def formatar_timestamp(segundos):
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    return f"[{horas:02d}:{minutos:02d}:{segs:02d}]"

def transcrever_audio(caminho_arquivo, model):
    """Transcrição ultra-rápida usando Cuda e Float16 na RTX 4070."""
    print(f"\n[Whisper] Transcrevendo: {os.path.basename(caminho_arquivo)}")
    
    segments, _ = model.transcribe(
        caminho_arquivo, 
        language="pt", 
        beam_size=2,
        vad_filter=True, # Remove silêncios para ganhar tempo
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    lista_segmentos = list(segments)
    texto_completo = " ".join([s.text.strip() for s in lista_segmentos])
    
    return {
        "text": texto_completo,
        "segments": [{"start": s.start, "text": s.text} for s in lista_segmentos]
    }

def gerar_narrativa(nome, papel, transcricao, url, modelo_llm):
    """Envia para o LM Studio."""
    print(f"[LLM] Gerando narrativa jurídica...")
    system_prompt = (
        "Você é um assistente jurídico. Converta a transcrição em uma narrativa formal "
        "em terceira pessoa. Inicie com: '[NOME], ouvido(a) em juízo, disse que'. "
        "Remova perguntas, corrija erros e mantenha o tom solene."
    )
    user_prompt = f"Depoente: {nome} ({papel})\n\nTranscrição:\n{transcricao}"

    payload = {
        "model": modelo_llm,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 8192
    }

    try:
        response = requests.post(url, json=payload, timeout=600)
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Erro no LM Studio: {e}")
        return None

def main():
    temp_dir, result_dir = criar_pastas()
    
    # Carregando modelo na GPU
    print(f"Carregando {MODELO_WHISPER} na RTX 4070 (Float16)...")
    try:
        model = WhisperModel(MODELO_WHISPER, device="cuda", compute_type="float16")
    except Exception as e:
        print(f"Erro ao iniciar GPU: {e}. Tentando modo CPU (lento)...")
        model = WhisperModel(MODELO_WHISPER, device="cpu", compute_type="int8")

    arquivos = [f for f in os.listdir(temp_dir) if f.lower().endswith(('.mp4', '.mp3', '.wav', '.m4a'))]
    
    for arq in arquivos:
        nome_base, _ = os.path.splitext(arq)
        if "_" not in nome_base:
            print(f"Pulei {arq} (Formato Nome_papel.ext necessário)")
            continue
            
        nome, papel = nome_base.rsplit("_", 1)
        res = transcrever_audio(os.path.join(temp_dir, arq), model)
        
        # Salva Timestamps
        txt_tempos = "\n".join([f"{formatar_timestamp(s['start'])} {s['text']}" for s in res["segments"]])
        with open(os.path.join(result_dir, f"{nome_base}_timestamps.txt"), "w", encoding="utf-8") as f:
            f.write(txt_tempos)

        # Narrativa
        narrativa = gerar_narrativa(nome, papel, res["text"], URL_LM_STUDIO_PADRAO, MODELO_LM_STUDIO)
        if narrativa:
            with open(os.path.join(result_dir, f"{nome_base}_narrativa.txt"), "w", encoding="utf-8") as f:
                f.write(narrativa)
            print(f"Concluído: {arq}")

if __name__ == "__main__":
    main()