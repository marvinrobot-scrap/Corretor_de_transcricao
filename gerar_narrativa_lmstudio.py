import os
import sys

# --- NOVA SOLUÇÃO ROBUSTA PARA DLLS NVIDIA ---
def configurar_ambiente_gpu():
    """Localiza e carrega as bibliotecas CUDA instaladas via pip."""
    import importlib.util
    
    for modulo_nome in ["nvidia.cublas", "nvidia.cudnn"]:
        spec = importlib.util.find_spec(modulo_nome)
        if spec and spec.origin:
            # Pega a pasta 'bin' dentro do pacote instalado
            modulo_dir = os.path.dirname(spec.origin)
            bin_path = os.path.join(modulo_dir, "bin")
            
            if os.path.exists(bin_path):
                # Adiciona ao PATH para compatibilidade geral
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                # Método específico para Python 3.8+ no Windows
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_path)
                    except Exception:
                        pass

configurar_ambiente_gpu()
# ---------------------------------------------

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