import os
import sys
import time

# --- SOLUÇÃO PARA DLLS NVIDIA ---
def configurar_ambiente_gpu():
    import importlib.util
    for modulo_nome in ["nvidia.cublas", "nvidia.cudnn"]:
        spec = importlib.util.find_spec(modulo_nome)
        if spec and spec.origin:
            modulo_dir = os.path.dirname(spec.origin)
            bin_path = os.path.join(modulo_dir, "bin")
            if os.path.exists(bin_path):
                os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(bin_path)
                    except: pass

configurar_ambiente_gpu()

import requests

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Erro: O módulo 'faster-whisper' não está instalado.")
    input("\nPressione Enter para sair...")
    sys.exit(1)

# ============== CONFIGURAÇÕES ==============
MODELO_LM_STUDIO = "qwen2.5-7b-instruct-1m@q8_0" 
URL_LM_STUDIO_PADRAO = "http://localhost:1234/v1/chat/completions"
MODELO_WHISPER = "large-v3" 
# ===========================================

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
    print(f"\n[Whisper] Transcrevendo: {os.path.basename(caminho_arquivo)}")
    segments, _ = model.transcribe(
        caminho_arquivo, 
        language="pt", 
        beam_size=2,
        vad_filter=True
    )
    lista_segmentos = list(segments)
    texto_completo = " ".join([s.text.strip() for s in lista_segmentos])
    return {
        "text": texto_completo,
        "segments": [{"start": s.start, "text": s.text} for s in lista_segmentos]
    }

def gerar_narrativa(nome, papel, transcricao, url, modelo_llm):
    print(f"[LLM] Gerando narrativa jurídica...")
    system_prompt = (
        "Você é um assistente jurídico especializado. Converta a transcrição em uma narrativa formal "
        "em terceira pessoa. Inicie com: '[NOME], ouvido(a) em juízo, disse que'. "
        "Remova perguntas, corrija erros gramaticais e mantenha o tom solene."
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
    try:
        temp_dir, result_dir = criar_pastas()
        
        # LISTA AMPLIADA DE FORMATOS (Incluindo .webm)
        extensoes_suportadas = (
            '.mp4', '.mp3', '.wav', '.m4a', '.webm', 
            '.mkv', '.avi', '.mov', '.flac', '.ogg', '.wmv'
        )
        
        print(f"Carregando {MODELO_WHISPER} na RTX 4070 (Float16)...")
        try:
            model = WhisperModel(MODELO_WHISPER, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"\nAVISO: Não foi possível usar a GPU. Erro: {e}")
            print("Tentando carregar na CPU (será lento)...")
            model = WhisperModel(MODELO_WHISPER, device="cpu", compute_type="int8")

        arquivos = [f for f in os.listdir(temp_dir) if f.lower().endswith(extensoes_suportadas)]
        
        if not arquivos:
            print(f"\nNenhum arquivo encontrado na pasta: {temp_dir}")
            print(f"Formatos procurados: {', '.join(extensoes_suportadas)}")
            print("Certifique-se de que os arquivos seguem o padrão: Nome_papel.extensao")
        
        for arq in arquivos:
            nome_base, _ = os.path.splitext(arq)
            if "_" not in nome_base:
                print(f"\n[PULADO] Arquivo '{arq}' não possui '_' no nome.")
                print("Use o formato: Nome do Depoente_papel.webm")
                continue
                
            nome, papel = nome_base.rsplit("_", 1)
            res = transcrever_audio(os.path.join(temp_dir, arq), model)
            
            # Salvar Timestamps
            txt_tempos = "\n".join([f"{formatar_timestamp(s['start'])} {s['text'].strip()}" for s in res["segments"]])
            with open(os.path.join(result_dir, f"{nome_base}_timestamps.txt"), "w", encoding="utf-8") as f:
                f.write(txt_tempos)

            # Gerar Narrativa
            narrativa = gerar_narrativa(nome, papel, res["text"], URL_LM_STUDIO_PADRAO, MODELO_LM_STUDIO)
            if narrativa:
                with open(os.path.join(result_dir, f"{nome_base}_narrativa.txt"), "w", encoding="utf-8") as f:
                    f.write(narrativa)
                print(f"Sucesso: {arq}")

    except Exception as e:
        print(f"\n--- ERRO CRÍTICO ---")
        print(e)
    
    input("\nFim do processo. Pressione Enter para fechar...")

if __name__ == "__main__":
    main()