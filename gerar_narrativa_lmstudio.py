import argparse
import os
import sys
import requests

# Tenta importar o faster-whisper
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Erro: O módulo 'faster-whisper' não está instalado.")
    print("Instale com: py -m pip install faster-whisper")
    sys.exit(1)

# ============== CONFIGURAÇÕES PADRÃO ==============
MODELO_LM_STUDIO = "qwen2.5-7b-instruct-1m@q8_0" 
URL_LM_STUDIO_PADRAO = "http://localhost:1234/v1/chat/completions"
MODELO_WHISPER = "large-v3" 
# ==================================================

PAPEIS_VALIDOS = ["vitima", "vítima", "testemunha", "informante", "acusado"]

def criar_pastas():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp")
    result_dir = os.path.join(script_dir, "result")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    return script_dir, temp_dir, result_dir

def listar_arquivos_midia(temp_dir):
    extensoes_validas = {'.mp4', '.mp3', '.wav', '.m4a', '.webm', '.mkv', '.avi', '.mov', '.flac', '.ogg'}
    return sorted([f for f in os.listdir(temp_dir) if os.path.splitext(f)[1].lower() in extensoes_validas])

def extrair_nome_papel(nome_arquivo):
    nome_base, _ = os.path.splitext(nome_arquivo)
    if "_" not in nome_base: return None, None
    partes = nome_base.rsplit("_", 1)
    if len(partes) != 2: return None, None
    nome_depoente = partes[0].strip()
    papel_depoente = partes[1].strip().lower()
    if papel_depoente == "vitima": papel_depoente = "vítima"
    return (nome_depoente, papel_depoente) if papel_depoente in PAPEIS_VALIDOS else (nome_depoente, None)

def formatar_timestamp(segundos):
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    return f"[{horas:02d}:{minutos:02d}:{segs:02d}]"

def transcrever_audio(caminho_arquivo, model):
    """Usa faster-whisper para máxima performance na RTX 4070."""
    print(f"Transcrevendo com aceleração GPU: {os.path.basename(caminho_arquivo)}")
    
    # beam_size=2 é o equilíbrio ideal para sua GPU
    segments, info = model.transcribe(
        caminho_arquivo, 
        language="pt", 
        beam_size=2,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    lista_segmentos = list(segments)
    texto_completo = " ".join([seg.text.strip() for seg in lista_segmentos])
    
    return {
        "text": texto_completo,
        "segments": [{"start": seg.start, "text": seg.text} for seg in lista_segmentos]
    }

def escrever_arquivo_txt(caminho, conteudo):
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
        return True
    except Exception as e:
        print(f"Erro ao escrever arquivo: {e}")
        return False

def gerar_narrativa(nome_depoente, papel_depoente, transcricao, url_lm_studio, modelo_lm):
    system_prompt = (
        "Você é um assistente jurídico especializado. Converta a transcrição em uma narrativa formal "
        "em terceira pessoa. Inicie com: '[NOME], ouvido(a) em juízo, disse que'. "
        "Remova perguntas, corrija erros gramaticais e mantenha o tom solene."
    )
    user_prompt = f"Depoente: {nome_depoente} ({papel_depoente})\n\nTranscrição:\n{transcricao}"

    payload = {
        "model": modelo_lm,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 8192
    }

    try:
        response = requests.post(url_lm_studio, json=payload, timeout=600)
        return response.json()["choices"][0]["message"]["content"].strip()
    except:
        return None

def processar_arquivo(caminho_midia, nome_arquivo, result_dir, whisper_model, url_url, modelo_llm):
    nome_depoente, papel_depoente = extrair_nome_papel(nome_arquivo)
    if not nome_depoente or not papel_depoente:
        print(f"Erro no nome do arquivo: {nome_arquivo}")
        return False

    nome_base, _ = os.path.splitext(nome_arquivo)
    res = transcrever_audio(caminho_midia, whisper_model)
    
    # Salvar Transcrição Limpa
    escrever_arquivo_txt(os.path.join(result_dir, f"{nome_base}_transcricao.txt"), res["text"])
    
    # Salvar Timestamps
    texto_tempos = "\n".join([f"{formatar_timestamp(s['start'])} {s['text'].strip()}" for s in res["segments"]])
    escrever_arquivo_txt(os.path.join(result_dir, f"{nome_base}_timestamps.txt"), texto_tempos)
    
    # Gerar Narrativa
    print(f"Gerando narrativa via LM Studio...")
    narrativa = gerar_narrativa(nome_depoente, papel_depoente, res["text"], url_url, modelo_llm)
    if narrativa:
        escrever_arquivo_txt(os.path.join(result_dir, f"{nome_base}_narrativa.txt"), narrativa)
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--whisper-model", default=MODELO_WHISPER)
    parser.add_argument("-u", "--url", default=URL_LM_STUDIO_PADRAO)
    parser.add_argument("-m", "--modelo", default=MODELO_LM_STUDIO)
    args = parser.parse_args()

    _, temp_dir, result_dir = criar_pastas()
    arquivos = listar_arquivos_midia(temp_dir)
    
    if not arquivos:
        print("Pasta /temp vazia.")
        return

    print(f"Carregando {args.whisper_model} na RTX 4070...")
    # Configuração otimizada para sua GPU
    whisper_model = WhisperModel(args.whisper_model, device="cuda", compute_type="float16")
    
    for arq in arquivos:
        processar_arquivo(os.path.join(temp_dir, arq), arq, result_dir, whisper_model, args.url, args.modelo)

if __name__ == "__main__":
    main()