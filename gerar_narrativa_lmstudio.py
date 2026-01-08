import argparse
import os
import sys
import requests

try:
    import whisper
except ImportError:
    print("Erro: O módulo 'whisper' não está instalado.")
    print("Instale com: py -m pip install openai-whisper")
    print("Também é necessário ter o FFmpeg instalado no sistema.")
    sys.exit(1)


# ============== CONFIGURAÇÕES PADRÃO ==============
MODELO_LM_STUDIO = "qwen2.5-7b-instruct-1m@q8_0"
URL_LM_STUDIO_PADRAO = "http://localhost:1234/v1/chat/completions"
MODELO_WHISPER = "large-v3"
# ==================================================

PAPEIS_VALIDOS = ["vitima", "vítima", "testemunha", "informante", "acusado"]


def criar_pastas():
    """Cria as pastas temp e result se não existirem."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp")
    result_dir = os.path.join(script_dir, "result")
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    return script_dir, temp_dir, result_dir


def listar_arquivos_midia(temp_dir):
    """Lista arquivos de vídeo/áudio na pasta temp."""
    extensoes_validas = {'.mp4', '.mp3', '.wav', '.m4a', '.webm', '.mkv', '.avi', '.mov', '.flac', '.ogg'}
    arquivos = []
    for arquivo in os.listdir(temp_dir):
        _, ext = os.path.splitext(arquivo)
        if ext.lower() in extensoes_validas:
            arquivos.append(arquivo)
    return sorted(arquivos)


def extrair_nome_papel(nome_arquivo):
    """Extrai nome do depoente e papel a partir do nome do arquivo."""
    nome_base, _ = os.path.splitext(nome_arquivo)
    if "_" not in nome_base:
        return None, None
    partes = nome_base.rsplit("_", 1)
    if len(partes) != 2:
        return None, None
    nome_depoente = partes[0].strip()
    papel_depoente = partes[1].strip().lower()
    if papel_depoente == "vitima":
        papel_depoente = "vítima"
    if papel_depoente not in PAPEIS_VALIDOS:
        return nome_depoente, None
    return nome_depoente, papel_depoente


def formatar_timestamp(segundos):
    """Converte segundos para o formato [HH:MM:SS]."""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    return f"[{horas:02d}:{minutos:02d}:{segs:02d}]"


def transcrever_audio(caminho_arquivo, model):
    """Transcreve áudio/vídeo usando Whisper e retorna o dicionário completo de resultados."""
    print(f"Transcrevendo: {os.path.basename(caminho_arquivo)}")
    
    result = model.transcribe(
        caminho_arquivo,
        language="pt",
        task="transcribe",
        beam_size=5,
        best_of=5,
        temperature=0,
        condition_on_previous_text=True,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=True,
        verbose=False
    )
    return result


def escrever_arquivo_txt(caminho, conteudo):
    """Escreve conteúdo em arquivo de texto."""
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
    except Exception as e:
        print(f"Erro ao escrever o arquivo: {e}")
        return False
    return True


def construir_prompt_sistema():
    return (
        "Você é um assistente jurídico especializado em direito processual penal brasileiro. "
        "Sua tarefa é receber a transcrição de uma audiência judicial e convertê-la em uma narrativa formal em terceira pessoa.\n\n"
        "REGRAS OBRIGATÓRIAS:\n"
        "1) Converter para terceira pessoa, removendo perguntas e respostas.\n"
        "2) Iniciar EXATAMENTE com: \"[NOME_DO_DEPONENTE], ouvido(a) em juízo, disse que\".\n"
        "3) Usar tom formal e impessoal.\n"
        "4) Corrigir erros de transcrição e gramática.\n"
        "5) Indicar ininteligíveis com: \"[trecho ininteligível]\".\n"
        "6) NÃO incluir formalidades, juramentos ou dados de RG/CPF.\n"
        "7) Saída em parágrafos contínuos, sem títulos."
    )


def construir_prompt_usuario(nome_depoente, papel_depoente, transcricao):
    return (
        f"DADOS DO DEPOENTE:\n- Nome: {nome_depoente}\n- Papel: {papel_depoente}\n\n"
        f"TRANSCRIÇÃO BRUTA:\n"
        f"{'='*50}\n{transcricao}\n{'='*50}\n\n"
        "Gere APENAS a narrativa final."
    )


def gerar_narrativa(nome_depoente, papel_depoente, transcricao, url_lm_studio, modelo_lm):
    """Gera narrativa usando LM Studio."""
    system_prompt = construir_prompt_sistema()
    user_prompt = construir_prompt_usuario(nome_depoente, papel_depoente, transcricao)

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
        if response.status_code >= 400:
            return None
        resultado = response.json()
        return resultado["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def processar_arquivo(caminho_midia, nome_arquivo, result_dir, whisper_model, url_lm_studio, modelo_lm):
    """Processa um único arquivo gerando transcrição limpa, com tempo e narrativa."""
    nome_depoente, papel_depoente = extrair_nome_papel(nome_arquivo)
    
    if not nome_depoente or not papel_depoente:
        print(f"[ERRO] Nome de arquivo inválido: {nome_arquivo}")
        return False
    
    nome_base, _ = os.path.splitext(nome_arquivo)
    caminho_transcricao = os.path.join(result_dir, f"{nome_base}_transcricao.txt")
    caminho_timestamps = os.path.join(result_dir, f"{nome_base}_timestamps.txt")
    caminho_narrativa = os.path.join(result_dir, f"{nome_base}_narrativa.txt")
    
    # 1. Transcrição
    print(f"\n  [1/2] Transcrevendo com Whisper...")
    res = transcrever_audio(caminho_midia, whisper_model)
    
    # Gerar versão com marcadores de tempo
    texto_com_tempos = ""
    for seg in res["segments"]:
        texto_com_tempos += f"{formatar_timestamp(seg['start'])} {seg['text'].strip()}\n"
    
    escrever_arquivo_txt(caminho_transcricao, res["text"])
    escrever_arquivo_txt(caminho_timestamps, texto_com_tempos)
    print(f"        Arquivos de transcrição e timestamps salvos.")
    
    # 2. Narrativa
    print(f"  [2/2] Gerando narrativa com LM Studio...")
    narrativa = gerar_narrativa(nome_depoente, papel_depoente, res["text"], url_lm_studio, modelo_lm)
    
    if narrativa:
        escrever_arquivo_txt(caminho_narrativa, narrativa)
        print(f"        Narrativa salva com sucesso.")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Processamento de audiências com timestamps.")
    parser.add_argument("-w", "--whisper-model", default=MODELO_WHISPER)
    parser.add_argument("-u", "--url", default=URL_LM_STUDIO_PADRAO)
    parser.add_argument("-m", "--modelo", default=MODELO_LM_STUDIO)
    args = parser.parse_args()

    _, temp_dir, result_dir = criar_pastas()
    arquivos = listar_arquivos_midia(temp_dir)
    
    if not arquivos:
        print("Nenhum arquivo encontrado em /temp.")
        sys.exit(1)
    
    print(f"Carregando Whisper {args.whisper_model}...")
    whisper_model = whisper.load_model(args.whisper_model)
    
    for nome_arquivo in arquivos:
        caminho_midia = os.path.join(temp_dir, nome_arquivo)
        processar_arquivo(caminho_midia, nome_arquivo, result_dir, whisper_model, args.url, args.modelo)


if __name__ == "__main__":
    main()