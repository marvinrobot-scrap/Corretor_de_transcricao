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
# Modelos recomendados para português (em ordem de qualidade):
# 1. Qwen3-8B ou Qwen2.5-7B-Instruct - Excelente para português
# 2. boto-9B-it - Específico para português brasileiro
# 3. Qwen3-14B - Melhor qualidade se tiver hardware
# 4. meta-llama-3.1-8b-instruct - Bom, mas inferior aos Qwen para PT-BR

MODELO_LM_STUDIO = "qwen2.5-7b-instruct-1m@q8_0"  # Altere conforme o modelo carregado
URL_LM_STUDIO_PADRAO = "http://localhost:1234/v1/chat/completions"
MODELO_WHISPER = "large-v3"  # Máxima precisão para transcrição
# ==================================================

# Papéis válidos para extração do nome do arquivo
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
    """
    Extrai nome do depoente e papel a partir do nome do arquivo.
    Formato esperado: Nome do Depoente_papel.extensao
    Exemplo: João Silva_testemunha.mp4 -> ("João Silva", "testemunha")
    """
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


def transcrever_audio(caminho_arquivo, model):
    """
    Transcreve áudio/vídeo usando Whisper com configurações otimizadas para máxima precisão.
    """
    print(f"Transcrevendo: {os.path.basename(caminho_arquivo)}")
    print("Usando configurações otimizadas para máxima precisão...")
    print("Isso pode levar alguns minutos dependendo do tamanho do arquivo...")
    
    # Configurações otimizadas para máxima precisão:
    # - language="pt": Força português (evita detecção errada)
    # - beam_size=5: Aumenta precisão (padrão é 1)
    # - best_of=5: Considera mais candidatos
    # - temperature=0: Determinístico, mais preciso
    # - condition_on_previous_text=True: Usa contexto anterior
    # - compression_ratio_threshold=2.4: Filtra segmentos ruins
    # - logprob_threshold=-1.0: Aceita mais palavras
    # - no_speech_threshold=0.6: Detecta melhor silêncios
    
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
        word_timestamps=True,  # Timestamps por palavra para melhor precisão
        verbose=False
    )
    
    return result["text"]


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
        "Sua tarefa é receber a transcrição de uma audiência judicial (um único depoimento) "
        "e convertê-la em uma narrativa formal em terceira pessoa, adequada a um contexto judicial.\n\n"
        "REGRAS OBRIGATÓRIAS - SIGA RIGOROSAMENTE:\n\n"
        "1) Converter a transcrição para uma narrativa em terceira pessoa, removendo perguntas e respostas diretas.\n\n"
        "2) Iniciar a narrativa EXATAMENTE com o formato: \"[NOME_DO_DEPONENTE], ouvido(a) em juízo, disse que\" "
        "(substituindo [NOME_DO_DEPONENTE] pelo nome real fornecido).\n\n"
        "3) Referir-se às partes conforme o papel indicado (vítima, testemunha, informante, acusado), "
        "utilizando um tom impessoal e formal.\n\n"
        "4) CORRIGIR palavras mal reconhecidas pela transcrição automática, garantindo que o texto faça sentido "
        "dentro do contexto jurídico e factual do depoimento.\n\n"
        "5) Ajustar ortografia, gramática e pontuação em língua portuguesa (Brasil).\n\n"
        "6) Estruturar o texto como um registro formal do depoimento, em tom adequado ao contexto judicial.\n\n"
        "7) Indicar trechos REALMENTE ininteligíveis com: \"[trecho ininteligível]\". "
        "Use isso APENAS quando for impossível deduzir o significado pelo contexto.\n\n"
        "8) NÃO incluir número de processo, data ou horário da audiência.\n\n"
        "9) Completar frases truncadas respeitando a lógica do depoimento. "
        "NÃO INVENTE fatos que não estejam implícitos na transcrição.\n\n"
        "10) Resolver trechos confusos interpretando o contexto de forma coerente.\n\n"
        "11) NÃO reproduzir juramentos, qualificações pessoais (RG, CPF, endereço) ou formalidades do ato.\n\n"
        "12) A saída deve ser texto contínuo em parágrafos, SEM cabeçalhos, títulos ou comentários.\n\n"
        "13) Suprimir perguntas. Quando necessário para o entendimento, use: "
        "\"Indagado(a) sobre [tema], afirmou que...\" ou \"Questionado(a), esclareceu que...\"\n\n"
        "14) Presumir que há apenas UM depoente. Falas de juiz, promotor e defensor são apenas contexto.\n\n"
        "IMPORTANTE: Seja FIEL ao conteúdo da transcrição. Não adicione informações, "
        "não faça inferências além do texto, não invente detalhes. "
        "Apenas reorganize e formalize o que foi dito."
    )


def construir_prompt_usuario(nome_depoente, papel_depoente, transcricao):
    instrucoes_papel = (
        f"DADOS DO DEPOENTE:\n"
        f"- Nome: {nome_depoente}\n"
        f"- Papel processual: {papel_depoente}\n\n"
        f"Use expressões como \"a {papel_depoente} relatou\", \"a {papel_depoente} afirmou\", "
        f"\"a {papel_depoente} declarou\", mantendo coerência com o papel indicado.\n\n"
    )

    texto_usuario = (
        instrucoes_papel
        + "TRANSCRIÇÃO BRUTA DA AUDIÊNCIA:\n"
        + "(Pode conter erros de reconhecimento de fala, palavras trocadas e falas de juiz/promotor/defensor)\n\n"
        + "="*50 + "\n"
        + transcricao
        + "\n" + "="*50 + "\n\n"
        + "TAREFA: Gere APENAS a narrativa final em terceira pessoa, seguindo TODAS as regras do sistema. "
        + "Não explique o que fez, não adicione comentários. Entregue somente o texto final."
    )

    return texto_usuario


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
        "temperature": 0.2,  # Baixa temperatura para mais precisão e menos criatividade
        "max_tokens": 8192,  # Aumentado para narrativas longas
        "top_p": 0.9,
        "frequency_penalty": 0.1,  # Evita repetições
        "presence_penalty": 0.1
    }

    try:
        response = requests.post(url_lm_studio, json=payload, timeout=600)  # Timeout maior
        if response.status_code >= 400:
            print("HTTP Status:", response.status_code)
            print("Resposta do LM Studio:", response.text)
            return None
        resultado = response.json()
        return resultado["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Erro ao conectar com LM Studio: {e}")
        print("Verifique:")
        print("  1) Se o LM Studio está aberto.")
        print("  2) Se o servidor local está rodando na aba 'Local Server'.")
        print(f"  3) Se a URL está correta: {url_lm_studio}")
        print(f"  4) Se o modelo '{modelo_lm}' está carregado.")
        print("  5) Se o contexto do modelo é suficiente (aumente para 8192+ tokens).")
        return None


def processar_arquivo(caminho_midia, nome_arquivo, result_dir, whisper_model, url_lm_studio, modelo_lm):
    """Processa um único arquivo de mídia."""
    
    nome_depoente, papel_depoente = extrair_nome_papel(nome_arquivo)
    
    if not nome_depoente or not papel_depoente:
        print(f"\n[ERRO] Não foi possível extrair nome/papel do arquivo: {nome_arquivo}")
        print(f"       Formato esperado: Nome do Depoente_papel.extensao")
        print(f"       Exemplo: João Silva_testemunha.mp4")
        print(f"       Papéis válidos: {', '.join(PAPEIS_VALIDOS)}")
        return False
    
    print(f"\n  Depoente: {nome_depoente}")
    print(f"  Papel: {papel_depoente}")
    
    nome_base, _ = os.path.splitext(nome_arquivo)
    caminho_transcricao = os.path.join(result_dir, f"{nome_base}_transcricao.txt")
    caminho_narrativa = os.path.join(result_dir, f"{nome_base}_narrativa.txt")
    
    # Etapa 1: Transcrição com Whisper
    print("\n  [1/2] Transcrevendo áudio com Whisper (configurações de alta precisão)...")
    transcricao = transcrever_audio(caminho_midia, whisper_model)
    
    if not escrever_arquivo_txt(caminho_transcricao, transcricao):
        return False
    print(f"        Transcrição salva: {os.path.basename(caminho_transcricao)}")
    
    # Etapa 2: Geração de narrativa com LM Studio
    print("\n  [2/2] Gerando narrativa com LM Studio...")
    narrativa = gerar_narrativa(
        nome_depoente=nome_depoente,
        papel_depoente=papel_depoente,
        transcricao=transcricao,
        url_lm_studio=url_lm_studio,
        modelo_lm=modelo_lm
    )
    
    if narrativa is None:
        print("        [ERRO] Falha ao gerar narrativa.")
        return False
    
    if not escrever_arquivo_txt(caminho_narrativa, narrativa):
        return False
    print(f"        Narrativa salva: {os.path.basename(caminho_narrativa)}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Transcrever vídeos/áudios de audiências e converter em narrativas jurídicas (processamento em lote).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODELOS RECOMENDADOS PARA PORTUGUÊS (LM Studio):
  1. Qwen2.5-7B-Instruct ou Qwen3-8B - Excelente qualidade para PT-BR
  2. boto-9B-it - Específico para português brasileiro  
  3. Qwen3-14B - Melhor qualidade (requer mais VRAM)
  4. meta-llama-3.1-8b-instruct - Bom, mas inferior aos Qwen

FORMATO DO NOME DOS ARQUIVOS:
  Nome do Depoente_papel.extensao
  Exemplo: João Silva_testemunha.mp4
  Papéis válidos: vitima, testemunha, informante, acusado
        """
    )

    parser.add_argument(
        "-w", "--whisper-model",
        default=MODELO_WHISPER,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help=f"Modelo Whisper (padrão: {MODELO_WHISPER}). Use 'large-v3' para máxima precisão."
    )
    
    parser.add_argument(
        "-u", "--url",
        default=URL_LM_STUDIO_PADRAO,
        help=f"URL do servidor LM Studio (padrão: {URL_LM_STUDIO_PADRAO})."
    )
    
    parser.add_argument(
        "-m", "--modelo",
        default=MODELO_LM_STUDIO,
        help=f"Nome do modelo no LM Studio (padrão: {MODELO_LM_STUDIO})."
    )

    args = parser.parse_args()

    script_dir, temp_dir, result_dir = criar_pastas()
    
    arquivos = listar_arquivos_midia(temp_dir)
    
    if not arquivos:
        print(f"Nenhum arquivo de vídeo/áudio encontrado na pasta: {temp_dir}")
        print("Extensões suportadas: .mp4, .mp3, .wav, .m4a, .webm, .mkv, .avi, .mov, .flac, .ogg")
        print(f"\nFormato do nome do arquivo: Nome do Depoente_papel.extensao")
        print(f"Exemplo: João Silva_testemunha.mp4")
        print(f"Papéis válidos: {', '.join(PAPEIS_VALIDOS)}")
        sys.exit(1)
    
    print("="*60)
    print("PROCESSAMENTO EM LOTE DE AUDIÊNCIAS")
    print("="*60)
    print(f"\nConfigurações:")
    print(f"  URL LM Studio: {args.url}")
    print(f"  Modelo LLM: {args.modelo}")
    print(f"  Modelo Whisper: {args.whisper_model}")
    print(f"\nArquivos encontrados: {len(arquivos)}")
    for i, arq in enumerate(arquivos, 1):
        print(f"  {i}. {arq}")
    
    print(f"\nCarregando modelo Whisper '{args.whisper_model}'...")
    print("(Isso pode demorar na primeira execução devido ao download)")
    whisper_model = whisper.load_model(args.whisper_model)
    print("Modelo Whisper carregado com sucesso!")
    
    sucessos = 0
    falhas = 0
    arquivos_com_erro = []
    
    for i, nome_arquivo in enumerate(arquivos, 1):
        print("\n" + "="*60)
        print(f"PROCESSANDO ARQUIVO {i}/{len(arquivos)}: {nome_arquivo}")
        print("="*60)
        
        caminho_midia = os.path.join(temp_dir, nome_arquivo)
        
        if processar_arquivo(caminho_midia, nome_arquivo, result_dir, whisper_model, args.url, args.modelo):
            sucessos += 1
        else:
            falhas += 1
            arquivos_com_erro.append(nome_arquivo)
    
    print("\n" + "="*60)
    print("PROCESSAMENTO CONCLUÍDO!")
    print("="*60)
    print(f"\nTotal de arquivos: {len(arquivos)}")
    print(f"  Sucessos: {sucessos}")
    print(f"  Falhas: {falhas}")
    
    if arquivos_com_erro:
        print(f"\nArquivos com erro:")
        for arq in arquivos_com_erro:
            print(f"  - {arq}")
    
    print(f"\nResultados salvos na pasta: {result_dir}")


if __name__ == "__main__":
    main()