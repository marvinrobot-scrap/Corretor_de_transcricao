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


# ============== CONFIGURAÇÕES ==============
MODELO_LM_STUDIO = "meta-llama-3.1-8b-instruct"
URL_LM_STUDIO = "http://localhost:1234/v1/chat/completions"
MODELO_WHISPER = "large"  # Opções: tiny, base, small, medium, large
# ===========================================

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
    # Remove a extensão
    nome_base, _ = os.path.splitext(nome_arquivo)
    
    # Tenta encontrar o separador underscore
    if "_" not in nome_base:
        return None, None
    
    # Divide pelo último underscore
    partes = nome_base.rsplit("_", 1)
    if len(partes) != 2:
        return None, None
    
    nome_depoente = partes[0].strip()
    papel_depoente = partes[1].strip().lower()
    
    # Normaliza "vitima" para "vítima"
    if papel_depoente == "vitima":
        papel_depoente = "vítima"
    
    # Verifica se o papel é válido
    if papel_depoente not in PAPEIS_VALIDOS:
        return nome_depoente, None
    
    return nome_depoente, papel_depoente


def transcrever_audio(caminho_arquivo, model):
    """Transcreve áudio/vídeo usando Whisper."""
    print(f"Transcrevendo: {os.path.basename(caminho_arquivo)}")
    print("Isso pode levar alguns minutos dependendo do tamanho do arquivo...")
    
    result = model.transcribe(caminho_arquivo, language="pt")
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
        "Regras obrigatórias de formatação e conteúdo:\n"
        "1) Converter a transcrição para uma narrativa em terceira pessoa, removendo perguntas e respostas diretas.\n"
        "2) Iniciar a narrativa de forma direta, utilizando o formato exato: "
        "\"[NOME_DO_DEPONENTE], ouvido em juízo, disse que\" (substituindo [NOME_DO_DEPONENTE] corretamente).\n"
        "3) Referir-se às partes conforme o papel indicado (vítima, testemunha, informante, acusado, promotor, juiz), "
        "utilizando um tom impessoal e formal.\n"
        "4) Corrigir palavras trocadas ou mal reconhecidas durante a transcrição, garantindo que o texto faça sentido "
        "dentro do contexto.\n"
        "5) Ajustar ortografia, gramática e pontuação em língua portuguesa (Brasil).\n"
        "6) Estruturar o texto para que pareça um registro formal do depoimento, em tom adequado ao contexto judicial.\n"
        "7) Indicar claramente qualquer trecho efetivamente ininteligível, usando exatamente: \"[trecho ininteligível]\".\n"
        "8) NÃO incluir número de processo, data ou horário da audiência, ainda que constem na transcrição.\n"
        "9) Completar frases truncadas ou desconexas, respeitando a lógica do depoimento e o contexto fornecido, "
        "sem inventar fatos que não guardem relação com o que está no texto.\n"
        "10) Resolver trechos de difícil entendimento interpretando o contexto e reconstruindo o sentido de forma coerente, "
        "desde que plausível a partir da transcrição.\n"
        "11) Não reproduzir juramentos, qualificações pessoais excessivas (nome dos pais, RG, CPF, endereço etc.) "
        "ou formalidades do ato, a menos que sejam relevantes para a compreensão do conteúdo do depoimento.\n"
        "12) A saída deve ser um texto contínuo, sem parágrafos, pronto para ser salvo como .txt, sem cabeçalhos extras.\n"
        "13) Suprimir as perguntas ou, quando estritamente necessário para o entendimento, convertê-las em trechos "
        "narrativos breves (por exemplo: \"Indagado sobre determinado fato, o depoente afirmou que...\").\n"
        "14) Presumir que há apenas UM depoente no arquivo recebido. Caso apareçam falas de juiz, promotor, defensor etc., "
        "devem ser utilizadas apenas como contexto para compreender melhor as respostas do depoente, não devendo "
        "ser transcritas literalmente como perguntas e respostas.\n"
    )


def construir_prompt_usuario(nome_depoente, papel_depoente, transcricao):
    instrucoes_papel = (
        f"O depoente se chama \"{nome_depoente}\" e deve ser tratado na narrativa como "
        f"\"{papel_depoente}\" (por exemplo: \"a vítima\", \"a testemunha\", \"o acusado\").\n\n"
        "Use, quando adequado, expressões como \"a vítima relatou\", \"a testemunha afirmou\", "
        "\"o acusado declarou\", mantendo sempre a coerência com o papel indicado.\n\n"
    )

    texto_usuario = (
        instrucoes_papel
        + "A seguir está a transcrição bruta da audiência, em formato de perguntas e respostas, "
          "podendo constar falas do juiz, promotor, defensor e do próprio depoente. "
          "Seu foco deve ser extrair apenas o teor do depoimento deste único depoente, "
          "conforme as regras do sistema.\n\n"
          "[INÍCIO DA TRANSCRIÇÃO]\n"
        + transcricao
        + "\n[FIM DA TRANSCRIÇÃO]\n\n"
        "Agora, gere apenas a narrativa final em terceira pessoa, já revisada, no formato solicitado. "
        "Não explique o que fez, não adicione comentários metajurídicos, nem títulos. "
        "Entregue somente o texto final da narrativa."
    )

    return texto_usuario


def gerar_narrativa(nome_depoente, papel_depoente, transcricao):
    """Gera narrativa usando LM Studio."""
    system_prompt = construir_prompt_sistema()
    user_prompt = construir_prompt_usuario(nome_depoente, papel_depoente, transcricao)

    payload = {
        "model": MODELO_LM_STUDIO,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4096
    }

    try:
        response = requests.post(URL_LM_STUDIO, json=payload, timeout=300)
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
        print("  3) Se a URL está correta.")
        print(f"  4) Se o modelo '{MODELO_LM_STUDIO}' está carregado.")
        return None


def processar_arquivo(caminho_midia, nome_arquivo, result_dir, whisper_model):
    """Processa um único arquivo de mídia."""
    
    # Extrair nome e papel do nome do arquivo
    nome_depoente, papel_depoente = extrair_nome_papel(nome_arquivo)
    
    if not nome_depoente or not papel_depoente:
        print(f"\n[ERRO] Não foi possível extrair nome/papel do arquivo: {nome_arquivo}")
        print(f"       Formato esperado: Nome do Depoente_papel.extensao")
        print(f"       Exemplo: João Silva_testemunha.mp4")
        print(f"       Papéis válidos: {', '.join(PAPEIS_VALIDOS)}")
        return False
    
    print(f"\n  Depoente: {nome_depoente}")
    print(f"  Papel: {papel_depoente}")
    
    # Preparar nomes de saída
    nome_base, _ = os.path.splitext(nome_arquivo)
    caminho_transcricao = os.path.join(result_dir, f"{nome_base}_transcricao.txt")
    caminho_narrativa = os.path.join(result_dir, f"{nome_base}_narrativa.txt")
    
    # Etapa 1: Transcrição com Whisper
    print("\n  [1/2] Transcrevendo áudio...")
    transcricao = transcrever_audio(caminho_midia, whisper_model)
    
    if not escrever_arquivo_txt(caminho_transcricao, transcricao):
        return False
    print(f"        Transcrição salva: {os.path.basename(caminho_transcricao)}")
    
    # Etapa 2: Geração de narrativa com LM Studio
    print("\n  [2/2] Gerando narrativa...")
    narrativa = gerar_narrativa(
        nome_depoente=nome_depoente,
        papel_depoente=papel_depoente,
        transcricao=transcricao
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
        description="Transcrever vídeos/áudios de audiências e converter em narrativas jurídicas (processamento em lote)."
    )

    parser.add_argument(
        "-w", "--whisper-model",
        default=MODELO_WHISPER,
        choices=["tiny", "base", "small", "medium", "large"],
        help=f"Modelo Whisper a usar (padrão: {MODELO_WHISPER})."
    )

    args = parser.parse_args()

    # Criar estrutura de pastas
    script_dir, temp_dir, result_dir = criar_pastas()
    
    # Listar arquivos de mídia
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
    print(f"\nArquivos encontrados: {len(arquivos)}")
    for i, arq in enumerate(arquivos, 1):
        print(f"  {i}. {arq}")
    
    # Carregar modelo Whisper uma única vez
    print(f"\nCarregando modelo Whisper '{args.whisper_model}'...")
    whisper_model = whisper.load_model(args.whisper_model)
    print("Modelo carregado com sucesso!")
    
    # Processar cada arquivo
    sucessos = 0
    falhas = 0
    arquivos_com_erro = []
    
    for i, nome_arquivo in enumerate(arquivos, 1):
        print("\n" + "="*60)
        print(f"PROCESSANDO ARQUIVO {i}/{len(arquivos)}: {nome_arquivo}")
        print("="*60)
        
        caminho_midia = os.path.join(temp_dir, nome_arquivo)
        
        if processar_arquivo(caminho_midia, nome_arquivo, result_dir, whisper_model):
            sucessos += 1
        else:
            falhas += 1
            arquivos_com_erro.append(nome_arquivo)
    
    # Resumo final
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