import argparse
import os
import sys
import requests

try:
    import whisper
except ImportError:
    print("Erro: O módulo 'whisper' não está instalado.")
    print("Instale com: pip install openai-whisper")
    print("Também é necessário ter o FFmpeg instalado no sistema.")
    sys.exit(1)


# ============== CONFIGURAÇÕES ==============
MODELO_LM_STUDIO = "meta-llama-3.1-8b-instruct"
URL_LM_STUDIO = "http://192.168.0..144:1234/v1/chat/completions"
MODELO_WHISPER = "large-v3"
# ===========================================


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
    
    return arquivos


def transcrever_audio(caminho_arquivo, modelo_whisper):
    """Transcreve áudio/vídeo usando Whisper."""
    print(f"Carregando modelo Whisper '{modelo_whisper}'...")
    model = whisper.load_model(modelo_whisper)
    
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
        sys.exit(1)


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
        "12) A saída deve ser um texto contínuo, em parágrafos, pronto para ser salvo como .txt, sem cabeçalhos extras.\n"
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
            sys.exit(1)
        resultado = response.json()
        return resultado["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Erro ao conectar com LM Studio: {e}")
        print("Verifique:")
        print("  1) Se o LM Studio está aberto.")
        print("  2) Se o servidor local está rodando na aba 'Local Server'.")
        print("  3) Se a URL está correta.")
        print(f"  4) Se o modelo '{MODELO_LM_STUDIO}' está carregado.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Transcrever vídeo/áudio de audiência e converter em narrativa jurídica."
    )

    parser.add_argument(
        "-a", "--arquivo",
        help=(
            "Nome do arquivo de vídeo/áudio na pasta 'temp'. "
            "Se não informado, processa o primeiro arquivo encontrado."
        )
    )
    parser.add_argument(
        "-n", "--nome",
        required=True,
        help="Nome completo do depoente (ex.: 'Fulano de Tal')."
    )
    parser.add_argument(
        "-p", "--papel",
        required=True,
        choices=["vítima", "vitima", "testemunha", "informante", "acusado"],
        help="Papel do depoente: vítima, testemunha, informante ou acusado."
    )
    parser.add_argument(
        "-w", "--whisper-model",
        default=MODELO_WHISPER,
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help=f"Modelo Whisper a usar (padrão: {MODELO_WHISPER})."
    )

    args = parser.parse_args()

    # Criar estrutura de pastas
    script_dir, temp_dir, result_dir = criar_pastas()
    
    # Encontrar arquivo de mídia
    if args.arquivo:
        caminho_midia = os.path.join(temp_dir, args.arquivo)
        if not os.path.isfile(caminho_midia):
            print(f"Arquivo não encontrado: {caminho_midia}")
            sys.exit(1)
        nome_arquivo = args.arquivo
    else:
        arquivos = listar_arquivos_midia(temp_dir)
        if not arquivos:
            print(f"Nenhum arquivo de vídeo/áudio encontrado na pasta: {temp_dir}")
            print("Extensões suportadas: .mp4, .mp3, .wav, .m4a, .webm, .mkv, .avi, .mov, .flac, .ogg")
            sys.exit(1)
        nome_arquivo = arquivos[0]
        caminho_midia = os.path.join(temp_dir, nome_arquivo)
        print(f"Arquivo encontrado: {nome_arquivo}")

    # Preparar nomes de saída
    nome_base, _ = os.path.splitext(nome_arquivo)
    caminho_transcricao = os.path.join(result_dir, f"{nome_base}_transcricao.txt")
    caminho_narrativa = os.path.join(result_dir, f"{nome_base}_narrativa.txt")

    # Normalizar papel
    nome_depoente = args.nome.strip()
    papel_depoente = args.papel.strip().lower()
    if papel_depoente == "vitima":
        papel_depoente = "vítima"

    # Etapa 1: Transcrição com Whisper
    print("\n" + "="*50)
    print("ETAPA 1: TRANSCRIÇÃO COM WHISPER")
    print("="*50)
    
    transcricao = transcrever_audio(caminho_midia, args.whisper_model)
    escrever_arquivo_txt(caminho_transcricao, transcricao)
    print(f"Transcrição salva em: {caminho_transcricao}")

    # Etapa 2: Geração de narrativa com LM Studio
    print("\n" + "="*50)
    print("ETAPA 2: GERAÇÃO DE NARRATIVA COM LM STUDIO")
    print("="*50)
    print(f"Usando modelo: {MODELO_LM_STUDIO}")
    print("Gerando narrativa em terceira pessoa. Isso pode levar alguns minutos...")

    narrativa = gerar_narrativa(
        nome_depoente=nome_depoente,
        papel_depoente=papel_depoente,
        transcricao=transcricao
    )

    escrever_arquivo_txt(caminho_narrativa, narrativa)
    print(f"Narrativa salva em: {caminho_narrativa}")

    # Resumo final
    print("\n" + "="*50)
    print("PROCESSAMENTO CONCLUÍDO!")
    print("="*50)
    print(f"Arquivos gerados na pasta 'result':")
    print(f"  - Transcrição: {nome_base}_transcricao.txt")
    print(f"  - Narrativa:   {nome_base}_narrativa.txt")


if __name__ == "__main__":
    main()