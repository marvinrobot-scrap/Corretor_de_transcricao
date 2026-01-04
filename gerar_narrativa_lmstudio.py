import argparse
import os
import sys
import requests


# Nome do modelo fixo - altere aqui se necessário
MODELO_LM_STUDIO = "meta-llama-3.1-8b-instruct"


def ler_arquivo_txt(caminho):
    try:
        with open(caminho, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"Erro ao ler o arquivo de entrada: {e}")
        sys.exit(1)


def escrever_arquivo_txt(caminho, conteudo):
    try:
        with open(caminho, "w", encoding="utf-8") as f:
            f.write(conteudo)
    except Exception as e:
        print(f"Erro ao escrever o arquivo de saída: {e}")
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
        "12) A saída deve ser um texto contínuo, em um único parágrafo, pronto para ser salvo como .txt, sem cabeçalhos extras.\n"
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


def gerar_narrativa(nome_depoente, papel_depoente, transcricao, url_servidor):
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
        response = requests.post(url_servidor, json=payload, timeout=300)
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
        description="Converter transcrição de audiência (.txt) em narrativa em terceira pessoa usando LM Studio."
    )

    parser.add_argument(
        "entrada",
        help="Caminho do arquivo .txt com a transcrição do depoimento."
    )
    parser.add_argument(
        "-o", "--saida",
        help=(
            "Caminho do arquivo .txt de saída com a narrativa. "
            "Se não informado, será usado o nome do arquivo de entrada com sufixo _narrativa.txt."
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
        "-u", "--url",
        default="http://192.168.0.63:1234/v1/chat/completions",
        help=(
            "URL do servidor LM Studio (padrão: http://192.168.0.63:1234/v1/chat/completions). "
            "Ajuste se o LM Studio indicar outra."
        )
    )

    args = parser.parse_args()

    caminho_entrada = args.entrada
    if not os.path.isfile(caminho_entrada):
        print(f"Arquivo de entrada não encontrado: {caminho_entrada}")
        sys.exit(1)

    if args.saida:
        caminho_saida = args.saida
    else:
        base, ext = os.path.splitext(caminho_entrada)
        caminho_saida = base + "_narrativa.txt"

    nome_depoente = args.nome.strip()
    papel_depoente = args.papel.strip().lower()
    if papel_depoente == "vitima":
        papel_depoente = "vítima"

    transcricao = ler_arquivo_txt(caminho_entrada)

    print(f"Usando modelo: {MODELO_LM_STUDIO}")
    print("Gerando narrativa em terceira pessoa usando LM Studio. Isso pode levar alguns minutos...")

    narrativa = gerar_narrativa(
        nome_depoente=nome_depoente,
        papel_depoente=papel_depoente,
        transcricao=transcricao,
        url_servidor=args.url
    )

    escrever_arquivo_txt(caminho_saida, narrativa)

    print(f"Narrativa gerada com sucesso em: {caminho_saida}")


if __name__ == "__main__":
    main()