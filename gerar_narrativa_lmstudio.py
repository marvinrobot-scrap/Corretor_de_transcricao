import os
import sys
import time
import requests

# ============== CONFIGURAÇÕES ==============
MODELO_LM_STUDIO = "qwen2.5-7b-instruct-1m@q8_0" 
URL_LM_STUDIO_PADRAO = "http://localhost:1234/v1/chat/completions"
MODELO_WHISPER = "large-v3" 
# ===========================================

# --- LOCALIZADOR DE DLLS NVIDIA (FORÇADO NAS PASTAS ESPECÍFICAS) ---
def configurar_ambiente_gpu():
    print("[GPU] Forçando carregamento manual das bibliotecas NVIDIA...")
    
    # Pega o caminho base do seu Python (C:\Users\Usuario\...\Python312)
    base_python = sys.prefix 
    
    # Monta os caminhos exatos que você informou
    pastas_alvo = [
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cublas", "bin"),
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
        # Adiciona cudart e nvrtc caso existam (precaução)
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cuda_runtime", "bin"),
        os.path.join(base_python, "Lib", "site-packages", "nvidia", "cuda_nvrtc", "bin")
    ]

    encontrou_alguma = False

    for pasta in pastas_alvo:
        if os.path.exists(pasta):
            try:
                # O COMANDO CRÍTICO: Registra a pasta para o Python procurar DLLs lá
                if hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(pasta)
                
                # Atualiza o PATH também (para subprocessos ou legado)
                os.environ["PATH"] = pasta + os.pathsep + os.environ["PATH"]
                
                print(f"   [OK] DLLs registradas: {pasta}")
                encontrou_alguma = True
            except Exception as e:
                print(f"   [ERRO] Falha ao registrar {pasta}: {e}")
        else:
            # Apenas avisa se não achar (cudart/nvrtc as vezes não estão instalados e ok)
            pass

    if not encontrou_alguma:
        print("\n[CRÍTICO] Nenhuma pasta da NVIDIA foi encontrada dentro do Python.")
        print(f"Verificado dentro de: {base_python}\\Lib\\site-packages\\nvidia\\...")

configurar_ambiente_gpu()

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Erro: O módulo 'faster-whisper' não está instalado.")
    print("Execute: py -m pip install faster-whisper")
    input("\nPressione Enter para sair...")
    sys.exit(1)

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
    print(f"[LLM] Gerando narrativa jurídica detalhada para {nome}...")
    
    system_prompt = (
        "Você é um Assistente Jurídico Sênior especializado em Direito Processual Penal Brasileiro.\n"
        "Sua tarefa é converter a transcrição crua de uma audiência judicial em um termo de depoimento formal.\n\n"
        
        "### DIRETRIZES DE ANÁLISE (Mental)\n"
        "Antes de escrever, analise o texto para identificar:\n"
        "- O contexto do crime (tráfico, roubo, homicídio, etc.) para corrigir termos técnicos.\n"
        "- A distinção entre quem pergunta (Juiz/Promotor/Advogado) e quem responde (Depoente).\n\n"
        
        "### REGRAS DE REDAÇÃO (Obrigatórias)\n"
        "1. **FORMATO:** Texto corrido, EM UM ÚNICO BLOCO (sem quebras de parágrafo), em terceira pessoa.\n"
        f"2. **INÍCIO PADRÃO:** Comece estritamente com: '{nome}, {papel}, ouvido em juízo, disse que...'\n"
        "3. **EXAUSTIVIDADE:** Não faça um resumo. Narre TODOS os fatos mencionados pelo depoente. Se ele descreveu detalhes visuais, horários ou pessoas, tudo deve constar.\n"
        "4. **TRATAMENTO DE PERGUNTAS:**\n"
        "   - Elimine as perguntas diretas.\n"
        "   - Incorpore a resposta na narrativa.\n"
        "   - Se a pergunta introduz um tema novo, use: 'Indagado sobre [tema], respondeu que...'\n"
        "   - As falas de terceiros (Juiz/Promotor) servem apenas para dar contexto; não as transcreva literalmente.\n"
        "5. **CORREÇÃO E CLAREZA:**\n"
        "   - Corrija erros de transcrição (ex: 'tráfego' por 'tráfico', 'meliciano' por 'miliciano') baseando-se no contexto.\n"
        "   - Complete frases truncadas para dar sentido lógico, sem inventar fatos.\n"
        "   - Se algo for impossível de entender, use '[trecho ininteligível]'.\n"
        "6. **PROIBIÇÕES:**\n"
        "   - NÃO coloque data, hora ou número do processo.\n"
        "   - NÃO transcreva juramentos ou qualificações (RG, CPF).\n"
        "   - NÃO use listas ou tópicos.\n"
    )

    user_message = (
        f"Abaixo está a transcrição bruta do depoimento de **{nome}** ({papel}).\n"
        "Gere a narrativa jurídica formal, contínua (bloco único de texto) e detalhada conforme as regras estabelecidas.\n\n"
        "--- INÍCIO DA TRANSCRIÇÃO ---\n"
        f"{transcricao}\n"
        "--- FIM DA TRANSCRIÇÃO ---\n\n"
        "Inicie a redação agora:"
    )
    
    payload = {
        "model": modelo_llm,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        "temperature": 0.2, 
        "max_tokens": -1
    }
    
    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Erro na comunicação com LM Studio: {e}")
        return None

def main():
    try:
        temp_dir, result_dir = criar_pastas()
        extensoes_suportadas = ('.mp4', '.mp3', '.wav', '.m4a', '.webm', '.mkv', '.avi', '.mov')
        
        print(f"Iniciando {MODELO_WHISPER} (RTX 4070 Mode)...")
        # Tenta carregar na GPU, se falhar, avisa mas tenta CPU
        try:
            model = WhisperModel(MODELO_WHISPER, device="cuda", compute_type="float16")
        except Exception as e:
            print(f"\n[ATENÇÃO] Erro ao iniciar GPU: {e}")
            if "cublas" in str(e).lower() or "cudnn" in str(e).lower():
                print("ERRO DE DLL PERSISTENTE: O Python não conseguiu carregar a DLL mesmo nas pastas indicadas.")
                print("Verifique se o arquivo 'cublas64_12.dll' existe realmente dentro da pasta nvidia/cublas/bin")
            print("Tentando rodar na CPU (será lento)...")
            model = WhisperModel(MODELO_WHISPER, device="cpu", compute_type="int8")

        arquivos = [f for f in os.listdir(temp_dir) if f.lower().endswith(extensoes_suportadas)]
        
        if not arquivos:
            print(f"\nNenhum arquivo encontrado em: {temp_dir}")
            print("Padrão aceito: Nome_papel.webm")
        
        for arq in arquivos:
            nome_base, _ = os.path.splitext(arq)
            if "_" not in nome_base:
                print(f"Arquivo ignorado (sem '_'): {arq}")
                continue
                
            nome, papel = nome_base.rsplit("_", 1)
            res = transcrever_audio(os.path.join(temp_dir, arq), model)
            
            txt_tempos = "\n".join([f"{formatar_timestamp(s['start'])} {s['text'].strip()}" for s in res["segments"]])
            with open(os.path.join(result_dir, f"{nome_base}_timestamps.txt"), "w", encoding="utf-8") as f:
                f.write(txt_tempos)

            narrativa = gerar_narrativa(nome, papel, res["text"], URL_LM_STUDIO_PADRAO, MODELO_LM_STUDIO)
            if narrativa:
                with open(os.path.join(result_dir, f"{nome_base}_narrativa.txt"), "w", encoding="utf-8") as f:
                    f.write(narrativa)
                print(f"Processo concluído: {arq}")

    except Exception as e:
        print(f"\nErro Crítico: {e}")
    
    input("\nFim do script. Pressione Enter para fechar...")

if __name__ == "__main__":
    main()