O objejivo desse script é fazer transcrições de audiência e exportar em um formato de texto contínuo e organizado.

Os arquivos para transcrição são colocados na pasta temp com o nome do arquivo seguido o padrão:
NomeDoDepoente_papel
Por exemplo:
temp/
├── João Silva_testemunha.mp4
├── Maria Santos_vitima.mp3
├── Pedro Costa_acusado.wav
└── Ana Oliveira_informante.mp4

Os resultados são exportados para a pasta result em dois arquivos de texto.

As instruções para executar a transcrição e o ajuste do texto são:

# Com modelo Llama
python gerar_narrativa_lmstudio.py -m "meta-llama-3.1-8b-instruct"

# Especificando modelo Qwen (padrão)
python gerar_narrativa_lmstudio.py

# Todas as opções
python transcricao_narrativa.py -w large-v3 -m "qwen2.5-7b-instruct" -u "http://localhost:1234/v1/chat/completions"

No momento essa configuração demora bastante, mas vou trabalhar primeiro para arrumar a precisão e depois a parte do tempo.