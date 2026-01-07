O objejivo desse script é fazer transcrições de audiência e exportar em um formato de texto contínuo e organizado.

1. Os arquivos devem ser nomeadso como Nome do Depoente_papel.extensao
Por exemplo:

temp/
├── João Silva_testemunha.mp4
├── Maria Santos_vitima.mp3
├── Pedro Costa_acusado.wav
└── Ana Oliveira_informante.mp4

2. Para executar a tarefa, usar o comando:
python transcricao_narrativa.py

3. Ou, caso deseje escolher o modelo do whisper:
python transcricao_narrativa.py -w small

4. Se precisar mudar o modelo no futuro:
mudar a linha 16 MODELO_LM_STUDIO = "meta-llama-3.1-8b-instruct"