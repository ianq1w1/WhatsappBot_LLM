from sentence_transformers import SentenceTransformer, util

import psutil
import os

print("Carregando modelo... (primeira vez baixa ~1GB, pode demorar)")
#model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
model = SentenceTransformer('intfloat/multilingual-e5-large')

print("Modelo carregado!\n")

# comandos conhecidos
comandos = {
    "ligar_ar": ["ligar ar", "ligar ar-condicionado", "ativa o ar", "acender o ar", "ligar"],
    "desligar_ar": ["desligar ar", "desliga o ar", "para o ar-condicionado", "desativa o ar"],
    "ajustar_temp": ["mudar temperatura para", "ajustar temperatura para", "coloca a temperatura em"]
}

# pre-computa os embeddings dos exemplos
embeddings_comandos = {
    intencao: model.encode(exemplos, convert_to_tensor=True)
    for intencao, exemplos in comandos.items()
}


processo = psutil.Process(os.getpid())

def log_recursos(etapa=""):
    cpu = processo.cpu_percent(interval=0.5)
    mem = processo.memory_info().rss / (1024 ** 2)  # em MB
    print(f"[{etapa}] CPU: {cpu:.1f}% | RAM: {mem:.1f} MB")


def detectar_intencao(texto_usuario, limiar=0.55):
    log_recursos("antes do embedding")
    emb_usuario = model.encode(texto_usuario, convert_to_tensor=True)
    log_recursos("depois do embedding")
    resultados = {}

    for intencao, emb_exemplos in embeddings_comandos.items():
        score = util.cos_sim(emb_usuario, emb_exemplos).max().item()
        resultados[intencao] = score

    melhor_intencao = max(resultados, key=resultados.get)
    melhor_score = resultados[melhor_intencao]

    print(f"  Scores: {resultados}")

    if melhor_score > limiar:
        return melhor_intencao, melhor_score
    return None, melhor_score

# loop de teste interativo
print("Digite frases pra testar (ou 'sair' pra encerrar)\n")
while True:
    texto = input(">>> ")
    if texto.lower() == "sair":
        break
    intencao, score = detectar_intencao(texto)
    print(f"  Intencao detectada: {intencao} (score: {score:.3f})\n")