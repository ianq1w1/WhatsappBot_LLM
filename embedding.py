import requests
import numpy as np
import re
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST") # ou IP da VPS, se for diferente
MODEL_NAME = os.getenv("EMBEDDING_MODEL")


def gerar_embedding(texto, prefixo="query"):
    resp = requests.post(f"{OLLAMA_HOST}/api/embeddings", json={
        "model": MODEL_NAME,
        "prompt": f"{prefixo}: {texto}"
    })
    resp.raise_for_status()
    return np.array(resp.json()["embedding"])


def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# comandos conhecidos
comandos = {
    "ligar_ar": ["ligar ar", "ligar ar-condicionado", "ativa o ar", "acender o ar", "ligar"],
    "desligar_ar": ["desligar ar", "desliga o ar", "para o ar-condicionado", "desativa o ar"],
    "ajustar_temp": ["mudar temperatura para", "ajustar temperatura para", "coloca a temperatura em"],
    "saudacao": ["oi", "olá", "bom dia", "boa tarde", "boa noite", "e aí", "opa"],
    "despedida": ["tchau", "até mais", "falou", "até logo", "obrigado", "valeu"]
}

# pré-computa os embeddings dos exemplos (roda 1x, quando o módulo é importado)
embeddings_comandos = {
    intencao: [gerar_embedding(ex, "passage") for ex in exemplos]
    for intencao, exemplos in comandos.items()
}


def detectar_intencao(texto_usuario, limiar=0.55):
    emb_usuario = gerar_embedding(texto_usuario, "query")
    resultados = {}

    for intencao, emb_exemplos in embeddings_comandos.items():
        score = max(cosine_sim(emb_usuario, emb) for emb in emb_exemplos)
        resultados[intencao] = score

    melhor_intencao = max(resultados, key=resultados.get)
    melhor_score = resultados[melhor_intencao]

    print(f"  Scores: {resultados}")

    if melhor_score > limiar:
        return melhor_intencao, melhor_score
    return None, melhor_score


def extrair_sala(texto):
    match = re.search(r'\b([A-Za-z]\d{1,2})\b', texto)
    return match.group(1).upper() if match else None


def extrair_temperatura(texto):
    match = re.search(r'(\d{1,2})\s*(?:graus|°c|°)', texto, re.IGNORECASE)
    return int(match.group(1)) if match else None