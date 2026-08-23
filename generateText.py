import requests
import numpy as np
import re
import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST") # ou IP da VPS, se for diferente
MODEL_NAME = os.getenv("GENERATIVE_MODEL")


def gerar_texto(intencao, sala=None, temperatura=None):
    # 1. monta a string única a partir dos argumentos separados
    partes = [f"Ação: {intencao}"]
    if sala:
        partes.append(f"Sala: {sala}")
    if temperatura:
        partes.append(f"Temperatura: {temperatura}°C")

    dados_formatados = " | ".join(partes)
    return dados_formatados


def send_prompt(data):
    prompt = f"Gere uma frase curta confirmando esta ação em português: {data}"

    # 2. manda essa string única pro Qwen
    resp = requests.post(f"{OLLAMA_HOST}/api/generate", json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    })
    resp.raise_for_status()
    return resp.json()["response"].strip()
