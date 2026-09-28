"""
NorBERTo_api.py

Router FastAPI (APIRouter) pra plugar no seu app existente - carrega o modelo
do Hugging Face Hub (ou de uma pasta local) e expoe:
  - POST /norberto/detectar  -> endpoint HTTP
  - detectar_comandos_texto() -> funcao Python, pra chamar direto no seu fluxo
    sem passar por HTTP

Configuracao por variaveis de ambiente:
  NORBERTO_REPO_ID            repo no Hub (ex: "seu-usuario/norberto-lse-ar-condicionado")
  NORBERTO_REVISION           (recomendado) hash de commit do repo, pra "travar" a versao
                              do codigo remoto que sera executado (trust_remote_code)
  NORBERTO_LOCAL_DIR          se definido, carrega dessa pasta em vez do Hub
  HF_TOKEN                    so necessario se o repo for privado
  NORBERTO_LIMIAR_CONFIANCA   piso de confianca (0.0 = desligado)
"""

import os
from functools import lru_cache

import torch
from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

REPO_ID = os.getenv("NORBERTO_REPO_ID", "ianzeraA/NorBERTo-Ar-condicionado")
REVISION = os.getenv("NORBERTO_REVISION")
LOCAL_DIR = os.getenv("NORBERTO_LOCAL_DIR")
HF_TOKEN = os.getenv("HF_TOKEN")
LIMIAR_CONFIANCA = float(os.getenv("NORBERTO_LIMIAR_CONFIANCA", "0.5"))

# copia do ESQUEMA_SLOTS_POR_COMANDO do config.py do projeto de treino -
# mantenha sincronizado se voce mudar os comandos/slots la
ESQUEMA_SLOTS_POR_COMANDO = {
    "ligar_ar": ["sala"],
    "desligar_ar": ["sala"],
    "ajustar_temp": ["sala", "temperatura"],
    "saudacao": [],
    "despedida": [],
}


@lru_cache(maxsize=1)
def carregar_modelo():
    """Carrega modelo + tokenizer UMA vez (cacheado). Chame no startup do app
    pra a primeira requisicao nao pagar o custo de carregamento."""
    fonte = LOCAL_DIR or REPO_ID

    kwargs = {"trust_remote_code": True}
    if not LOCAL_DIR:
        if REVISION:
            kwargs["revision"] = REVISION
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

    tokenizer = AutoTokenizer.from_pretrained(fonte, **kwargs)
    model = AutoModel.from_pretrained(fonte, **kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, tokenizer


def detectar_comandos_texto(texto: str) -> list[dict]:
    """Funcao pra chamar direto do seu codigo. Devolve lista de
    {"comando": str, "slots": {...}}; lista vazia se o texto for vazio."""
    if not texto or not texto.strip():
        return []
    model, tokenizer = carregar_modelo()
    return model.detectar_comandos(
        tokenizer,
        texto,
        limiar_confianca=LIMIAR_CONFIANCA,
        esquema_slots=ESQUEMA_SLOTS_POR_COMANDO,
    )


# ---------------------------------------------------------------------------
# Endpoint HTTP
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/norberto", tags=["norberto"])


class DetectarRequest(BaseModel):
    text: str


class Comando(BaseModel):
    comando: str
    slots: dict[str, str]


class DetectarResponse(BaseModel):
    comandos: list[Comando]


# `def` (nao `async def`) de proposito: a inferencia e CPU/GPU-bound e
# bloqueante, entao o FastAPI roda isso numa thread separada e nao trava o
# event loop enquanto o modelo processa.
@router.post("/detectar", response_model=DetectarResponse)
def detectar(req: DetectarRequest):
    return DetectarResponse(comandos=detectar_comandos_texto(req.text))


@router.get("/health")
def health():
    return {"modelo_carregado": carregar_modelo.cache_info().currsize > 0}