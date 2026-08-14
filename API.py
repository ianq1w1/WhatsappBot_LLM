from fastapi import FastAPI
from pydantic import BaseModel
from embedding import detectar_intencao, extrair_sala, extrair_temperatura

app = FastAPI()

class Mensagem(BaseModel):
    texto: str

@app.post("/detectar_intencao")
def endpoint_detectar(msg: Mensagem):
    intencao, score = detectar_intencao(msg.texto)
    sala = extrair_sala(msg.texto)
    temperatura = extrair_temperatura(msg.texto)
    return {
        "intencao": intencao,
        "score": score,
        "sala": sala,
        "temperatura": temperatura
    }