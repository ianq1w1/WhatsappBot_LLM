from fastapi import FastAPI
from pydantic import BaseModel
from embedding import detectar_intencao, extrair_sala, extrair_temperatura
from generateText import gerar_texto, send_prompt

app = FastAPI()

class Mensagem(BaseModel):
    texto: str

@app.post("/detectar_intencao")
def endpoint_detectar(msg: Mensagem):
    intencao = detectar_intencao(msg.texto)
    sala = extrair_sala(msg.texto)
    temperatura = extrair_temperatura(msg.texto)
    texto = gerar_texto(intencao, sala, temperatura)
    res = send_prompt(texto)
    return res
