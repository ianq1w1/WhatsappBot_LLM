from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from embedding import detectar_intencao, extrair_sala, extrair_temperatura
from uteis import NorBERTo_api 
from generateText import gerar_texto, send_prompt


@asynccontextmanager
async def lifespan(app: FastAPI):
    NorBERTo_api.carregar_modelo()   # baixa/carrega o modelo uma vez, no startup
    yield

app = FastAPI(lifespan=lifespan)   # se voce ja tem um lifespan, so adicione a linha acima nele
app.include_router(NorBERTo_api.router)


class Mensagem(BaseModel):
    texto: str

#endpoint do modelo intfloat(envia a response dele pro qwen)
@app.post("/detectar_intencao")
def endpoint_detectar(msg: Mensagem):
    intencao = detectar_intencao(msg.texto)
    sala = extrair_sala(msg.texto)
    temperatura = extrair_temperatura(msg.texto)
    texto = gerar_texto(intencao, sala, temperatura)
    res = send_prompt(texto)
    return res


#endpoint para teste do modelo intfloat 
@app.post("/embedding")
def endpoint_detectar(msg: Mensagem):
    intencao = detectar_intencao(msg.texto)
    sala = extrair_sala(msg.texto)
    temperatura = extrair_temperatura(msg.texto)
    return {
        "intencao": intencao,
        "sala": sala,
        "temperatura": temperatura
    }

#endpoint para o norberto
@app.post("/norberto_embedding")
def endpoint_norberto(msg:Mensagem):
    res = NorBERTo_api.detectar_comandos_texto(msg.texto)
    return res 