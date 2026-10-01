# WhatsApp Bot LLM — Assistente de Controle de Ar-Condicionado

API para entendimento e processamento de comandos de controle de ar-condicionado enviados em linguagem natural.

O projeto combina **NorBERTo fine-tuned para token classification**, um modelo de **embedding via Ollama** e um **LLM Qwen via Ollama** para transformar mensagens do usuário em comandos estruturados e, posteriormente, gerar uma resposta textual.

## Visão geral

O fluxo principal do sistema é:

```text
Usuário
   │
   ▼
Mensagem via WhatsApp
   │
   ▼
FastAPI
   │
   ▼
NorBERTo Fine-Tuned
   │
   ├── Detecta múltiplas ações
   │      ├── ligar_ar
   │      ├── desligar_ar
   │      └── ajustar_temperatura
   │
   └── Detecta múltiplos slots
          ├── sala
          └── temperatura
   │
   ▼
Comandos estruturados
   │
   ▼
Qwen via Ollama
   │
   ▼
Resposta em linguagem natural
```

O **NorBERTo** é responsável pelo entendimento estruturado da mensagem. O **Qwen**, executado através do Ollama, é utilizado posteriormente para interpretar esse resultado e gerar a resposta textual para o usuário.

---

# Como funciona

O sistema possui diferentes componentes de NLP, cada um com uma responsabilidade específica.

## 1. NorBERTo Fine-Tuned — entendimento do comando

O projeto utiliza uma versão fine-tuned do **NorBERTo-large**, treinada especificamente para o domínio de comandos de ar-condicionado.

Modelo:

**NorBERTo-Ar-condicionado**

https://huggingface.co/ianzeraA/NorBERTo-Ar-condicionado

O modelo utiliza **token classification** com duas heads de classificação:

```text
                    NorBERTo
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      Action Tagging        Slot Tagging
             │                   │
             ▼                   ▼
        Ações/Comandos        Campos
```

### Action Tagging

A primeira head identifica ações/comandos utilizando BIO tagging.

Exemplos:

```text
B-ligar_ar
I-ligar_ar

B-desligar_ar
I-desligar_ar

B-ajustar_temperatura
I-ajustar_temperatura
```

Isso permite identificar ações dentro de uma mensagem mesmo quando existem **vários comandos no mesmo texto**.

Por exemplo:

```text
"liga o ar da sala F5 e desliga o da sala B15"
```

pode resultar em múltiplas ações:

```json
[
    {
        "comando": "ligar_ar",
        "slots": {
            "sala": "F5"
        }
    },
    {
        "comando": "desligar_ar",
        "slots": {
            "sala": "B15"
        }
    }
]
```

O modelo não depende exclusivamente de palavras como `"e"` ou vírgulas para identificar que existem múltiplas ações.

---

## 2. Slot Tagging

A segunda head identifica os **slots**, ou seja, os dados associados ao comando.

Exemplos de slots:

```text
sala
temperatura
```

Utilizando BIO tagging:

```text
B-sala
I-sala

B-temperatura
I-temperatura
```

Por exemplo:

```text
"ajusta o ar da sala F5 para 22 graus"
```

pode ser interpretado como:

```json
{
    "comando": "ajustar_temperatura",
    "slots": {
        "sala": "F5",
        "temperatura": "22"
    }
}
```

---

## 3. Múltiplas ações

Uma das principais características do modelo é a capacidade de identificar **mais de uma ação dentro da mesma mensagem**.

Exemplo:

```text
"liga o ar da F5, desliga o da B15 e coloca o A10 em 22 graus"
```

O modelo pode identificar:

```json
[
    {
        "comando": "ligar_ar",
        "slots": {
            "sala": "F5"
        }
    },
    {
        "comando": "desligar_ar",
        "slots": {
            "sala": "B15"
        }
    },
    {
        "comando": "ajustar_temperatura",
        "slots": {
            "sala": "A10",
            "temperatura": "22"
        }
    }
]
```

O pós-processamento associa os slots às ações correspondentes.

---

## 4. Múltiplos valores de slots

O sistema também possui suporte para múltiplos valores do mesmo tipo de slot.

Por exemplo:

```text
"liga o ar das salas B15, A18 e F20"
```

O processamento pode transformar a informação em comandos separados:

```json
[
    {
        "comando": "ligar_ar",
        "slots": {
            "sala": "B15"
        }
    },
    {
        "comando": "ligar_ar",
        "slots": {
            "sala": "A18"
        }
    },
    {
        "comando": "ligar_ar",
        "slots": {
            "sala": "F20"
        }
    }
]
```

---

# 5. Qwen via Ollama

Depois que o NorBERTo identifica as ações e os slots, o resultado estruturado é utilizado no fluxo de geração de texto.

O projeto utiliza o **Ollama** para executar o Qwen localmente ou em um servidor configurado.

Fluxo:

```text
Mensagem
    │
    ▼
NorBERTo
    │
    ▼
Ações + Slots
    │
    ▼
Construção do prompt
    │
    ▼
Qwen via Ollama
    │
    ▼
Resposta textual
```

Por exemplo, o usuário pode enviar:

```text
"oi, liga o ar da sala F5"
```

O NorBERTo identifica:

```json
{
    "comando": "ligar_ar",
    "slots": {
        "sala": "F5"
    }
}
```

Esse resultado é utilizado pelo projeto para construir o contexto enviado ao Qwen.

O Qwen então fica responsável pela **geração da resposta em linguagem natural**, enquanto o NorBERTo fica responsável pela **extração estruturada do comando**.

Essa separação permite que cada modelo tenha uma responsabilidade específica:

| Componente        | Responsabilidade                     |
| ----------------- | ------------------------------------ |
| NorBERTo          | Entendimento estruturado da mensagem |
| Action Head       | Identificação das ações              |
| Slot Head         | Identificação dos campos             |
| Pós-processamento | Associação entre ações e slots       |
| Qwen              | Geração da resposta textual          |
| Ollama            | Execução local/remota dos modelos    |

---

# 6. Embedding com intfloat-multilingual-e5-large

O projeto também mantém um fluxo baseado no modelo:

```text
intfloat-multilingual-e5-large
```

executado através do Ollama.

Esse modelo pode ser utilizado pelo endpoint `/embedding` para o fluxo de detecção baseado em embeddings.

Nesse fluxo:

```text
Mensagem
   │
   ▼
intfloat-multilingual-e5-large
   │
   ▼
Embedding
   │
   ▼
Comparação por similaridade
   │
   ▼
Intenção
```

Esse mecanismo é mantido no projeto para testes e comparação com o fluxo baseado no NorBERTo.

O **fluxo principal baseado no NorBERTo não depende da classificação por similaridade de embeddings**.

---

# Requisitos

* [Python 3.10+](https://www.python.org/downloads/)
* [Ollama](https://ollama.com/download) instalado e rodando
* Git
* Acesso à internet na primeira execução para baixar o modelo NorBERTo do Hugging Face

---

# Modelos utilizados

## NorBERTo Fine-Tuned

Modelo utilizado para token classification e entendimento dos comandos:

```text
ianzeraA/NorBERTo-Ar-condicionado
```

Hugging Face:

https://huggingface.co/ianzeraA/NorBERTo-Ar-condicionado

O modelo é baseado no **NorBERTo-large** e possui duas heads de classificação:

```text
NorBERTo Encoder
       │
       ├── Action Tagging Head
       │
       └── Slot Tagging Head
```

O modelo utiliza código customizado do Hugging Face Hub e deve ser carregado com suporte a `trust_remote_code`.

---

## Embedding

Para o fluxo baseado em embeddings:

```text
jeffh/intfloat-multilingual-e5-large:f32
```

Baixe através do Ollama:

```bash
ollama pull jeffh/intfloat-multilingual-e5-large:f32
```

Existem outras variantes disponíveis, como `:q8_0` e `:f16`.

O nome utilizado no projeto deve corresponder ao configurado no `.env`.

---

## Qwen

O projeto utiliza um modelo Qwen através do Ollama para geração da resposta final.

O modelo específico utilizado pelo projeto deve estar disponível no Ollama e configurado conforme a implementação de `generateText.py`.

---

# Passo a passo para rodar o projeto

## 1. Clone o repositório

```bash
git clone https://ghttps://github.com/ianq1w1/WhatsappBot_LLM.git
cd WhatsappBot_LLM
```

---

## 2. Crie o ambiente virtual

```bash
python -m venv venv-embedding
```

---

## 3. Ative o ambiente virtual

### Windows

```bash
venv-embedding\Scripts\activate
```

### Linux/macOS

```bash
source venv-embedding/bin/activate
```

O terminal deve passar a exibir:

```text
(venv-embedding)
```

---

## 4. Instale as dependências

```bash
pip install -r requirements.txt
```

---

# 5. Configure o `.env`

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

No Windows, caso `cp` não esteja disponível, copie o arquivo manualmente.

Exemplo:

```env
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=intfloat-multilingual-e5-large:f32
```

Se o Ollama estiver rodando em outra máquina ou VPS:

```env
OLLAMA_HOST=http://IP_DA_VPS:11434
```

---

# 6. Instale os modelos do Ollama

Para o embedding:

```bash
ollama pull jeffh/intfloat-multilingual-e5-large:f32
```

Para o Qwen, instale o modelo utilizado pelo projeto, por exemplo:

```bash
ollama pull <modelo-qwen>
```

Confirme os modelos instalados:

```bash
ollama list
```

---

# 7. NorBERTo

O NorBERTo fine-tuned é carregado diretamente do Hugging Face:

```text
ianzeraA/NorBERTo-Ar-condicionado
```

Na inicialização da API, o modelo é carregado uma vez através do `lifespan` do FastAPI.

Conceitualmente:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):

    NorBERTo_api.carregar_modelo()

    yield
```

Dessa forma, o modelo não precisa ser carregado novamente a cada requisição.

---

# 8. Execute a API

```bash
uvicorn API:app --host 0.0.0.0 --port 8003
```

Se tudo estiver correto:

```text
INFO:     Uvicorn running on http://0.0.0.0:8003
```

---

# Testando a API

## Documentação automática

Com o servidor rodando:

```text
http://localhost:8003/docs
```

A interface Swagger do FastAPI permite testar os endpoints diretamente pelo navegador.

---

# Endpoint principal — NorBERTo + Qwen

```text
POST /detectar_intencao
```

Exemplo:

```json
{
    "texto": "oi, desliga o ar da sala F5"
}
```

Nesse endpoint, o fluxo é:

```text
texto
  │
  ▼
NorBERTo
  │
  ▼
comando + slots
  │
  ▼
gerar_texto()
  │
  ▼
Qwen / Ollama
  │
  ▼
resposta
```

O endpoint utiliza o resultado do NorBERTo para montar o contexto enviado ao Qwen.

---

# Endpoint de teste do embedding

```text
POST /embedding
```

Exemplo:

```json
{
    "texto": "desliga o ar da sala F5"
}
```

Esse endpoint utiliza o fluxo baseado no `intfloat-multilingual-e5-large`.

Exemplo de resposta:

```json
{
    "intencao": "desligar_ar",
    "sala": "F5",
    "temperatura": null
}
```

---

# Endpoint direto do NorBERTo

```text
POST /norberto_embedding
```

Exemplo:

```json
{
    "texto": "liga o ar da F5 e coloca o da B15 em 22 graus"
}
```

O endpoint retorna diretamente o resultado estruturado pelo NorBERTo, sem passar pela etapa de geração do Qwen.

Exemplo conceitual:

```json
[
    {
        "comando": "ligar_ar",
        "slots": {
            "sala": "F5"
        }
    },
    {
        "comando": "ajustar_temperatura",
        "slots": {
            "sala": "B15",
            "temperatura": "22"
        }
    }
]
```

---

# Exemplo via curl

## NorBERTo + Qwen

```bash
curl -X POST http://localhost:8003/detectar_intencao \
  -H "Content-Type: application/json" \
  -d "{\"texto\": \"oi, desliga o ar da sala F5\"}"
```

## NorBERTo

```bash
curl -X POST http://localhost:8003/norberto_embedding \
  -H "Content-Type: application/json" \
  -d "{\"texto\": \"liga o ar da F5 e coloca o da B15 em 22 graus\"}"
```

## Embedding

```bash
curl -X POST http://localhost:8003/embedding \
  -H "Content-Type: application/json" \
  -d "{\"texto\": \"desliga o ar da sala F5\"}"
```

---


---

# Arquitetura do sistema

O projeto atualmente possui dois fluxos relacionados ao entendimento de mensagens.

### Fluxo principal

```text
                    ┌───────────────────┐
                    │ Mensagem WhatsApp │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ NorBERTo Fine-    │
                    │ Tuned             │
                    └─────────┬─────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
             Action Tagging       Slot Tagging
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ Comandos + Slots  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Prompt / Contexto │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Qwen via Ollama   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Resposta ao       │
                    │ usuário           │
                    └───────────────────┘
```

### Fluxo alternativo de embedding

```text
Mensagem
   │
   ▼
intfloat-multilingual-e5-large
   │
   ▼
Embedding
   │
   ▼
Similaridade
   │
   ▼
Intenção + dados extraídos
```

---

# Por que utilizar os dois modelos?

Cada modelo possui uma função diferente dentro da aplicação.

### NorBERTo

Responsável por:

* identificar ações;
* identificar múltiplas ações;
* identificar slots;
* identificar múltiplos valores de slots;
* associar slots às respectivas ações;
* produzir uma estrutura de comandos.

### Qwen

Responsável por:

* receber o contexto estruturado;
* interpretar o resultado;
* gerar uma resposta em linguagem natural;
* conversar com o usuário.

### Embedding

O `intfloat-multilingual-e5-large` permanece disponível para o fluxo de detecção baseado em embeddings e para testes/comparações dentro da API.

---

# Desativando o ambiente virtual

Quando terminar de utilizar o projeto:

```bash
deactivate
```

---

# Modelo NorBERTo

O modelo fine-tuned utilizado pelo projeto está disponível no Hugging Face:

https://huggingface.co/ianzeraA/NorBERTo-Ar-condicionado

O modelo foi desenvolvido especificamente para o domínio de comandos de ar-condicionado e utiliza duas heads de token classification sobre um encoder NorBERTo.

---

