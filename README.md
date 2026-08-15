# WhatsApp Bot LLM — Assistente de Controle de Ar-Condicionado

API que detecta a intenção de comandos em linguagem natural (ex: "desliga o ar da sala F5") usando um modelo de embedding, e devolve a intenção detectada, sala e temperatura extraídas da mensagem.

## Como funciona

1. O usuário envia uma mensagem em texto (ex: "desliga o ar da sala F5")
2. Um modelo de embedding (`intfloat-multilingual-e5-large`, rodando via Ollama) converte a frase em um vetor numérico
3. Esse vetor é comparado, por similaridade de cosseno, com vetores pré-computados de comandos conhecidos (ligar, desligar, ajustar temperatura)
4. A API retorna a intenção detectada, o score de confiança, e dados extras extraídos da frase (sala, temperatura)

## Requisitos

- [Python 3.10+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/download) instalado e rodando
- Git (para clonar o repositório)

### Modelos de IA necessários no Ollama

Antes de rodar o projeto, baixe o modelo de embedding usado para detecção de intenção:

```bash
ollama pull intfloat-multilingual-e5-large:f32
```

> Existem outras variantes desse modelo disponíveis no Ollama (`:q8_0`, menor e mais rápida, ou `:f16`, intermediária). O nome usado no projeto deve bater exatamente com o configurado no `.env` (veja a seção abaixo).

Confirme que o modelo foi baixado corretamente:

```bash
ollama list
```

## Passo a passo para rodar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```

### 2. Crie o ambiente virtual (venv)

```bash
python -m venv venv-embedding
```

### 3. Ative o ambiente virtual

**Windows (cmd/PowerShell):**
```bash
venv-embedding\Scripts\activate
```

**Linux/macOS:**
```bash
source venv-embedding/bin/activate
```

> O terminal deve passar a exibir `(venv-embedding)` no início da linha, confirmando que o ambiente está ativo.

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

### 5. Configure o arquivo `.env`

Copie o arquivo de exemplo:

```bash
cp .env.example .env
```

Edite o `.env` com os valores do seu ambiente:

```
OLLAMA_HOST=http://localhost:11434
EMBEDDING_MODEL=intfloat-multilingual-e5-large:f32
```

> Se o Ollama estiver rodando em outra máquina/VPS, troque `OLLAMA_HOST` pelo endereço correto (ex: `http://IP_DA_VPS:11434`).

### 6. Certifique-se de que o Ollama está rodando

O Ollama normalmente já roda em background após a instalação. Para confirmar:

```bash
ollama list
```

Se o comando responder com a lista de modelos instalados, o serviço está no ar.

### 7. Execute a API

```bash
uvicorn API:app --host 0.0.0.0 --port 8003
```

Se tudo estiver certo, o terminal deve mostrar:

```
INFO:     Uvicorn running on http://0.0.0.0:8003
```

## Testando a API

### Via navegador (documentação automática)

Com o servidor rodando, acesse:

```
http://localhost:8003/docs
```

Essa é a interface Swagger gerada automaticamente pelo FastAPI — permite testar o endpoint diretamente pelo navegador, sem precisar de ferramentas externas.

### Via Postman/Insomnia ou `curl`

**Endpoint:** `POST http://localhost:8003/detectar_intencao`

**Body (JSON):**
```json
{
    "texto": "oi, desliga o ar da sala F5"
}
```

**Resposta esperada:**
```json
{
    "intencao": "desligar_ar",
    "score": 0.87,
    "sala": "F5",
    "temperatura": null
}
```

**Exemplo via `curl`:**
```bash
curl -X POST http://localhost:8003/detectar_intencao \
  -H "Content-Type: application/json" \
  -d "{\"texto\": \"desliga o ar da sala F5\"}"
```

## Estrutura do projeto

```
.
├── venv-embedding/       # ambiente virtual (não versionado)
├── .env                  # variáveis de ambiente locais (não versionado)
├── .env.example          # modelo de variáveis de ambiente (versionado)
├── requirements.txt      # dependências do projeto
├── embedding.py          # lógica de detecção de intenção via embedding
├── API.py                # API FastAPI, expõe o endpoint /detectar_intencao
└── README.md
```

## Desativando o ambiente virtual

Quando terminar de usar:

```bash
deactivate
```

