# WhatsApp Bot LLM — Assistente de Controle de Ar-Condicionado

Bot que reconhece comandos em linguagem natural (ex: "desliga o ar da sala F5") e executa ações reais de controle de ar-condicionado, usando um modelo de embedding para detectar a intenção do usuário e um LLM local (via Ollama) para gerar respostas em linguagem natural.

## Passo a passo para rodar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/WhatsappBot_LLM.git

cd WhatsappBot_LLM
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

> Você vai saber que o ambiente está ativo porque o terminal passa a mostrar `(venv-embedding)` no início da linha.

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

> A primeira instalação pode demorar um pouco (a lib `sentence-transformers` baixa o PyTorch, ~500MB-1GB).

### 5. Execute o projeto

```bash
python embedding_test.py
```

## Desativando o ambiente virtual

Quando terminar de usar:

```bash
deactivate
```
