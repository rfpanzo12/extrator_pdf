# Extrator ENADE Web — FastAPI + PostgreSQL + Railway

Aplicação web para extrair questões objetivas do ENADE a partir de PDF da prova e PDF do gabarito, classificar subáreas com OpenAI e salvar cada execução em banco PostgreSQL.

## Funcionalidades

- Upload do PDF da prova.
- Upload opcional do PDF do gabarito.
- Campo para subáreas separadas por vírgula.
- Opção de extrair figuras/tabelas.
- Opção de limpar questões com OpenAI.
- Opção de classificar subáreas com OpenAI.
- Salvamento automático de execuções e perguntas no banco.
- Tela de histórico e visualização de resultados.

## Rodar localmente

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Acesse: http://127.0.0.1:8000

## Variáveis de ambiente

- `OPENAI_API_KEY`: chave da API da OpenAI.
- `DATABASE_URL`: URL do PostgreSQL. Se não existir, usa SQLite local.
- `APP_NAME`: nome exibido na interface.

## Deploy Railway

1. Crie um repositório no GitHub e envie estes arquivos.
2. No Railway, crie um projeto usando **Deploy from GitHub repo**.
3. Adicione um serviço **PostgreSQL**.
4. No serviço da aplicação, configure:
   - `OPENAI_API_KEY`
   - `DATABASE_URL=${{Postgres.DATABASE_URL}}`
5. Faça o deploy.

