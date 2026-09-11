FROM python:3.13-slim

WORKDIR /app

# Dependencias primeiro (aproveita o cache de build)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Aplicacao (model.pkl ja treinado - nao e necessario o CSV no container)
COPY app.py index.html model.pkl train_model.py test_data.txt README.md ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]