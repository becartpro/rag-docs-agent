FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag_agent ./rag_agent
COPY docs ./docs
ENV DOCS_DIR=docs
EXPOSE 8000
CMD ["uvicorn", "rag_agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
