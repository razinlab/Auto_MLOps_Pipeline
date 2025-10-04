FROM python:3.9-slim

WORKDIR /app

# Install git to avoid MLflow warnings
RUN apt-get update && apt-get install -y git && apt-get clean

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models mlruns

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "src/orchestrator.py"]
