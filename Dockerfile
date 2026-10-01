FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY leadbot ./leadbot

ENV PYTHONUNBUFFERED=1 DB_PATH=/data/leads.sqlite3
CMD ["python", "-m", "leadbot.main"]
