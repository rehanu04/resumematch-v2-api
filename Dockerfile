FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["sh","-c","if [ -z "$PORT" ]; then P=8000; else P=$PORT; fi; gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$P"]
