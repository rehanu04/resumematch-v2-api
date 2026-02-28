FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway provides PORT at runtime. Bind to it.
CMD ["sh","-c","gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:"]
