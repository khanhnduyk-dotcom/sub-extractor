FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py index.html ./

RUN mkdir -p temp_uploads

EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--ws-max-size", "524288000", "--timeout-keep-alive", "300"]
