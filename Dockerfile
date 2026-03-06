FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=""

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]