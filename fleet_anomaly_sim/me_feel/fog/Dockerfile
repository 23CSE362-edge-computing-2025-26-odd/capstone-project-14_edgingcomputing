FROM python:3.9-slim
WORKDIR /app

COPY fog_node.py .
COPY requirements.txt .
COPY model.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "fog_node.py"]
