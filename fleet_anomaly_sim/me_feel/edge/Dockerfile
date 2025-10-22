FROM python:3.9-slim
WORKDIR /app

COPY edge_node.py .
COPY requirements.txt .
COPY model.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "edge_node.py"]
