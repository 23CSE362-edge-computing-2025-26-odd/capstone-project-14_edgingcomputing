FROM pytorch/pytorch:2.2.0-cpu
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY edge_node.py model.py ./
CMD ["python", "edge_node.py"]
