FROM python:3.9-slim
WORKDIR /app

COPY cloud_app.py .
COPY requirements.txt .
COPY model.py .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "cloud_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
