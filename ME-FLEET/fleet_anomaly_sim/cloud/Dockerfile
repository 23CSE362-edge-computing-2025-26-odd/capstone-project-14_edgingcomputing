FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY cloud_app.py ./
CMD ["streamlit", "run", "cloud_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
