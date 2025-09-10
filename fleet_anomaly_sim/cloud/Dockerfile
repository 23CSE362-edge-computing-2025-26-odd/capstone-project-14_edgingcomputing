
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cloud_app.py .
EXPOSE 8501
CMD ["streamlit", "run", "cloud_app.py", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
