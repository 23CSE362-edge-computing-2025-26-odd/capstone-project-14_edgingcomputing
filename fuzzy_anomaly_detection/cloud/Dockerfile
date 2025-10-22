# Start with a Python base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the local cloud folder into the container's /app folder
COPY . .

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Command to run when the container starts
# This uses streamlit's command to launch the app
CMD ["streamlit", "run", "cloud_app.py"]