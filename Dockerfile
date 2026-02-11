# 1. Base Image: Start with a lightweight Python Linux setup
FROM python:3.11-slim

# 2. Setup: Create a folder inside the container
WORKDIR /app

# 3. Dependencies: Copy just the requirements first (for caching speed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Application: Copy the code and models into the container
COPY main.py .
COPY energy_model.json .
COPY model_features.json .
COPY anomaly_model.pkl .

# 5. Run: The command to start the API
# We use port 8080 as the default inside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]