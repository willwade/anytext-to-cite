FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install llm CLI tool
RUN pip install --no-cache-dir llm

COPY . .

# Create necessary directories if they don't exist
RUN mkdir -p static templates

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "convert:app", "--host", "0.0.0.0", "--port", "8000"]
