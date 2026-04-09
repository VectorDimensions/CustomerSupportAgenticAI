FROM python:3.11-slim

WORKDIR /app

# Copy and install server dependencies
# Build context is the repo root: docker build -f server/Dockerfile .
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repo so env/, server/, support_ticket_env/ are all available
COPY . .

# Make sure the repo root is on PYTHONPATH so `import env.registry` works
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
