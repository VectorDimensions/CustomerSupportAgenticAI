# server/Dockerfile — Container image for the support_ticket_env server.
#
# Layer ordering rationale (important for build cache efficiency):
#   1. Base image — changes rarely, so it's the outermost cache layer
#   2. COPY requirements.txt — changes only when dependencies change
#   3. RUN pip install — expensive step, cached as long as requirements.txt is unchanged
#   4. COPY . — changes on every code edit, so it's the innermost layer
#
# This ordering means that editing Python source files (the most common change)
# only invalidates the last COPY layer, not the pip install layer. Without this
# ordering, every code change would trigger a full pip install — very slow.

FROM python:3.11-slim

# Set the working directory inside the container.
# /app is the conventional location for application code in Docker images.
WORKDIR /app

# Copy only the requirements file first so Docker can cache the pip install
# layer independently of the application code.
COPY requirements.txt .

# Install dependencies. --no-cache-dir keeps the image smaller by not storing
# the pip download cache inside the container layer.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server package and the client package (needed for models.py imports).
# We copy the entire project root so relative imports work correctly.
COPY . .

# HF Spaces uses port 7860 by default. The openenv.yaml port field must match.
# We bind to 0.0.0.0 (all interfaces) so the container is reachable from outside.
EXPOSE 7860

# Start the FastAPI server using uvicorn.
# app:app means "the `app` object in the `app` module" — i.e. server/app.py's
# module-level `app = create_app()` instance.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
