# Dockerfile for Hugging Face Spaces (Docker SDK). Also works with any Docker host.
#   docker build -t shopeasy-support .
#   docker run -p 7860:7860 -e GOOGLE_API_KEY=your_key shopeasy-support
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# Hugging Face Spaces run containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# CPU-only PyTorch first (avoids downloading several GB of CUDA libraries),
# then the rest of the pinned requirements.
COPY --chown=user requirements.txt .
RUN pip install --user torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user -r requirements.txt

COPY --chown=user . .

# Download the embedding model and build the FAISS index at build time,
# so the app starts quickly.
RUN python rag.py

EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
