# LungLens web demo container: Hugging Face Spaces (Docker SDK) or Render.
# Built by tools/build_hf_space.py - edit the template in deploy/hf_space/.
FROM python:3.11-slim

# Spaces run the container as uid 1000; build as that user so files are writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1
WORKDIR /home/user/app

# CPU-only torch first, from PyTorch's own index: the default PyPI wheel
# bundles CUDA and is ~10x larger for no benefit on a CPU Space.
RUN pip install --no-cache-dir --user \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 torchvision==0.28.0

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Render (and most hosts) pass the port in $PORT; Hugging Face expects 7860.
EXPOSE 7860
CMD exec python -m webapp.server --checkpoint model/model.pt \
     --device cpu --host 0.0.0.0 --port "${PORT:-7860}"
