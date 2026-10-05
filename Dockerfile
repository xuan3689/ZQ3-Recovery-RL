# Reproducible CPU-only environment for ZQ3-Recovery-RL.
#
#   docker build -t zq3-recovery-rl .
#   docker run --rm zq3-recovery-rl pytest -q                     # run the tests
#   docker run --rm zq3-recovery-rl python scripts/train.py --steps 400000 \
#       --rollout 2048 --seed 0 --out runs --run-name ppo_v3 --quiet
#
# The image installs the CPU wheel of PyTorch explicitly; the default PyPI
# wheel is a multi-GB CUDA build that this project never uses.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

WORKDIR /app

# --- dependencies -----------------------------------------------------------
# Copied on their own so the (slow) torch install is cached across code edits.
COPY pyproject.toml requirements.txt README.md ./
RUN pip install --upgrade pip \
 && pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install matplotlib pandas python-docx

# --- project ----------------------------------------------------------------
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY web/ ./web/

# Editable install so `import zq3rl` works without setting PYTHONPATH.
RUN pip install -e . --no-deps

# Sanity check baked into the image: the suite must pass at build time.
RUN pytest -q

# Default: drop into a shell so any script can be run interactively.
CMD ["bash"]
