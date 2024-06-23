FROM python:3.9-slim-buster
WORKDIR /src/backend

COPY requirements.txt requirements.txt

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch torchvision torchaudio
RUN pip3 install -r requirements.txt

COPY . .

# RUN echo "REMINDER >> Create src/backend/secret.txt && src/backend/backups/current.json" TODO get this to work

CMD ["gunicorn"  , "-b", "0.0.0.0:5055", "wgsi:app"]