FROM python:3.9-slim

# Evită prompturile interactive în timpul instalării
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Actualizează și instalează dependențele de bază
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-opencv \
    libopencv-dev \
    wget \
    unzip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Setează directorul de lucru
WORKDIR /workspace

# Instalează dependențele Python
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Descarcă modelul predictor pentru dlib
RUN mkdir -p models && \
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 && \
    bzip2 -d shape_predictor_68_face_landmarks.dat.bz2 && \
    mv shape_predictor_68_face_landmarks.dat models/

# Creează directoare pentru rezultate și date
RUN mkdir -p logs data recordings reports

CMD ["bash"]