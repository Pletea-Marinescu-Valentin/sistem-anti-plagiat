# Use Python slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for OpenCV, Qt, MediaPipe and GUI
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Qt dependencies for GUI
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    qt5-qmake \
    qtbase5-dev \
    # X11 and GUI dependencies
    libx11-6 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    libxss1 \
    libgconf-2-4 \
    libxtst6 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libatk1.0-0 \
    libcairo-gobject2 \
    libgtk-3-0 \
    libgdk-pixbuf2.0-0 \
    # Additional X11 support
    x11-apps \
    # Audio support
    pulseaudio \
    # Camera support
    v4l-utils \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Remove OpenCV Qt plugins that conflict
RUN rm -rf /usr/local/lib/python3.11/site-packages/cv2/qt/plugins || true

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p input_images analyzed_images logs reports recordings snapshots models

# Create runtime directory for XDG
RUN mkdir -p /tmp/runtime-root && chmod 700 /tmp/runtime-root

# Set environment variables for GUI
ENV PYTHONPATH=/app
ENV QT_QPA_PLATFORM_PLUGIN_PATH=""
ENV QT_QPA_PLATFORM=xcb
ENV XDG_RUNTIME_DIR=/tmp/runtime-root
ENV DISPLAY=:0

# Expose any ports if needed
EXPOSE 8080

# Default command
CMD ["python", "gui_app.py"]