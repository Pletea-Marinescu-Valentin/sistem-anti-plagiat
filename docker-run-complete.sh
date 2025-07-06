#!/bin/bash

echo "=== COMPLETE DOCKER SETUP ==="
echo "Running entire anti-plagiarism system in Docker..."

# Detect system and setup X11 properly
setup_display() {
    echo "Setting up display forwarding..."
    
    # Get actual DISPLAY
    ACTUAL_DISPLAY=$(echo $DISPLAY | sed 's/^://')
    echo "Detected DISPLAY: :$ACTUAL_DISPLAY"
    
    # Setup xhost permissions
    xhost +local:docker
    
    # Export for use
    export DOCKER_DISPLAY=":$ACTUAL_DISPLAY"
    echo "Using DOCKER_DISPLAY: $DOCKER_DISPLAY"
}

# Function to run complete system
run_complete_system() {
    echo "🚀 Starting complete anti-plagiarism system in Docker..."
    
    setup_display
    
    # Run with all necessary mounts and permissions
    docker run -it --rm \
        --name anti-plagiarism-complete \
        --privileged \
        --net=host \
        -v /dev:/dev \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v /run/user/$(id -u)/pulse:/run/user/1000/pulse \
        -v $(pwd)/input_images:/app/input_images \
        -v $(pwd)/analyzed_images:/app/analyzed_images \
        -v $(pwd)/analyzed_images_mediapipe:/app/analyzed_images_mediapipe \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/reports:/app/reports \
        -v $(pwd)/recordings:/app/recordings \
        -v $(pwd)/snapshots:/app/snapshots \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/config.json:/app/config.json \
        -e DISPLAY=$DOCKER_DISPLAY \
        -e QT_QPA_PLATFORM_PLUGIN_PATH="" \
        -e QT_QPA_PLATFORM=xcb \
        -e XDG_RUNTIME_DIR=/tmp/runtime-root \
        -e PULSE_RUNTIME_PATH=/run/user/1000/pulse \
        -e USER=root \
        -e PYTHONPATH=/app \
        anti-plagiat_anti-plagiarism \
        python gui_app.py
}

# Function to run analysis only
run_analysis_only() {
    echo "📊 Running image analysis in Docker..."
    
    docker run -it --rm \
        --name anti-plagiarism-analyzer \
        -v $(pwd)/input_images:/app/input_images \
        -v $(pwd)/analyzed_images_mediapipe:/app/analyzed_images_mediapipe \
        -v $(pwd)/logs:/app/logs \
        -v $(pwd)/config.json:/app/config.json \
        -v $(pwd)/image_gaze_analyzer.py:/app/image_gaze_analyzer.py \
        -e PYTHONPATH=/app \
        anti-plagiat_anti-plagiarism \
        python image_gaze_analyzer.py
}

# Menu
echo ""
echo "Choose what to run in Docker:"
echo "1) Complete GUI system (with camera)"
echo "2) Image analysis only"
echo "3) Exit"
echo ""

read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        run_complete_system
        ;;
    2)
        run_analysis_only
        ;;
    3)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

# Cleanup
echo ""
echo "Cleaning up X11 permissions..."
xhost -local:docker
echo "✅ Complete!"