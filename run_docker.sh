#!/bin/bash

# Docker setup script for Anti-Plagiarism System

echo "=== Anti-Plagiarism System Docker Setup ==="

# Function to check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed!"
        echo "Please install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi
    echo "✓ Docker found"
}

# Function to check if docker-compose is installed
check_compose() {
    if ! command -v docker-compose &> /dev/null; then
        echo "Error: Docker Compose is not installed!"
        echo "Please install Docker Compose first"
        exit 1
    fi
    echo "✓ Docker Compose found"
}

# Function to setup X11 forwarding for GUI
setup_x11() {
    echo "Setting up X11 forwarding for GUI..."
    xhost +local:docker
    export DISPLAY=${DISPLAY}
    echo "✓ X11 forwarding enabled"
}

# Function to create necessary directories
setup_directories() {
    echo "Creating necessary directories..."
    mkdir -p input_images analyzed_images analyzed_images_mediapipe logs reports recordings snapshots models
    echo "✓ Directories created"
}

# Function to build Docker image
build_image() {
    echo "Building Docker image..."
    docker-compose build
    if [ $? -eq 0 ]; then
        echo "✓ Docker image built successfully"
    else
        echo "✗ Failed to build Docker image"
        exit 1
    fi
}

# Function to run GUI application
run_gui() {
    echo "Detecting system type..."
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Linux detected - using X11 forwarding"
        setup_x11
        docker-compose --profile linux up anti-plagiarism-linux
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS detected - using host display"
        # For macOS, you might need XQuartz
        echo "Make sure XQuartz is running and configured to allow network connections"
        docker-compose up anti-plagiarism
    else
        echo "Windows detected - using default configuration"
        docker-compose up anti-plagiarism
    fi
}

# Function to run analysis only
run_analysis() {
    echo "Starting image analysis..."
    docker-compose --profile analysis up analyzer
}

# Function to stop all containers
stop_containers() {
    echo "Stopping all containers..."
    docker-compose down
    echo "✓ Containers stopped"
}

# Function to clean up
cleanup() {
    echo "Cleaning up Docker resources..."
    docker-compose down --rmi all --volumes
    echo "✓ Cleanup complete"
}

# Main menu
show_menu() {
    echo ""
    echo "Choose an option:"
    echo "1) Setup and build"
    echo "2) Run GUI application"
    echo "3) Run image analysis only"
    echo "4) Stop containers"
    echo "5) Cleanup everything"
    echo "6) Exit"
    echo ""
}

# Main script
main() {
    check_docker
    check_compose
    setup_directories

    while true; do
        show_menu
        read -p "Enter choice [1-6]: " choice
        
        case $choice in
            1)
                build_image
                ;;
            2)
                run_gui
                ;;
            3)
                run_analysis
                ;;
            4)
                stop_containers
                ;;
            5)
                cleanup
                ;;
            6)
                echo "Goodbye!"
                exit 0
                ;;
            *)
                echo "Invalid option. Please try again."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# Run main function
main