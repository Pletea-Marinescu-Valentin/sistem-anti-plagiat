# Anti-Plagiarism Monitoring System 

## Features 

* Core Monitoring Features 
* Real-time candidate activity monitoring using webcam video stream
* Accurate and improved gaze tracking with MediaPipe facial landmark detection with custom eye tracking algorithm
* Multi-method and geometrically validated pupil detection 
* Detection of prohibited objects (smartphone, smartwatch, other electronic gadgets) with YOLOv8
* Violation logging 
* Video recording with violations overlay 
* Live notifications on detected violations 

* Enhanced Detection System 
* 468-landmark MediaPipe Face Mesh detection for high-precision landmark localization (updated from FaceMesh module)
* Eye region isolation based on specific facial landmarks
* Multi-method enhanced pupil localization with shape and size estimation (updated)
* Automatic head rotation and tilt compensation (yaw/pitch control) (updated)
* Filtering based on detection confidence scores (updated) 
* Advanced temporal filtering for smooth, noise-free pupil tracking (updated)
* User-calibrated detection thresholds for individualized user settings 

* Reporting and Analysis 
* Automated violation report generation with screenshots, timestamps and descriptive violation annotations
* Comprehensive export of recorded data in various formats including JSON, CSV and PDF files
* Violation frequency and pattern analysis 
* Standalone utility for batch processing of images to test and analyze gaze detection

## ⚡ Quick Start 

**For advanced users - minimum steps to run:** 

```bash 
# 1. Install system dependencies (Linux) 
sudo apt update && sudo apt install cmake 
sudo apt install -y libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0

# 2. Create conda environment 
conda env create -f environment.yml 
conda activate anti-plagiat 

# 3. Install additional packages 
pip install ultralytics 
pip uninstall opencv-python opencv-contrib-python opencv-python-headless 
pip install opencv-python-headless==4.8.1.78 

# 4. Run application 
python gui_app.py 
``` 

## Requirements 

### System Requirements 

* **OS**: Linux (recommended: Ubuntu/Debian/Kali Linux, other flavors should also work but are not tested)
* **Python**: 3.11+ (recommended: 3.11.5) 
* **Memory**: Minimum 4GB (8GB recommended) 
* **Processor**: Multi-core processor (recommended for faster real-time image processing)
* **Camera**: Webcam or other compatible camera device 
* **Storage**: 2GB free space (for models, recordings) 

### Key Dependencies 

* **conda/miniconda** - conda environment management system 
* **CMake** - Required to compile dlib module 
* **OpenCV** - Open Source Computer Vision Library 
* **MediaPipe** - Face and facial landmarks detection models 
* **PyQt5/PySide6** - Python binding for Qt GUI application framework
* **YOLOv8 (ultralytics)** - State-of-the-art real-time object detection and computer vision
* **PyTorch** - Deep learning framework 

## Installation 

### Prerequisites 

**System Requirements:** 

* **Linux**: The system should run on Linux. Ubuntu, Debian or Kali Linux are recommended (others may work but not tested)
* **Python**: Make sure to have at least 3.11 (recommended 3.11.5) installed on your system.
* **Conda/Miniconda**: Must have conda or miniconda environment management tool installed
* **CMake**: Required to compile dlib module 

### 1. Install System Dependencies 

The following script will install: 

* **CMake** (required for dlib compilation) 
* **Qt/GUI** Linux dependencies 

Run these commands to install: 

```bash 
# Install CMake (required for dlib) 
sudo apt update 
sudo apt install cmake 

# Install Qt/GUI dependencies for Linux 
sudo apt install -y libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0
``` 

### 2. Clone the Repository 

Open a terminal and clone this repository by running:

```bash 
git clone <repository-url> 
cd anti-plagiat 
``` 

### 3. Create Conda Environment 

Run the following commands to create and activate the environment:

```bash 
# Create environment from environment.yml 
conda env create -f environment.yml 

# Activate environment 
conda activate anti-plagiat 
``` 

### 4. Install Additional Dependencies 

```bash 
# Install ultralytics (YOLO) from PyPi 
pip install ultralytics 

# Uninstall original OpenCV packages (required for GUI compatibility)
pip uninstall opencv-python opencv-contrib-python opencv-python-headless 

# Install a specific version of opencv-python-headless without GUI dependencies
pip install opencv-python-headless==4.8.1.78 
``` 

### 5. Run the Application 

```bash 
python gui_app.py 
``` 

### Alternative installation (pip only) 

```bash 
# Create virtual environment 
python -m venv venv 
source venv/bin/activate # On Windows: venv\Scripts\activate 

# Install dependencies 
pip install -r requirements.txt 
pip install ultralytics 

# Run the application 
python gui_app.py 
``` 

## Configuration 

System is configured using config.json: 

```json 
{
    "camera": {
        "source": 0,
        "mirror_image": false,
        "resolution": [
            640,
            480
        ],
        "fps": 30,
        "index": 0
    },
    "detection": {
        "gaze": {
            "left_limit": 0.65,
            "right_limit": 0.35,
            "down_limit": 0.6,
            "smoothing_factor": 0.3
        },
        "object": {
            "enabled": true,
            "confidence_thresholds": {
                "phone": 0.65,
                "smartwatch": 0.65
            },
            "objects_of_interest": [
                "phone",
                "smartwatch"
            ]
        }
    },
    "recording": {
        "save_path": "./recordings",
        "format": "mp4"
    },
    "reporting": {
        "save_path": "./reports",
        "export_formats": [
            "html",
            "csv",
            "json"
        ]
    },
    "snapshots": {
        "save_path": "./snapshots"
    }
}
``` 

### Gaze detection parameters 

* `left_limit` / `right_limit` - Horizontal gaze boundaries (values range from 0.0 to 1.0)
* `down_limit` - Vertical down gaze threshold 
* `smoothing_factor` - Temporal smoothing factor (range from 0.0 to 1.0)

### Object detection parameters 

* `phone_confidence` - Minimum required confidence threshold for phone detection
* `watch_confidence` - Minimum required confidence threshold for smartwatch detection
* `nms_threshold` - Non-maximum suppression threshold 

### Recording parameters 

* `save_path` - Path for saving recorded video files 
* `format` - Recording file format (mp4, avi) 
* `quality` - Recording quality (low, medium, high) 

## Usage 

### Basic Usage 

1. **Run Monitoring** - Open a terminal and run: 

```bash 
python gui_app.py 
``` 

2. **Generate Test Images** (Optional) - Run: 

```bash 
python generate_test_images.py 
``` 

3. **Start Exam** - In GUI, click "Start Monitoring" to begin candidate activity surveillance.
Monitoring system automatically detects and logs violations, issues live violation alerts (shown as red alerts in system window).

### Batch image processing 

Batch processing utility to test gaze detection on static images.

Steps: 
1. Place all test images in input_images/ folder 
2. Run image analysis 

```bash 
python image_gaze_analyzer.py 
``` 

3. Check analyzed images in analyzed_images/ folder and logs/

The image gaze analyzer automatically detects and analyzes all images in input_images/ folder and saves them to analyzed_images/. Check logs/ folder for analysis results. 

### Advanced Usage 

#### Custom Violation Rules 

You can add custom violation detection rules by extending the violation detection logic:

```python 
# Example: Add custom violation detection rule 
system.violation_monitor.add_custom_rule( 
name="extended_downward_gaze", 
condition=lambda data: data['v_ratio'] > 0.7, 
duration_threshold=3.0 
) 
``` 

#### Testing and Analysis Tools 

* **Generate Test Images** - PyQt5-based image capturing tool to generate calibrated test images for gaze detection testing
* **Image Gaze Analyzer** - Batch image processing tool with verbose logging and detailed analysis options
* **Configuration Testing** - Various helper tools to validate and optimize detection thresholds

## Architecture 

### High-level architecture 

``` 
anti-plagiat/ 
├── modules/ 
│ ├── gaze_tracking/ 
│ │ ├── gaze_tracker.py # MediaPipe-based gaze tracking engine 
│ │ ├── eye.py # Eye region detection and analysis 
│ │ ├── pupil.py # Enhanced pupil detection 
│ │ └── pupil_tracker.py # Temporal tracking and filtering 
│ ├── face_detector.py # Face detection integration 
│ ├── object_detector.py # YOLOv8-based object detection 
│ ├── violation_monitor.py # Violation detection logic 
│ ├── video_handler.py # Video processing and recording 
│ └── report_generator.py # Report generation 
├── gui_app.py # Main PyQt5 GUI application 
├── main.py # Core system logic 
├── generate_test_images.py # PyQt5-based test image generator 
├── image_gaze_analyzer.py # Batch image analysis tool 
└── config.json # Configuration file 
``` 

### Algorithm Pipeline 

#### Gaze Tracking Pipeline 

1. Face detection - Implemented using MediaPipe Face Mesh detection with 468 facial landmarks.
2. Eye region isolation - Detects left and right eye regions based on specific eye landmarks.
3. Pupil detection - Improved pupil localization with multi-method validation, including geometric constraints.
4. Gaze Calculation - Calculates horizontal and vertical gaze ratios to determine gaze direction.
5. Temporal Filtering - Applies smoothing and noise reduction for stable gaze tracking.
6. Violation Assessment - Compares gaze ratios against configurable thresholds.

#### Enhanced Object Detection 

* Dual YOLOv8 models - Uses two separate specialized models (phone and watch)
* Custom confidence thresholds - Applies different confidence levels for different objects
* Dimension-based object classification - Additional dimension-based validation for detected objects
* Temporal consistency checks - Multi-frame validation to reduce false positives

#### MediaPipe Integration 

* Face Mesh detection - Provides 468 facial landmarks required for precise eye tracking
* Real-time optimized - Designed to work with live video streams
* Robust to conditions - Performs well under various lighting conditions and head poses
* Silent - Suppresses output for cleaner integration with the main application

## 🔧 Troubleshooting 

### Common Issues and Solutions 

#### 1. CMake Error During Installation 

``` 
CMake is not installed on your system! 
... 
CMake Error: CMake could not find an appropriate module for processing the Information "PROJECT_NAME".
``` 

**Solution:** Install CMake. 

Run these commands: 

```bash 
sudo apt update 
sudo apt install cmake 
cmake --version # Verify it is installed 
``` 

#### 2. Qt Platform Plugin Error 

``` 
Could not load the Qt platform plugin "xcb" in ""
This application failed to start because no Qt platform plugin could be initialized.
Reinstalling the application may fix this problem. 
``` 

**Solution:** Install GUI dependencies or run headless. 

Run these commands: 

```bash 
sudo apt install -y libxcb-xinerama0 libxcb-cursor0 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-shape0
``` 

Alternative way: Run application in headless mode 

```bash 
export QT_QPA_PLATFORM=offscreen 
python gui_app.py 
``` 

#### 3. ModuleNotFoundError: ultralytics 

```pycon 
ModuleNotFoundError: No module named 'ultralytics' 
``` 

**Solution:** Install ultralytics. 

Run these commands: 

```bash 
conda activate anti-plagiat 
pip install ultralytics 
``` 

#### 4. OpenCV GUI related issues 

GUI freezing or improper display 

**Solution:** Use headless OpenCV 

Run these commands: 

```bash 
pip uninstall opencv-python opencv-contrib-python opencv-python-headless 
pip install opencv-python-headless==4.8.1.78 
``` 

#### 5. Conda environment issues 

Errors during environment creation or activation 

**Solution:** Clean conda cache and retry. 

Run these commands: 

```bash 
conda clean --all 
conda env remove -n anti-plagiat # If it already exists
conda env create -f environment.yml 
``` 

#### 6. Camera access not granted 

Camera device not detected or permission denied 

**Solution:** Check camera permissions. 

Run these commands: 

```bash 
sudo usermod -a -G video $USER 
# Logout and log back in again 

# Test camera 
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAIL')"
``` 

#### 7. Low FPS issues (slow performance) 

Application runs slowly 

**Solution:** Multiple options: 

1. Reduce camera resolution (adjust in config.json) 
2. Close other apps 
3. GPU acceleration (if supported) 
4. Check CPU load: `htop` 

#### 8. Export environment to other device 

Run these commands: 

```bash 
conda env export > environment.yml 

# In new device 
conda env create -f environment.yml 
conda activate anti-plagiat 
``` 

## Contributing 

1. Fork the project 
2. Create a feature branch 
3. Make your changes 
4. Add tests for new features or components 
5. Submit a pull request 

### Development Guidelines 

* Follow PEP 8 Python style guide 
* Add docstrings to all functions 
* Include unit tests for any new features 
* Update documentation to match changes 
* Test with both standard PyQt5 GUI and batch image processing

## License 

This project is licensed under the MIT License - see the LICENSE file for details.
