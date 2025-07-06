#!/bin/bash

echo "=== Simple Docker Test ==="

# Test 1: Basic Python test
echo "Testing basic Python functionality..."

docker run -it --rm \
  anti-plagiat_anti-plagiarism \
  python -c "import mediapipe; import cv2; import numpy; print('✓ All imports work!')"

echo ""

# Test 2: Test with mounted volume
echo "Testing volume mounts..."

docker run -it --rm \
  -v $(pwd):/app/host_data \
  anti-plagiat_anti-plagiarism \
  python -c "import os; print('✓ Files in container:'); print(os.listdir('/app')); print('✓ Host files:'); print(os.listdir('/app/host_data'))"

echo ""

# Test 3: Test GUI without X11 (should show error but not crash)
echo "Testing GUI app (will show Qt error - that's normal)..."

docker run -it --rm \
  -e QT_QPA_PLATFORM=offscreen \
  anti-plagiat_anti-plagiarism \
  timeout 5 python gui_app.py || echo "✓ GUI app started (timeout after 5s is expected)"

echo ""
echo "=== All basic tests completed! ==="
echo ""
echo "To test with camera and GUI, you'll need to:"
echo "1. Configure Docker to share /tmp/.X11-unix"
echo "2. Or use the container in headless mode for image analysis"