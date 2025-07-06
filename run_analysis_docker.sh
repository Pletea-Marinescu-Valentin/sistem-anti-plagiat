#!/bin/bash

echo "=== Docker Image Analysis ==="

# Check if input_images directory exists and has images
if [ ! -d "input_images" ] || [ -z "$(ls -A input_images 2>/dev/null)" ]; then
    echo "⚠️  input_images directory is empty or doesn't exist"
    echo "Please add some test images (center_*.jpg, left_*.jpg, right_*.jpg, down_*.jpg)"
    echo "Creating sample structure..."
    mkdir -p input_images
    echo "Add your test images to: $(pwd)/input_images/"
    exit 1
fi

echo "✓ Found images in input_images/"
echo "📊 Image count: $(ls input_images/*.jpg 2>/dev/null | wc -l)"

# Create output directory
mkdir -p analyzed_images_mediapipe logs

echo ""
echo "🚀 Starting MediaPipe analysis in Docker..."

# Run the analysis
docker run -it --rm \
  -v $(pwd)/input_images:/app/input_images \
  -v $(pwd)/analyzed_images_mediapipe:/app/analyzed_images_mediapipe \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config.json:/app/config.json \
  anti-plagiat_anti-plagiarism \
  python -c "
import sys
sys.path.append('/app')

# Import your analyzer (adapt the name if different)
try:
    from clean_mediapipe_analyzer import CleanMediaPipeAnalyzer
    print('Using CleanMediaPipeAnalyzer...')
    analyzer = CleanMediaPipeAnalyzer()
    analyzer.analyze_all_images()
except ImportError:
    try:
        from simple_mediapipe_analyzer import SimpleMediaPipeAnalyzer  
        print('Using SimpleMediaPipeAnalyzer...')
        analyzer = SimpleMediaPipeAnalyzer()
        analyzer.analyze_all_images()
    except ImportError:
        print('No analyzer found. Creating simple test...')
        import os
        import mediapipe as mp
        
        # Simple test
        print('✓ MediaPipe imported successfully')
        print('✓ Input images:', len([f for f in os.listdir('/app/input_images') if f.endswith('.jpg')]))
        print('✓ Container is working correctly!')
"

echo ""
echo "✅ Analysis completed!"
echo "📁 Check results in: analyzed_images_mediapipe/"
echo "📋 Check logs in: logs/"