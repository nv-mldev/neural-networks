#!/bin/bash

# Script to render and merge Perceptron Convergence videos
# This script renders both animations and combines them into a single video

echo "================================="
echo "Perceptron Convergence Video Generator"
echo "================================="
echo ""

# Set quality (you can change this: -ql, -qm, -qh, -qk for low, medium, high, 4k)
QUALITY="-qh"

# Output directory for manim (default is media/videos/)
SCRIPT_FILE="src/perceptron_convergence_geometry.py"

echo "Step 1: Rendering 2D Bounds Visualization..."
manim ${QUALITY} ${SCRIPT_FILE} PerceptronConvergence2D

if [ $? -ne 0 ]; then
    echo "Error: Failed to render 2D animation"
    exit 1
fi

echo ""
echo "Step 2: Rendering 3D Geometric Interpretation..."
manim ${QUALITY} ${SCRIPT_FILE} PerceptronConvergenceGeometry

if [ $? -ne 0 ]; then
    echo "Error: Failed to render 3D animation"
    exit 1
fi

echo ""
echo "Step 3: Locating rendered videos..."

# Find the rendered videos (manim outputs to media/videos/script_name/quality/)
VIDEO_DIR="media/videos/perceptron_convergence_geometry"

# Find quality subdirectory
if [ -d "${VIDEO_DIR}/1080p60" ]; then
    QUALITY_DIR="1080p60"
elif [ -d "${VIDEO_DIR}/720p30" ]; then
    QUALITY_DIR="720p30"
elif [ -d "${VIDEO_DIR}/480p15" ]; then
    QUALITY_DIR="480p15"
elif [ -d "${VIDEO_DIR}/2160p60" ]; then
    QUALITY_DIR="2160p60"
else
    echo "Error: Could not find rendered videos"
    exit 1
fi

VIDEO1="${VIDEO_DIR}/${QUALITY_DIR}/PerceptronConvergence2D.mp4"
VIDEO2="${VIDEO_DIR}/${QUALITY_DIR}/PerceptronConvergenceGeometry.mp4"

# Check if videos exist
if [ ! -f "$VIDEO1" ]; then
    echo "Error: Could not find $VIDEO1"
    exit 1
fi

if [ ! -f "$VIDEO2" ]; then
    echo "Error: Could not find $VIDEO2"
    exit 1
fi

echo "Found video 1: $VIDEO1"
echo "Found video 2: $VIDEO2"

echo ""
echo "Step 4: Creating file list for ffmpeg..."

# Create a temporary file list for ffmpeg
FILELIST="video_list.txt"
echo "file '../../${VIDEO1}'" > ${FILELIST}
echo "file '../../${VIDEO2}'" >> ${FILELIST}

echo ""
echo "Step 5: Merging videos with ffmpeg..."

OUTPUT_FILE="perceptron_convergence_complete.mp4"

# Merge videos using ffmpeg
ffmpeg -f concat -safe 0 -i ${FILELIST} -c copy ${OUTPUT_FILE} -y

if [ $? -ne 0 ]; then
    echo "Warning: Direct copy failed, trying with re-encoding..."
    # If direct copy fails, re-encode
    ffmpeg -f concat -safe 0 -i ${FILELIST} -c:v libx264 -preset medium -crf 23 -c:a aac ${OUTPUT_FILE} -y
fi

# Clean up
rm ${FILELIST}

if [ -f "${OUTPUT_FILE}" ]; then
    echo ""
    echo "================================="
    echo "Success! Combined video created:"
    echo "${OUTPUT_FILE}"
    echo "================================="
    echo ""
    echo "Video details:"
    ffprobe -v error -show_entries format=duration,size,bit_rate -of default=noprint_wrappers=1 ${OUTPUT_FILE}
else
    echo "Error: Failed to create merged video"
    exit 1
fi
