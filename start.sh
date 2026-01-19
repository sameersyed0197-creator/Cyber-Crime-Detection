#!/bin/bash

# Render startup script
echo "🚀 Starting CyberGuard AI on Render..."

# Create data directory if it doesn't exist
mkdir -p /opt/render/project/src/data

# Set environment variable for Render
export RENDER=true

# Initialize database
python -c "from database import init_db; init_db()"

# Start the application
python -m uvicorn main:app --host 0.0.0.0 --port $PORT