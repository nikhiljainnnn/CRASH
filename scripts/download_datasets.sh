#!/bin/bash

# Script to download datasets and pretrained models
# for Crash Detection System

set -e  # Exit on error

echo "========================================="
echo "Crash Detection System - Setup Script"
echo "========================================="

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/{raw,processed,annotations,samples}
mkdir -p models/{detection,tracking,temporal,graph,fusion,risk,severity}
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# Download public datasets
echo ""
echo "Downloading public datasets..."
echo "Note: This may take a while depending on your connection"

# UA-DETRAC
echo "1. UA-DETRAC Dataset..."
if [ ! -d "data/raw/ua-detrac" ]; then
    mkdir -p data/raw/ua-detrac
    echo "   Please download UA-DETRAC from:"
    echo "   https://detrac-db.rit.albany.edu/"
    echo "   Extract to: data/raw/ua-detrac/"
else
    echo "   UA-DETRAC directory exists. Skipping..."
fi

# KITTI
echo "2. KITTI Dataset..."
if [ ! -d "data/raw/kitti" ]; then
    mkdir -p data/raw/kitti
    echo "   Please download KITTI from:"
    echo "   http://www.cvlibs.net/datasets/kitti/"
    echo "   Extract to: data/raw/kitti/"
else
    echo "   KITTI directory exists. Skipping..."
fi

# BDD100K
echo "3. BDD100K Dataset..."
if [ ! -d "data/raw/bdd100k" ]; then
    mkdir -p data/raw/bdd100k
    echo "   Please download BDD100K from:"
    echo "   https://bdd-data.berkeley.edu/"
    echo "   Extract to: data/raw/bdd100k/"
else
    echo "   BDD100K directory exists. Skipping..."
fi

# Download pretrained models
echo ""
echo "Downloading pretrained models..."

# YOLOv8
echo "1. YOLOv8 Detection Model..."
if [ ! -f "models/detection/yolov8s.pt" ]; then
    wget -P models/detection/ \
        https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
else
    echo "   YOLOv8 model exists. Skipping..."
fi

# Sample video for testing
echo ""
echo "Downloading sample video..."
if [ ! -f "data/samples/traffic_sample.mp4" ]; then
    echo "   Please place a sample traffic video in data/samples/"
    echo "   You can use any dashcam or traffic camera footage"
fi

# Download CARLA simulator (optional)
echo ""
echo "CARLA Simulator (for synthetic data generation):"
echo "   Download from: https://github.com/carla-simulator/carla/releases"
echo "   Version: 0.9.14 or later"
echo "   This is optional but recommended for data augmentation"

# Create sample configuration files
echo ""
echo "Creating sample configuration files..."

# Camera calibration sample
cat > configs/camera_calibration.yaml << EOF
# Camera Calibration Configuration
camera_id: "CAM_001"
resolution:
  width: 1920
  height: 1080
focal_length:
  fx: 1000.0
  fy: 1000.0
principal_point:
  cx: 960.0
  cy: 540.0
distortion:
  k1: 0.0
  k2: 0.0
  p1: 0.0
  p2: 0.0
  k3: 0.0
homography:
  # 3x3 homography matrix (image plane to ground plane)
  matrix: [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
  ]
EOF

# Database initialization SQL
mkdir -p deployment/sql
cat > deployment/sql/init.sql << EOF
-- Initialize database for crash detection system

CREATE TABLE IF NOT EXISTS crash_events (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    crash_probability FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    risk_score FLOAT,
    num_vehicles INTEGER,
    location_lat FLOAT,
    location_lon FLOAT,
    severity VARCHAR(20),
    alert_sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS detections (
    id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES crash_events(id),
    timestamp TIMESTAMP NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    vehicle_type VARCHAR(50),
    bbox_x FLOAT,
    bbox_y FLOAT,
    bbox_w FLOAT,
    bbox_h FLOAT,
    confidence FLOAT,
    track_id INTEGER
);

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES crash_events(id),
    timestamp TIMESTAMP NOT NULL,
    alert_type VARCHAR(50),
    message TEXT,
    recipients TEXT[],
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_crash_events_timestamp ON crash_events(timestamp);
CREATE INDEX idx_crash_events_camera ON crash_events(camera_id);
CREATE INDEX idx_detections_event ON detections(event_id);
CREATE INDEX idx_alerts_event ON alerts(event_id);
EOF

# Prometheus configuration
mkdir -p monitoring/prometheus
cat > monitoring/prometheus/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'crash-detection-edge'
    static_configs:
      - targets: ['detection-service:9091']

  - job_name: 'crash-detection-cloud'
    static_configs:
      - targets: ['cloud-predictor:9092']

  - job_name: 'crash-detection-api'
    static_configs:
      - targets: ['api-server:8000']

  - job_name: 'crash-detection-alert'
    static_configs:
      - targets: ['alert-service:9093']
EOF

# Grafana datasources
mkdir -p monitoring/grafana/datasources
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Create .gitignore
echo ""
echo "Creating .gitignore..."
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Data
data/raw/*
data/processed/*
!data/samples/.gitkeep
*.h5
*.hdf5
*.pkl
*.pickle

# Models
models/*.pth
models/*.pt
models/*.onnx
models/*.engine
checkpoints/*.pth

# Logs
logs/
*.log
wandb/

# Results
results/*
!results/.gitkeep

# Environment
.env
.env.local

# OS
.DS_Store
Thumbs.db

# Docker
*.pid
*.seed
*.pid.lock

# Temporary
tmp/
temp/
*.tmp
EOF

# Create placeholder files
touch data/samples/.gitkeep
touch results/.gitkeep
touch logs/.gitkeep

# Create example .env file
cat > .env.example << EOF
# Environment variables for Crash Detection System

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=crash_detection
POSTGRES_USER=crash_user
POSTGRES_PASSWORD=change_me

# Kafka
KAFKA_BROKER=localhost:9092

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API Keys
EMERGENCY_API_KEY=your_emergency_api_key_here
WANDB_API_KEY=your_wandb_api_key_here

# Deployment
DEPLOYMENT_ENV=development
LOG_LEVEL=INFO

# Model paths
DETECTION_MODEL_PATH=models/detection/yolov8_small_tensorrt.engine
TEMPORAL_MODEL_PATH=models/temporal/mstt_ca.pth
GNN_MODEL_PATH=models/graph/st_gnn.pth
EOF

echo ""
echo "========================================="
echo "Setup completed!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Download datasets from the URLs provided above"
echo "2. Copy .env.example to .env and configure"
echo "3. Install dependencies: pip install -r requirements.txt"
echo "4. Train models: python training/train_crash_predictor.py"
echo "5. Run inference: python inference/pipeline.py --video data/samples/your_video.mp4"
echo ""
echo "For Docker deployment:"
echo "  docker-compose up -d"
echo ""
echo "For development:"
echo "  python api/server.py"
echo ""
echo "========================================="
