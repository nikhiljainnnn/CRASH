"""
FastAPI Server for Crash Detection System
Provides REST API endpoints for inference and monitoring
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import cv2
import numpy as np
import base64
import torch
import yaml
from typing import Optional, List
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import Response, StreamingResponse

from models.temporal.mstt_transformer import MSTT_CA
from scripts.train_st_gnn import SimpleSTGNN
from inference.pipeline import CrashDetectionPipeline

# Initialize FastAPI
app = FastAPI(
    title="Crash Detection API",
    description="Real-time crash detection and prediction API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
PREDICTION_COUNT = Counter('crash_api_predictions', 'Total number of predictions')
CRASH_ALERTS = Counter('crash_api_alerts', 'Total number of crash alerts')
INFERENCE_LATENCY = Histogram('crash_api_inference_latency', 'Inference latency')
CRASH_PROBABILITY = Gauge('crash_api_probability', 'Current crash probability')
CRASH_UNCERTAINTY = Gauge('crash_api_uncertainty', 'Current crash mathematical uncertainty')

# Load configuration
with open('configs/inference_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = CrashDetectionPipeline('configs/inference_config.yaml')


# Request/Response Models
class PredictionRequest(BaseModel):
    camera_id: str
    frame_base64: str
    timestamp: Optional[float] = None


class PredictionResponse(BaseModel):
    camera_id: str
    timestamp: float
    crash_probability: float
    risk_level: str
    risk_score: float
    uncertainty: Optional[float]
    time_to_collision: Optional[float]
    num_vehicles: int
    detections: List[dict]
    latency_ms: float
    alert: Optional[dict]


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_predictions: int
    total_alerts: int
    avg_latency_ms: float
    avg_crash_probability: float
    avg_uncertainty: float


# Global state
start_time = time.time()


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Crash Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "video_feed": "/video_feed",
            "health": "/health",
            "metrics": "/metrics",
            "prometheus": "/prometheus"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=str(pipeline.device),
        uptime_seconds=time.time() - start_time
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict crash probability from a single frame
    
    Args:
        request: Prediction request with camera_id and base64-encoded frame
    
    Returns:
        Prediction results including crash probability and risk level
    """
    try:
        start = time.time()
        
        # Decode frame
        frame_bytes = base64.b64decode(request.frame_base64)
        frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        # Process frame
        with INFERENCE_LATENCY.time():
            result = pipeline.process_frame(frame)
        
        # Update metrics
        PREDICTION_COUNT.inc()
        CRASH_PROBABILITY.set(result['crash_probability'])
        CRASH_UNCERTAINTY.set(result.get('uncertainty', 0.0))
        
        if result.get('alert'):
            CRASH_ALERTS.inc()
        
        # Prepare response
        response = PredictionResponse(
            camera_id=request.camera_id,
            timestamp=request.timestamp or time.time(),
            crash_probability=result['crash_probability'],
            risk_level=result['risk_level'],
            risk_score=result.get('risk_score', 0.0),
            uncertainty=result.get('uncertainty', 0.0),
            time_to_collision=result.get('time_to_collision', None),
            num_vehicles=len(result['detections']),
            detections=[
                {
                    'bbox': det['bbox'].tolist() if hasattr(det['bbox'], 'tolist') else det['bbox'],
                    'confidence': float(det['confidence']),
                    'class': det['class']
                }
                for det in result['detections']
            ],
            latency_ms=result['latency']['total'],
            alert=result.get('alert')
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch prediction on multiple frames
    
    Args:
        files: List of uploaded image files
    
    Returns:
        List of prediction results
    """
    results = []
    
    for file in files:
        try:
            # Read file
            contents = await file.read()
            frame_array = np.frombuffer(contents, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Process
            result = pipeline.process_frame(frame)
            
            results.append({
                'filename': file.filename,
                'crash_probability': result['crash_probability'],
                'risk_level': result['risk_level'],
                'num_vehicles': len(result['detections'])
            })
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {"results": results, "total": len(results)}


async def generate_frames(camera_id: str = "CAM_01"):
    """
    Generator function that yields JPEG frames from a video source.
    Currently streams a default video or camera, running it through the pipeline.
    """
    # Open default camera or a sample video file
    cap = cv2.VideoCapture(0) # Change to video file path if needed
    
    # Try to set MJPG format for webcams if possible
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    while True:
        success, frame = cap.read()
        if not success:
            # If video ends, loop or break. Here we just wait and retry for webcams.
            time.sleep(0.1)
            continue
            
        # Optional: run through pipeline to get bounding boxes
        # This will slow down the stream depending on GPU/CPU 
        try:
            result = pipeline.process_frame(frame)
            # Update metrics automatically
            CRASH_PROBABILITY.set(result['crash_probability'])
            PREDICTION_COUNT.inc()
            CRASH_UNCERTAINTY.set(result.get('uncertainty', 0.0))
            if result.get('alert'):
                CRASH_ALERTS.inc()
                
            # Draw detections (basic implementation)
            for det in result['detections']:
                bbox = det['bbox']
                # Basic drawing - convert tensor/array back to int coords
                if hasattr(bbox, 'cpu'):
                    bbox = bbox.cpu().numpy()
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                           
            # Draw risk level and uncertainty
            uncertainty_text = f" (Unc: {result.get('uncertainty', 0.0):.2f})" if 'uncertainty' in result else ""
            cv2.putText(frame, f"Risk: {result['risk_level']} ({result['crash_probability']:.2%}){uncertainty_text}", 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                       (0, 0, 255) if result['crash_probability'] > 0.7 else (0, 255, 0), 2)
        except Exception as e:
            print(f"Pipeline error in stream: {e}")
            pass

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
               
        # Control framerate
        time.sleep(0.05) 


@app.get("/video_feed")
async def video_feed(camera_id: str = "CAM_01"):
    """
    MJPEG streaming endpoint for live video feed
    """
    return StreamingResponse(
        generate_frames(camera_id), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get aggregated metrics"""
    return MetricsResponse(
        total_predictions=int(PREDICTION_COUNT._value.get()),
        total_alerts=int(CRASH_ALERTS._value.get()),
        avg_latency_ms=np.mean(pipeline.latency_tracker['total']) if pipeline.latency_tracker['total'] else 0.0,
        avg_crash_probability=CRASH_PROBABILITY._value.get(),
        avg_uncertainty=CRASH_UNCERTAINTY._value.get()
    )


@app.get("/prometheus")
async def prometheus_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/stats")
async def get_statistics():
    """Get detailed pipeline statistics"""
    stats = {}
    
    for key in ['detection', 'tracking', 'cloud', 'total']:
        if pipeline.latency_tracker[key]:
            times = pipeline.latency_tracker[key]
            stats[key] = {
                'mean_ms': float(np.mean(times)),
                'median_ms': float(np.median(times)),
                'p95_ms': float(np.percentile(times, 95)),
                'p99_ms': float(np.percentile(times, 99)),
                'min_ms': float(np.min(times)),
                'max_ms': float(np.max(times))
            }
    
    stats['alerts'] = {
        'total': len(pipeline.alert_history),
        'by_level': {}
    }
    
    if pipeline.alert_history:
        risk_levels = [a['risk_level'] for a in pipeline.alert_history]
        for level in ['critical', 'high', 'medium', 'low']:
            stats['alerts']['by_level'][level] = risk_levels.count(level)
    
    return stats


@app.post("/reset")
async def reset_metrics():
    """Reset all metrics and statistics"""
    pipeline.latency_tracker = {
        'detection': [],
        'tracking': [],
        'cloud': [],
        'total': []
    }
    pipeline.alert_history = []
    
    return {"message": "Metrics reset successfully"}


@app.get("/alerts")
async def get_alerts(limit: int = 100):
    """
    Get recent alerts
    
    Args:
        limit: Maximum number of alerts to return
    
    Returns:
        List of recent alerts
    """
    return {
        "alerts": pipeline.alert_history[-limit:],
        "total": len(pipeline.alert_history)
    }


@app.get("/config")
async def get_config():
    """Get current configuration"""
    return config


def main():
    """Run the API server"""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1  # Single worker for GPU compatibility
    )


if __name__ == "__main__":
    main()
