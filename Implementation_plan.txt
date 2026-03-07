# Implementation Plan: Crash Detection System Enhancements

## Goal Description
Expand the existing Intelligent Multimodal Edge-AI Crash Detection System according to the published roadmap, integrating synthetic training data pipelines, V2X communication handling, mobile-based alerts, and scalable regional deployments.

## Proposed Changes

### 1. Data Integration (CCD + CARLA)
#### [NEW] `data/crash_dataset.py`
Implement dataset loaders and preprocessing specific to the Car Crash Dataset (CCD) and CARLA synthetic representations, matching the PyTorch Dataset interface used by `train_crash_predictor.py`.

### 2. V2X Communication
#### [NEW] `models/fusion/v2x_module.py`
Build functionality to parse Vehicle-to-Everything (V2X) incoming telemetry formats and incorporate standard fields into the current Multi-Scale Temporal Transformer architecture via cross-modal attention.
#### [MODIFY] `api/server.py`
Add secure endpoints to ingest live streaming V2X payloads from connected roadside units or regional MQTT brokers.

### 3. Mobile Alert Application
#### [NEW] `mobile_app/`
Create a new directory containing a basic mobile interface (e.g., using React Native or Flutter) geared towards subscribing and displaying incoming emergency push notifications and predicted severity levels.
#### [MODIFY] `api/server.py`
Add WebSocket or Server-Sent Events (SSE) support for pushing real-time prediction alerts directly to registered mobile client connections.

### 4. Traffic Control & Deployment Scalability
#### [MODIFY] `deployment/kubernetes/`
Update Kubernetes configurations to support multi-region/multi-city cluster routing and auto-scaling rules based on inference load.
#### [NEW] `api/traffic_control.py`
Create an integration adapter to interface with standardized smart-city traffic light APIs, automatically requesting signal preemption (e.g., turning lights red at an intersection) when a severe crash is predicted.

## Verification Plan

### Automated Tests
- Run `pytest tests/unit/` to ensure the new `crash_dataset.py` parsers handle invalid data gracefully and return consistent batch shapes.
- Create unit tests simulating high-frequency V2X data bursts to measure potential latency bloat on the MSTT-CA pipeline.

### Manual Verification
- Deploy the updated REST API server locally and use an HTTP client to simulate a mobile app receiving low-latency WebSocket notifications upon a simulated crash detection event.
- Use a mock traffic light simulation endpoint and verify if the `traffic_control.py` integration successfully emits the preemption command under 60ms.
