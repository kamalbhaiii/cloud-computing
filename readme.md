
## Wildlife Detection Using Raspberry Pi and Kubernetes Clustering

## Introduction
This project focuses on building a lightweight, scalable wildlife detection system using a Kubernetes cluster of Raspberry Pi devices. It detects specific animals — cats, dogs, and fish — using camera sensors connected to Raspberry Pi boards. Detection happens in real time using a pre-trained YOLO model, and the system is designed for flexibility, modularity, and edge processing.

The architecture integrates microservices (frontend, backend, database, monitoring, and notification bot), ensuring that each service is independently deployed and managed using Kubernetes.


##  System Components
1. Sensor Node (Raspberry Pi 4)
    Equipped with a Pi Camera Module.
    Captures live images from the environment.
    Performs pre-processing (cropping, resizing, background removal if needed).
    Locally infers using a YOLO lightweight model.
    Sends detection results to the backend service.

2. Master Node (Raspberry Pi 5)
    Acts as the Kubernetes Control Plane.
    Manages deployment, scaling, and service discovery.
    Handles the orchestration of microservices and sensor communications.

3. Service Nodes (Raspberry Pi 3s)
    Each Raspberry Pi 3 is dedicated to running a particular service:
    
    Frontend: Displays real-time detections and historical data via a user-friendly web interface.
    
    Backend: Acts as the main API server to collect data from the sensor node and manage data flow between services.
    
    Database (MariaDB): Stores annotations (detection results like labels, time stamps, coordinates).
    
    Monitoring: Collects logs and system metrics from all nodes and services (e.g., Prometheus + Grafana stack).
    
    Notification Bot: Sends immediate alerts to users when specific animals are detected (e.g., through Telegram or email).
## Architecture
The system comprises a Raspberry Pi 4 as a sensor node equipped with a camera module and a trained lightweight YOLO model for object detection. A Raspberry Pi 5 acts as the Kubernetes master node managing the cluster, and several Raspberry Pi 3 boards serve as worker nodes hosting microservices including:

Frontend Interface

Backend API

MariaDB Database

Monitoring Service (e.g., Prometheus/Grafana)

Notification Bot (for alerts)
## Workflow
1.Data Preparation and Model Training:
    Collect images from open-source datasets.
    Pre-process the images (background removal, resizing).
    Train a YOLO model in the cloud and optimize it for edge devices.

2.Real-time Detection:
    Camera module captures image → Pre-processing → YOLO model detects object → Sends annotation to backend.

3.Storage:
    Images are stored in MinIO (S3-compatible object storage).
    Annotations (object name, confidence, timestamp) are stored in MariaDB.

4.Monitoring:
    Cluster metrics, pod status, and system health visualized via a Monitoring Dashboard.

5.Notification:
    If a specific detection (e.g., "Dog detected") occurs, the Notification Bot sends an alert.
    
![App Screenshot]("C:\Users\Varshitha Ramamurthy\OneDrive\Desktop\CC image.png")
   

## Technologies Used


Object Detection :                    YOLOv5 / YOLOv8 (optimized)

Edge Device :                         Raspberry Pi (4, 5, 3)

Orchestration :                       Kubernetes (kubeadm, kubectl)

Database :                            MariaDB

File Storage :                        MinIO

Backend :                            Python (FastAPI or Flask)

Frontend :                            React.js 

Monitoring :                          Prometheus + Grafana

Notification :                         Python Bot (Telegram )