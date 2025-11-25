# Crowd Management System - Complete Documentation

## What This System Does

This is a **Crowd Management System** that watches live video from a camera and automatically counts how many people are in the frame. It can also show you exactly where each person is located by drawing boxes around them, and optionally create precise outlines (masks) around each person's body.

Think of it like a smart security camera that:
- Counts people in real-time
- Shows you where each person is
- Calculates how crowded an area is
- Streams this information live to any connected device

---

## How The Process Works - Step by Step

### The Big Picture Flow

1. **Camera Connection**: The system connects to a video camera (can be an IP camera, webcam, or phone camera)
2. **Frame Capture**: It continuously grabs individual pictures (frames) from the video stream
3. **Person Detection**: Each frame is analyzed to find all the people in it
4. **Counting & Analysis**: The system counts how many people were found and calculates crowd density
5. **Visual Marking**: It draws boxes or outlines around each detected person
6. **Live Streaming**: The processed frames with all the information are sent to connected clients in real-time

### Detailed Process Flow

#### **Step 1: System Startup**
When you start the application:
- The system loads two AI models:
  - **YOLOv8**: A fast person detection model (always used)
  - **SAM (Segment Anything Model)**: An optional advanced model for precise person outlines (only if available)
- It connects to your camera using the configured URL
- It sets up a web server that can accept connections from clients

#### **Step 2: Continuous Video Processing Loop**
Once running, the system enters a continuous loop:

**A. Frame Reading**
- The system reads one frame (picture) from the camera
- If the camera is unavailable, it waits and tries again

**B. Person Detection (YOLOv8)**
- The frame is sent to the YOLOv8 model
- YOLOv8 scans the entire image looking for people
- For each person found, it provides:
  - A bounding box (rectangle coordinates showing where the person is)
  - A confidence score (how sure it is that it's actually a person)

**C. Optional Segmentation (SAM)**
- If SAM is enabled and available:
  - For each person detected by YOLOv8, SAM creates a precise mask (outline)
  - This mask shows the exact shape of the person, not just a box
  - The masks are combined to show all people at once

**D. Visual Overlay Creation**
- The system creates a visual display by:
  - Drawing colored boxes around each detected person
  - If SAM is used, overlaying green masks/outlines on people
  - Adding text showing the person count
  - Adding confidence scores for each detection

**E. Statistics Calculation**
- The system calculates:
  - **Count**: Total number of people detected
  - **Density Score**: What percentage of the frame is occupied by people
  - **Mask Area**: Total area covered by all people (in pixels)
  - **Frame Area**: Total area of the frame
  - **Average Confidence**: How confident the detection was overall

**F. Data Packaging**
- The processed frame is converted to a JPEG image
- The image is encoded as base64 (text format) so it can be sent over the network
- A JSON message is created containing:
  - The encoded image
  - All the statistics
  - A timestamp

**G. Live Streaming**
- The packaged data is sent to any connected WebSocket clients
- The system waits a small amount of time (to control frame rate, default 30 FPS)
- Then it goes back to Step A and repeats

#### **Step 3: Client Connection**
When a client (like a web browser or mobile app) connects:
- The client establishes a WebSocket connection to `/ws/crowd`
- The server immediately starts sending processed frames
- Each message contains a complete frame with all detections and statistics
- The client can display this in real-time

---

## How The Code Works - Component Breakdown

### **1. Configuration (`config.py`)**
**What it does**: Stores all the settings for the system

**Key Settings**:
- Camera URL: Where to find the video feed
- SAM Model Settings: Which AI model to use and where to find it
- WebSocket Settings: How fast to send frames (FPS)
- Server Settings: What port and address to run on

**How it works**: Reads settings from environment variables or uses sensible defaults. This makes it easy to configure without changing code.

---

### **2. Crowd Detector (`crowd_detector.py`)**
**What it does**: The brain of the system - detects and analyzes people in images

**How it works**:

**Initialization**:
- Loads the YOLOv8 model (downloads automatically if not present)
- Optionally loads SAM if requested and available
- Sets up detection parameters (confidence thresholds, etc.)

**Detection Process** (`_detect_people_yolo`):
- Takes a frame (image) as input
- Sends it to YOLOv8 model
- YOLOv8 returns bounding boxes for all detected people
- Filters to only include "person" class (class ID 0)
- Returns list of boxes and confidence scores

**Segmentation Process** (`_segment_with_sam`):
- Takes the frame and bounding boxes from YOLOv8
- For each box, tells SAM: "Segment the person in this area"
- SAM creates a precise mask showing the person's shape
- Combines all masks into one combined mask
- Returns the combined mask

**Main Processing** (`process`):
- Orchestrates the entire detection pipeline
- Calls detection to find people
- Optionally calls segmentation for precise outlines
- Creates visual overlay (boxes, masks, text)
- Calculates all statistics
- Returns the processed frame and statistics dictionary

---

### **3. Main Application (`app.py`)**
**What it does**: The web server that handles connections and coordinates everything

**How it works**:

**Startup** (`startup_event`):
- Runs once when the server starts
- Initializes the CrowdDetector (loads AI models)
- Connects to the camera
- Sets up global variables for the detector and camera

**Shutdown** (`shutdown_event`):
- Runs when the server stops
- Releases the camera connection
- Cleans up resources

**WebSocket Endpoint** (`/ws/crowd`):
- Accepts WebSocket connections from clients
- Enters a continuous loop:
  1. Reads a frame from camera
  2. Processes it with the detector
  3. Encodes the frame as JPEG/base64
  4. Packages everything into JSON
  5. Sends to client
  6. Waits to control frame rate
  7. Repeats
- Handles disconnections gracefully

**REST Endpoints**:
- `GET /`: Returns system status
- `GET /health`: Health check (camera and detector status)
- `POST /api/process-frame`: Processes a single frame on demand (for testing)

---

### **4. SAM Model Integration (`sam_model.py`)**
**What it does**: Provides advanced segmentation capabilities (currently not the primary implementation)

**Note**: This file exists but the main system uses SAM through `crowd_detector.py`. This file provides an alternative implementation that could be used separately.

---

## What Has Been Implemented So Far

### ✅ **Completed Features**

1. **Core Detection System**
   - YOLOv8 person detection fully integrated
   - Automatic model downloading if not present
   - Real-time bounding box drawing
   - Confidence score display

2. **Advanced Segmentation (Optional)**
   - SAM integration for precise person outlines
   - Automatic detection of SAM checkpoint availability
   - Graceful fallback to YOLOv8-only mode if SAM unavailable
   - Combined mask visualization

3. **Web Server**
   - FastAPI-based REST API
   - WebSocket streaming for real-time updates
   - CORS support for web clients
   - Health check endpoints

4. **Live Video Processing**
   - Continuous frame capture from IP cameras
   - Frame rate control (configurable FPS)
   - Error handling and reconnection logic
   - Base64 image encoding for network transmission

5. **Statistics & Analytics**
   - Real-time people counting
   - Crowd density calculation
   - Mask area measurement
   - Average confidence tracking
   - Detector type reporting

6. **Visual Feedback**
   - Colored bounding boxes around people
   - Green mask overlays (when SAM enabled)
   - On-screen statistics display
   - Confidence labels per person

7. **Configuration System**
   - Environment variable support
   - Sensible defaults
   - Easy camera URL configuration
   - Model selection (SAM model types)

8. **Testing & Development Tools**
   - WebSocket test client
   - Live display script for local testing
   - Single frame processing endpoint

### 🔄 **Current Architecture**

The system uses a **two-stage detection approach**:

1. **Stage 1 - Detection (YOLOv8)**: Fast, efficient person detection
   - Scans entire frame
   - Provides bounding boxes
   - Always active

2. **Stage 2 - Segmentation (SAM)**: Optional precise outlining
   - Uses YOLOv8 boxes as starting points
   - Creates pixel-perfect masks
   - Only active if SAM checkpoint is available

This hybrid approach gives you:
- **Speed**: YOLOv8 is very fast
- **Precision**: SAM provides accurate outlines when needed
- **Flexibility**: Works with or without SAM

### 📊 **Data Flow Summary**

```
Camera → Frame Capture → YOLOv8 Detection → [Optional: SAM Segmentation] 
  → Visual Overlay → Statistics Calculation → Base64 Encoding 
  → JSON Packaging → WebSocket Streaming → Client Display
```

### 🎯 **Use Cases Supported**

1. **Real-time Monitoring**: Live crowd counting in public spaces
2. **Density Analysis**: Understanding how crowded an area is
3. **Security Applications**: Person detection and tracking
4. **Analytics**: Collecting crowd statistics over time
5. **Web Integration**: Can be integrated into web dashboards

---

## System Capabilities

### **What It Can Do**
- ✅ Detect multiple people in a single frame
- ✅ Count people in real-time
- ✅ Calculate crowd density
- ✅ Draw visual indicators (boxes/masks)
- ✅ Stream processed video live
- ✅ Handle camera disconnections
- ✅ Work with various camera types (IP cameras, webcams)
- ✅ Provide REST API for integration
- ✅ Support multiple simultaneous clients

### **What It Requires**
- A video camera source (IP camera URL or local webcam)
- Python environment with required packages
- YOLOv8 model (downloads automatically)
- Optional: SAM checkpoint for advanced segmentation
- Optional: GPU for faster processing (works on CPU too)

---

## Technical Stack (Simplified)

- **FastAPI**: Web framework for the API server
- **OpenCV**: Video capture and image processing
- **YOLOv8**: Person detection AI model
- **SAM**: Advanced segmentation AI model (optional)
- **WebSockets**: Real-time bidirectional communication
- **PyTorch**: AI model runtime (for SAM)

---

## Summary

This Crowd Management System is a complete, working solution for real-time people detection and counting. It combines fast detection (YOLOv8) with optional precise segmentation (SAM) to provide accurate, real-time crowd analysis. The system is designed to be flexible, working with or without advanced features, and can stream results to multiple clients simultaneously.

The code is organized into clear components: configuration, detection logic, and web server, making it easy to understand and maintain. All the core functionality is implemented and working, with graceful fallbacks when optional components aren't available.
