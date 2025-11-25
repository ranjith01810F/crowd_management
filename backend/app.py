"""FastAPI application for crowd management with SAM and WebSocket streaming."""

import cv2
import base64
import asyncio
import json
import logging
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from backend.config import Config
    from backend.crowd_detector import CrowdDetector, YOLO_AVAILABLE
except ImportError:
    from config import Config
    from crowd_detector import CrowdDetector, YOLO_AVAILABLE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Crowd Management API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.CORS_ORIGINS if Config.CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model: Optional[CrowdDetector] = None
cap: Optional[cv2.VideoCapture] = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and camera on startup."""
    global model, cap
    
    try:
        # Initialize crowd detector (YOLOv8 with optional SAM)
        if YOLO_AVAILABLE:
            try:
                # Check if SAM checkpoint exists
                from pathlib import Path
                sam_checkpoint = Path(Config.SAM_CHECKPOINT)
                use_sam = sam_checkpoint.exists() if sam_checkpoint.is_absolute() else Path.cwd().joinpath(sam_checkpoint).exists()
                
                if use_sam:
                    logger.info(f"Initializing detector with SAM3 segmentation...")
                    model = CrowdDetector(use_sam=True)
                    logger.info("Crowd detector (YOLOv8 + SAM3) initialized successfully")
                else:
                    logger.info("Initializing detector with YOLOv8 only (SAM checkpoint not found)")
                    logger.info(f"To enable SAM3, download checkpoint to: {Config.SAM_CHECKPOINT}")
                    model = CrowdDetector(use_sam=False)
                    logger.info("Crowd detector (YOLOv8) initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize crowd detector: {e}")
                logger.warning("Running without crowd detection")
                model = None
        else:
            logger.warning("YOLOv8 not available - install with: pip install ultralytics")
            model = None
        
        # Initialize camera
        logger.info(f"Connecting to camera: {Config.CAMERA_URL}")
        cap = cv2.VideoCapture(Config.CAMERA_URL)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera: {Config.CAMERA_URL}")
            cap = None
        else:
            logger.info("Camera connected successfully")
            
    except Exception as e:
        logger.error(f"Error during startup: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global cap
    if cap is not None:
        cap.release()
        logger.info("Camera released")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Crowd Management API",
        "status": "running",
        "camera_connected": cap is not None and cap.isOpened() if cap else False,
        "detector_available": model is not None
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    camera_status = cap is not None and cap.isOpened() if cap else False
    return {
        "status": "healthy" if camera_status else "degraded",
        "camera_connected": camera_status,
        "detector_available": model is not None
    }


@app.websocket("/ws/crowd")
async def crowd_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming crowd detection and segmentation.
    
    Sends JSON messages with:
    - image: base64-encoded JPEG frame
    - stats: dictionary with crowd statistics
    """
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            if cap is None or not cap.isOpened():
                await websocket.send_json({
                    "error": "Camera not available",
                    "stats": {"count": 0, "density_score": 0.0}
                })
                await asyncio.sleep(1.0)
                continue
            
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                await asyncio.sleep(Config.WS_FRAME_DELAY)
                continue
            
            # Process frame with crowd detector
            if model is not None:
                try:
                    processed_frame, stats = model.process(frame)
                    frame_to_send = processed_frame
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    frame_to_send = frame
                    stats = {"count": 0, "density_score": 0.0, "error": str(e)}
            else:
                # No detector - send raw frame
                frame_to_send = frame
                stats = {"count": 0, "density_score": 0.0, "message": "YOLOv8 not available. Install with: pip install ultralytics"}
            
            # Encode frame as JPEG
            try:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
                _, buffer = cv2.imencode(".jpg", frame_to_send, encode_params)
                jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            except Exception as e:
                logger.error(f"Error encoding frame: {e}")
                await asyncio.sleep(Config.WS_FRAME_DELAY)
                continue
            
            # Prepare payload
            payload = {
                "image": jpg_as_text,
                "stats": stats,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Send via WebSocket
            try:
                await websocket.send_text(json.dumps(payload))
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
                break
            
            # Control frame rate
            await asyncio.sleep(Config.WS_FRAME_DELAY)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("WebSocket connection closed")


@app.post("/api/process-frame")
async def process_frame_endpoint():
    """
    REST endpoint to process a single frame (for testing).
    Returns the processed frame and stats.
    """
    if cap is None or not cap.isOpened():
        return JSONResponse(
            status_code=503,
            content={"error": "Camera not available"}
        )
    
    ret, frame = cap.read()
    if not ret:
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to read frame"}
        )
    
    if model is not None:
        try:
            segmented_frame, stats = model.process(frame)
            # Encode frame
            _, buffer = cv2.imencode(".jpg", segmented_frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            
            return {
                "image": jpg_as_text,
                "stats": stats
            }
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Processing error: {str(e)}"}
            )
    else:
        # No model - return raw frame
        _, buffer = cv2.imencode(".jpg", frame)
        jpg_as_text = base64.b64encode(buffer).decode("utf-8")
        
        return {
            "image": jpg_as_text,
            "stats": {"count": 0, "density_score": 0.0, "message": "Detector not available"}
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=True
    )

