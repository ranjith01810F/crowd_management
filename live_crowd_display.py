"""Live video display with crowd detection and SAM3 segmentation."""

import cv2
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config import Config
from backend.crowd_detector import CrowdDetector, YOLO_AVAILABLE, SAM_AVAILABLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to display live crowd detection."""
    # Initialize camera
    camera_url = Config.CAMERA_URL
    logger.info(f"Connecting to camera: {camera_url}")
    
    # Set buffer size to 1 to get latest frame (avoid lag)
    cap = cv2.VideoCapture(camera_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        logger.error(f"Failed to open camera: {camera_url}")
        logger.error("Make sure:")
        logger.error("  1. Camera URL is correct")
        logger.error("  2. IP Webcam app is running on your phone")
        logger.error("  3. Phone and computer are on the same Wi-Fi")
        return
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Camera connected successfully - Resolution: {width}x{height}, FPS: {fps}")
    
    # Initialize crowd detector
    if not YOLO_AVAILABLE:
        logger.error("YOLOv8 not available. Install with: pip install ultralytics")
        cap.release()
        return
    
    try:
        # Initialize detector with SAM if available
        # Set use_sam=True to enable SAM3 segmentation (requires SAM checkpoint)
        # Check if SAM checkpoint exists
        from pathlib import Path
        sam_checkpoint = Path(Config.SAM_CHECKPOINT)
        use_sam = sam_checkpoint.exists() if sam_checkpoint.is_absolute() else Path.cwd().joinpath(sam_checkpoint).exists()
        
        if use_sam:
            logger.info(f"Initializing detector with SAM3 segmentation...")
            detector = CrowdDetector(use_sam=True)
        else:
            logger.info("Initializing detector with YOLOv8 only (SAM checkpoint not found)")
            logger.info(f"To enable SAM3, download checkpoint to: {Config.SAM_CHECKPOINT}")
            detector = CrowdDetector(use_sam=False)
        logger.info("Crowd detector initialized")
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        cap.release()
        return
    
    logger.info("Starting live display...")
    logger.info("Controls:")
    logger.info("  'q' - Quit")
    logger.info("  's' - Toggle SAM segmentation (if available)")
    logger.info("  '+' - Increase confidence threshold")
    logger.info("  '-' - Decrease confidence threshold")
    
    frame_count = 0
    import time
    last_time = time.time()
    target_fps = 15  # Target FPS for processing (adjust based on performance)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                # Try to reconnect
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(camera_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if not cap.isOpened():
                    logger.error("Failed to reconnect to camera")
                    break
                continue
            
            # Always process frames to show bounding boxes
            # But limit processing rate to maintain responsiveness
            current_time = time.time()
            elapsed = current_time - last_time
            min_frame_time = 1.0 / target_fps
            
            # Process frame immediately (bounding boxes are important)
            last_time = current_time
            
            # Process frame with detector (non-blocking approach)
            try:
                # Process frame - this should include bounding boxes
                processed_frame, stats = detector.process(frame)
                
                # Ensure we have a valid processed frame
                if processed_frame is None or processed_frame.size == 0:
                    processed_frame = frame.copy()
                
                # Add additional info overlay
                info_text = [
                    f"People: {stats.get('count', 0)}",
                    f"Density: {stats.get('density_score', 0.0):.2%}",
                    f"Detector: {stats.get('detector', 'N/A')}",
                ]
                
                # Draw info panel (simpler, faster)
                y_offset = 30
                for i, text in enumerate(info_text):
                    # Simple text with shadow for visibility
                    cv2.putText(
                        processed_frame,
                        text,
                        (12, y_offset + 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 0),
                        3
                    )
                    cv2.putText(
                        processed_frame,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                    y_offset += 35
                
                # Display frame immediately
                cv2.imshow("Live Crowd Detection", processed_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(f"Frames processed: {frame_count}, People: {stats.get('count', 0)}")
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                # Show raw frame on error
                cv2.imshow("Live Crowd Detection", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("Quitting...")
                break
            elif key == ord('s'):
                # Toggle SAM (if available)
                if hasattr(detector, 'use_sam') and SAM_AVAILABLE:
                    detector.use_sam = not detector.use_sam
                    logger.info(f"SAM segmentation: {'ON' if detector.use_sam else 'OFF'}")
                else:
                    logger.warning("SAM not available. Install SAM and download checkpoint to enable.")
            elif key == ord('+') or key == ord('='):
                detector.confidence_threshold = min(0.95, detector.confidence_threshold + 0.05)
                logger.info(f"Confidence threshold: {detector.confidence_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):
                detector.confidence_threshold = max(0.1, detector.confidence_threshold - 0.05)
                logger.info(f"Confidence threshold: {detector.confidence_threshold:.2f}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed")


if __name__ == "__main__":
    main()

