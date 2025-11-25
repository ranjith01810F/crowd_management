"""Crowd detection using YOLOv8 for person detection and optional SAM for segmentation."""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

# Try to import YOLOv8
try:
    from ultralytics import YOLO  # type: ignore
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLOv8 (ultralytics) not available. Install with: pip install ultralytics")

# Try to import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

try:
    from backend.config import Config
except ImportError:
    from config import Config


class CrowdDetector:
    """Crowd detection using YOLOv8 with optional SAM segmentation."""
    
    def __init__(
        self,
        yolo_model_path: Optional[str] = None,
        use_sam: bool = False,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: Optional[str] = None
    ):
        """
        Initialize crowd detector.
        
        Args:
            yolo_model_path: Path to YOLO model file. If None, uses default YOLOv8n
            use_sam: Whether to use SAM for segmentation (requires SAM to be installed)
            sam_checkpoint: Path to SAM checkpoint (required if use_sam=True)
            sam_model_type: SAM model type (vit_h, vit_l, vit_b)
        """
        self.use_sam = use_sam and SAM_AVAILABLE
        self.yolo_model = None
        self.sam_predictor = None
        
        # Initialize YOLO
        if not YOLO_AVAILABLE:
            raise ImportError(
                "YOLOv8 is not installed. Install with: pip install ultralytics"
            )
        
        try:
            if yolo_model_path:
                logger.info(f"Loading YOLO model from {yolo_model_path}")
                self.yolo_model = YOLO(yolo_model_path)
            else:
                logger.info("Loading default YOLOv8n model (will download automatically)")
                self.yolo_model = YOLO('yolov8n.pt')  # nano model, fastest
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {e}")
            raise
        
        # Initialize SAM if requested
        if self.use_sam:
            if not SAM_AVAILABLE:
                logger.warning("SAM requested but not available. Continuing without SAM.")
                self.use_sam = False
            else:
                try:
                    checkpoint = sam_checkpoint or Config.SAM_CHECKPOINT
                    model_type = sam_model_type or Config.SAM_MODEL_TYPE
                    device = Config.SAM_DEVICE
                    
                    if device == "cuda" and not torch.cuda.is_available():
                        logger.warning("CUDA not available, using CPU for SAM")
                        device = "cpu"
                    
                    logger.info(f"Loading SAM model: {model_type} from {checkpoint}")
                    sam = sam_model_registry[model_type](checkpoint=checkpoint)
                    sam.to(device=device)
                    self.sam_predictor = SamPredictor(sam)
                    logger.info("SAM model loaded successfully")
                except Exception as e:
                    logger.error(f"Error loading SAM model: {e}")
                    logger.warning("Continuing without SAM segmentation")
                    self.use_sam = False
        
        # Detection parameters
        self.confidence_threshold = 0.25  # YOLO confidence threshold
        self.person_class_id = 0  # COCO class 0 is 'person'
    
    def _detect_people_yolo(self, frame: np.ndarray) -> Tuple[List, List]:
        """
        Detect people using YOLO.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (boxes, confidences)
            boxes: List of [x1, y1, x2, y2]
            confidences: List of confidence scores
        """
        if self.yolo_model is None:
            return [], []
        
        # YOLO expects RGB
        results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)
        
        boxes = []
        confidences = []
        
        for result in results:
            boxes_data = result.boxes
            for i in range(len(boxes_data)):
                # Check if it's a person (class 0)
                cls = int(boxes_data.cls[i])
                if cls == self.person_class_id:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = boxes_data.xyxy[i].cpu().numpy()
                    conf = float(boxes_data.conf[i].cpu().numpy())
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    confidences.append(conf)
        
        return boxes, confidences
    
    def _segment_with_sam(self, frame: np.ndarray, boxes: List) -> np.ndarray:
        """
        Segment people using SAM with YOLO bounding boxes as prompts.
        
        Args:
            frame: Input BGR frame
            boxes: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            Combined mask for all people
        """
        if not self.use_sam or self.sam_predictor is None or len(boxes) == 0:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
        
        self.sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        masks = []
        for box in boxes:
            box_array = np.array(box)
            mask, scores, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_array[None, :],
                multimask_output=False,
            )
            masks.append(mask[0])
        
        if len(masks) > 0:
            combined_mask = np.logical_or.reduce(masks)
        else:
            combined_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
        
        return combined_mask
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a frame to detect and segment people.
        
        Args:
            frame: Input BGR frame (numpy array)
            
        Returns:
            Tuple of:
            - processed_frame: BGR frame with detection/segmentation overlay
            - stats: Dictionary with crowd statistics
        """
        if frame is None or frame.size == 0:
            return frame, {"count": 0, "density_score": 0.0, "error": "Invalid frame"}
        
        try:
            # Detect people with YOLO
            boxes, confidences = self._detect_people_yolo(frame)
            count = len(boxes)
            
            # Create overlay frame
            processed_frame = frame.copy()
            
            if count > 0:
                # Get segmentation mask if SAM is enabled
                if self.use_sam:
                    mask = self._segment_with_sam(frame, boxes)
                    
                    # Create colored overlay for masks with better visibility
                    overlay = processed_frame.copy()
                    # Use bright green for SAM masks
                    color = (0, 255, 0)  # BGR format - Green
                    overlay[mask] = color
                    # Blend overlay with original frame (more visible)
                    processed_frame = cv2.addWeighted(processed_frame, 0.5, overlay, 0.5, 0)
                    
                    # Draw mask contours for better visibility
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(processed_frame, contours, -1, (0, 255, 0), 2)
                
                # Draw bounding boxes with different colors
                colors = [
                    (0, 255, 0),    # Green
                    (255, 0, 0),    # Blue
                    (0, 0, 255),    # Red
                    (255, 255, 0),  # Cyan
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Yellow
                ]
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    conf = confidences[i] if i < len(confidences) else 0.0
                    
                    # Use different colors for variety
                    color = colors[i % len(colors)]
                    
                    # Draw rectangle with thicker lines
                    thickness = 3 if self.use_sam else 2
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw confidence score with background
                    label = f"Person {conf:.2f}"
                    (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    # Background rectangle for text
                    cv2.rectangle(processed_frame, (x1, y1 - h - 8), (x1 + w + 4, y1), color, -1)
                    cv2.putText(processed_frame, label, (x1 + 2, y1 - 4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw count on frame
                count_text = f"People: {count}"
                # Draw background for count text
                (text_w, text_h), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2.rectangle(processed_frame, (5, 5), (15 + text_w, 35), (0, 0, 0), -1)
                cv2.putText(processed_frame, count_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Calculate statistics
            frame_area = frame.shape[0] * frame.shape[1]
            
            if self.use_sam and count > 0:
                mask = self._segment_with_sam(frame, boxes)
                mask_area = np.sum(mask)
            else:
                # Estimate mask area from bounding boxes
                mask_area = sum((box[2] - box[0]) * (box[3] - box[1]) for box in boxes)
            
            density_score = mask_area / frame_area if frame_area > 0 else 0.0
            
            stats = {
                "count": count,
                "density_score": float(density_score),
                "mask_area": int(mask_area),
                "frame_area": int(frame_area),
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "detector": "YOLOv8" + (" + SAM" if self.use_sam else "")
            }
            
            return processed_frame, stats
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, {"count": 0, "density_score": 0.0, "error": str(e)}

