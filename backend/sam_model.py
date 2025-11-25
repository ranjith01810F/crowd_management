"""SAM (Segment Anything Model) integration for crowd detection and segmentation."""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from segment_anything import sam_model_registry, SamPredictor  # type: ignore
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning("SAM (segment_anything) not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

try:
    from backend.config import Config
except ImportError:
    from config import Config


class SamCrowdModel:
    """SAM-based crowd detection and segmentation model."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, model_type: Optional[str] = None):
        """
        Initialize SAM model.
        
        Args:
            checkpoint_path: Path to SAM checkpoint file. If None, uses Config.SAM_CHECKPOINT
            model_type: SAM model type (vit_h, vit_l, vit_b). If None, uses Config.SAM_MODEL_TYPE
        """
        self.checkpoint_path = checkpoint_path or Config.SAM_CHECKPOINT
        self.model_type = model_type or Config.SAM_MODEL_TYPE
        self.device = Config.SAM_DEVICE
        
        if not SAM_AVAILABLE:
            raise ImportError(
                "SAM is not installed. Install with: "
                "pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        # Check if CUDA is available
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        self.predictor = None
        self._load_model()
        
        # Detection parameters
        self.min_contour_area = 500  # Minimum area for a person detection
        self.density_threshold = 0.3  # Threshold for density calculation
        
    def _load_model(self):
        """Load SAM model from checkpoint."""
        try:
            logger.info(f"Loading SAM model: {self.model_type} from {self.checkpoint_path}")
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            logger.info("SAM model loaded successfully")
        except FileNotFoundError:
            logger.error(f"SAM checkpoint not found at {self.checkpoint_path}")
            logger.info("Please download SAM checkpoint from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
            raise
        except Exception as e:
            logger.error(f"Error loading SAM model: {e}")
            raise
    
    def _detect_people_boxes(self, frame: np.ndarray) -> list:
        """
        Detect bounding boxes for people in the frame.
        This is a placeholder - in production, use a person detector (YOLO, etc.)
        For now, we'll use a simple approach or require manual prompts.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            List of bounding boxes [x1, y1, x2, y2]
        """
        # TODO: Integrate with a person detector (YOLO, etc.)
        # For now, return empty list - will need prompts or detector
        return []
    
    def _segment_with_prompts(self, frame: np.ndarray, boxes: list) -> Tuple[np.ndarray, int]:
        """
        Segment people using SAM with bounding box prompts.
        
        Args:
            frame: Input BGR frame
            boxes: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            Tuple of (masks, count)
        """
        if len(boxes) == 0:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=bool), 0
        
        self.predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        masks = []
        for box in boxes:
            box_array = np.array(box)
            mask, scores, _ = self.predictor.predict(
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
        
        return combined_mask, len(masks)
    
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a frame to detect and segment people.
        
        Args:
            frame: Input BGR frame (numpy array)
            
        Returns:
            Tuple of:
            - segmented_frame: BGR frame with segmentation overlay
            - stats: Dictionary with crowd statistics
        """
        if frame is None or frame.size == 0:
            return frame, {"count": 0, "density_score": 0.0, "error": "Invalid frame"}
        
        try:
            # For now, use a simple approach: detect people boxes first
            # In production, integrate with YOLO or another person detector
            boxes = self._detect_people_boxes(frame)
            
            # Segment using SAM
            if len(boxes) > 0:
                mask, count = self._segment_with_prompts(frame, boxes)
            else:
                # Fallback: return frame with no segmentation
                # In production, you'd want to use a detector here
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=bool)
                count = 0
            
            # Create overlay
            segmented_frame = frame.copy()
            if count > 0:
                # Create colored overlay for masks
                overlay = segmented_frame.copy()
                color = (0, 255, 0)  # Green
                overlay[mask] = color
                segmented_frame = cv2.addWeighted(segmented_frame, 0.7, overlay, 0.3, 0)
                
                # Draw bounding boxes
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(segmented_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate statistics
            frame_area = frame.shape[0] * frame.shape[1]
            mask_area = np.sum(mask)
            density_score = mask_area / frame_area if frame_area > 0 else 0.0
            
            stats = {
                "count": count,
                "density_score": float(density_score),
                "mask_area": int(mask_area),
                "frame_area": int(frame_area),
            }
            
            return segmented_frame, stats
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, {"count": 0, "density_score": 0.0, "error": str(e)}
    
    def process_with_detector(self, frame: np.ndarray, detector_boxes: list) -> Tuple[np.ndarray, Dict]:
        """
        Process frame with pre-detected bounding boxes (from external detector).
        
        Args:
            frame: Input BGR frame
            detector_boxes: List of bounding boxes from detector [x1, y1, x2, y2]
            
        Returns:
            Tuple of (segmented_frame, stats)
        """
        if frame is None or frame.size == 0:
            return frame, {"count": 0, "density_score": 0.0, "error": "Invalid frame"}
        
        try:
            mask, count = self._segment_with_prompts(frame, detector_boxes)
            
            # Create overlay
            segmented_frame = frame.copy()
            if count > 0:
                overlay = segmented_frame.copy()
                color = (0, 255, 0)  # Green
                overlay[mask] = color
                segmented_frame = cv2.addWeighted(segmented_frame, 0.7, overlay, 0.3, 0)
                
                # Draw bounding boxes
                for box in detector_boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(segmented_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Calculate statistics
            frame_area = frame.shape[0] * frame.shape[1]
            mask_area = np.sum(mask)
            density_score = mask_area / frame_area if frame_area > 0 else 0.0
            
            stats = {
                "count": count,
                "density_score": float(density_score),
                "mask_area": int(mask_area),
                "frame_area": int(frame_area),
            }
            
            return segmented_frame, stats
            
        except Exception as e:
            logger.error(f"Error processing frame with detector: {e}")
            return frame, {"count": 0, "density_score": 0.0, "error": str(e)}

