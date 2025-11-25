"""Configuration settings for the crowd management backend."""

import os
from typing import Optional

class Config:
    """Application configuration."""
    
    # Camera settings
    CAMERA_URL: str = os.getenv("CAMERA_URL", "http://192.168.1.11:8080/video")
    
    # SAM Model settings
    SAM_CHECKPOINT: str = os.getenv("SAM_CHECKPOINT", "sam_vit_b_01ec64.pth")
    SAM_MODEL_TYPE: str = os.getenv("SAM_MODEL_TYPE", "vit_b")  # vit_h, vit_l, vit_b
    SAM_DEVICE: str = os.getenv("SAM_DEVICE", "cuda")  # cuda or cpu
    
    # WebSocket settings
    WS_FPS: float = float(os.getenv("WS_FPS", "30"))  # Frames per second
    WS_FRAME_DELAY: float = 1.0 / WS_FPS
    
    # Image encoding settings
    JPEG_QUALITY: int = int(os.getenv("JPEG_QUALITY", "85"))
    
    # CORS settings
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",")
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))

