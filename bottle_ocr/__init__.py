"""
Bottle OCR Module

A Flask-compatible module for recognizing text from rotating bottles
using video frame capture and stitching.
"""

from .processor import BottleOCRProcessor
from .blueprint import bottle_ocr_bp
from .feedback import FrameAnalyzer

__version__ = "0.1.0"
__all__ = ["BottleOCRProcessor", "bottle_ocr_bp", "FrameAnalyzer"]
