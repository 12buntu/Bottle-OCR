"""
Live feedback analysis for bottle capture.

Provides real-time guidance to users during capture.
"""

import cv2
import numpy as np
import base64
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


class FeedbackType(Enum):
    """Types of feedback messages."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    INFO = "info"


@dataclass
class Feedback:
    """Feedback message with type and details."""
    message: str
    type: FeedbackType
    action: str = ""  # Suggested action

    def to_dict(self) -> Dict[str, str]:
        return {
            'message': self.message,
            'type': self.type.value,
            'action': self.action
        }


class FrameAnalyzer:
    """
    Analyzes frames in real-time to provide capture guidance.

    Checks for:
    - Blur/motion blur
    - Lighting issues
    - Bottle detection
    - Rotation coverage
    - Text visibility
    """

    def __init__(self):
        self.previous_frame = None
        self.frame_history: List[np.ndarray] = []
        self.rotation_coverage = 0.0
        self.feature_detector = cv2.ORB_create(nfeatures=500)

    def decode_frame(self, frame_data: str) -> np.ndarray:
        """Decode base64 frame."""
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def check_blur(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check if frame is blurry using Laplacian variance.

        Returns:
            Tuple of (is_blurry, blur_score)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Threshold determined empirically
        is_blurry = laplacian_var < 100
        return is_blurry, laplacian_var

    def check_lighting(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Check lighting conditions.

        Returns:
            Tuple of (lighting_status, brightness)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness < 50:
            return "too_dark", brightness
        elif brightness > 220:
            return "too_bright", brightness
        else:
            return "ok", brightness

    def check_contrast(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Check if there's enough contrast for text detection.

        Returns:
            Tuple of (has_good_contrast, contrast_score)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()

        has_good_contrast = contrast > 30
        return has_good_contrast, contrast

    def detect_bottle(self, frame: np.ndarray) -> Tuple[bool, Dict[str, int]]:
        """
        Detect if a bottle-like object is in frame.

        Returns:
            Tuple of (bottle_detected, bounding_box)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return False, {}

        # Look for tall, narrow contours (bottle-like)
        frame_h, frame_w = frame.shape[:2]
        min_area = (frame_h * frame_w) * 0.05  # At least 5% of frame

        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = h / w if w > 0 else 0

            # Bottles are typically taller than wide
            if aspect_ratio > 1.2 or w > frame_w * 0.3:
                return True, {'x': x, 'y': y, 'w': w, 'h': h}

        return False, {}

    def estimate_rotation(self, frame: np.ndarray) -> float:
        """
        Estimate rotation progress by comparing with previous frames.

        Returns:
            Estimated rotation angle change in degrees
        """
        if self.previous_frame is None:
            self.previous_frame = frame
            return 0.0

        # Feature matching between frames
        gray1 = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.feature_detector.detectAndCompute(gray1, None)
        kp2, des2 = self.feature_detector.detectAndCompute(gray2, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            self.previous_frame = frame
            return 0.0

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)

        if len(matches) < 5:
            self.previous_frame = frame
            return 0.0

        # Calculate average horizontal displacement
        displacements = []
        for m in matches:
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            displacements.append(pt1[0] - pt2[0])

        avg_displacement = np.median(displacements)

        # Convert displacement to rough angle estimate
        # Assuming bottle takes up ~1/3 of frame width
        frame_w = frame.shape[1]
        bottle_width_approx = frame_w / 3
        angle_per_pixel = 360 / (bottle_width_approx * np.pi)  # Full rotation
        rotation = avg_displacement * angle_per_pixel * 0.5  # Scale factor

        self.previous_frame = frame
        return rotation

    def check_text_presence(self, frame: np.ndarray) -> Tuple[bool, int]:
        """
        Quick check for potential text regions.

        Returns:
            Tuple of (text_likely_present, text_region_count)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use MSER for text detection (fast)
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        # Filter regions by aspect ratio (text-like)
        text_regions = 0
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            if w > 5 and h > 5:  # Minimum size
                aspect = w / h
                if 0.1 < aspect < 10:  # Reasonable text aspect ratio
                    text_regions += 1

        return text_regions > 5, text_regions

    def analyze_frame(self, frame_data: str, frame_index: int = 0,
                      total_frames_needed: int = 12) -> Dict[str, Any]:
        """
        Analyze a frame and provide feedback.

        Args:
            frame_data: Base64-encoded frame
            frame_index: Current frame index in capture sequence
            total_frames_needed: Target number of frames for full rotation

        Returns:
            Analysis results with feedback messages
        """
        frame = self.decode_frame(frame_data)
        feedbacks: List[Feedback] = []

        # Check blur
        is_blurry, blur_score = self.check_blur(frame)
        if is_blurry:
            feedbacks.append(Feedback(
                message="Image is blurry",
                type=FeedbackType.WARNING,
                action="Hold the bottle steady or rotate more slowly"
            ))

        # Check lighting
        lighting_status, brightness = self.check_lighting(frame)
        if lighting_status == "too_dark":
            feedbacks.append(Feedback(
                message="Image is too dark",
                type=FeedbackType.WARNING,
                action="Move to better lighting"
            ))
        elif lighting_status == "too_bright":
            feedbacks.append(Feedback(
                message="Image is overexposed",
                type=FeedbackType.WARNING,
                action="Reduce lighting or move away from direct light"
            ))

        # Check contrast
        has_contrast, contrast_score = self.check_contrast(frame)
        if not has_contrast:
            feedbacks.append(Feedback(
                message="Low contrast detected",
                type=FeedbackType.INFO,
                action="Ensure text on bottle is clearly visible"
            ))

        # Detect bottle
        bottle_found, bottle_bbox = self.detect_bottle(frame)
        if not bottle_found:
            feedbacks.append(Feedback(
                message="Bottle not detected",
                type=FeedbackType.ERROR,
                action="Move bottle into frame"
            ))
        else:
            # Check bottle position
            frame_h, frame_w = frame.shape[:2]
            center_x = bottle_bbox['x'] + bottle_bbox['w'] / 2

            if center_x < frame_w * 0.3:
                feedbacks.append(Feedback(
                    message="Bottle too far left",
                    type=FeedbackType.INFO,
                    action="Move bottle to center"
                ))
            elif center_x > frame_w * 0.7:
                feedbacks.append(Feedback(
                    message="Bottle too far right",
                    type=FeedbackType.INFO,
                    action="Move bottle to center"
                ))

            # Check if bottle fills enough of frame
            bottle_area = bottle_bbox['w'] * bottle_bbox['h']
            frame_area = frame_h * frame_w
            if bottle_area < frame_area * 0.15:
                feedbacks.append(Feedback(
                    message="Bottle is too small",
                    type=FeedbackType.WARNING,
                    action="Move camera closer to the bottle"
                ))
            elif bottle_area > frame_area * 0.85:
                feedbacks.append(Feedback(
                    message="Bottle is too large",
                    type=FeedbackType.WARNING,
                    action="Move camera away from the bottle"
                ))

        # Check text presence
        text_found, text_regions = self.check_text_presence(frame)
        if not text_found:
            feedbacks.append(Feedback(
                message="No text detected in this view",
                type=FeedbackType.INFO,
                action="Rotate bottle to show label"
            ))

        # Estimate rotation
        rotation = self.estimate_rotation(frame)
        self.rotation_coverage += abs(rotation)

        # Rotation progress
        progress = min(100, (frame_index + 1) / total_frames_needed * 100)
        rotation_complete = self.rotation_coverage >= 300  # ~300 degrees should be enough

        # Generate overall status
        error_count = sum(1 for f in feedbacks if f.type == FeedbackType.ERROR)
        warning_count = sum(1 for f in feedbacks if f.type == FeedbackType.WARNING)

        if error_count > 0:
            overall_status = "error"
            overall_message = "Fix issues before continuing"
        elif warning_count > 0:
            overall_status = "warning"
            overall_message = "Capture quality may be affected"
        elif rotation_complete:
            overall_status = "complete"
            overall_message = "Full rotation captured! Ready to process"
        else:
            overall_status = "ok"
            remaining = max(0, total_frames_needed - frame_index - 1)
            overall_message = f"Keep rotating ({remaining} more frames suggested)"

        return {
            'status': overall_status,
            'message': overall_message,
            'feedbacks': [f.to_dict() for f in feedbacks],
            'metrics': {
                'blur_score': round(blur_score, 2),
                'brightness': round(brightness, 2),
                'contrast': round(contrast_score, 2),
                'text_regions': text_regions,
                'rotation_covered': round(self.rotation_coverage, 1),
                'bottle_detected': bottle_found,
                'bottle_bbox': bottle_bbox if bottle_found else None
            },
            'progress': round(progress, 1),
            'rotation_complete': rotation_complete
        }

    def reset(self):
        """Reset analyzer state for new capture session."""
        self.previous_frame = None
        self.frame_history = []
        self.rotation_coverage = 0.0
