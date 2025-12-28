"""
Core OCR processing module for bottle text recognition.

Handles frame stitching, panorama creation, and OCR extraction.
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
import io
import base64
from typing import List, Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BottleOCRProcessor:
    """
    Processes video frames of a rotating bottle to extract text.

    Workflow:
    1. Receive frames from browser capture
    2. Detect overlapping regions between consecutive frames
    3. Stitch frames into a cylindrical panorama
    4. Run OCR on the stitched result
    5. Segment text by detected regions/lines
    """

    def __init__(self, use_easyocr: bool = False, languages: List[str] = None):
        """
        Initialize the processor.

        Args:
            use_easyocr: Use EasyOCR instead of Tesseract (better for complex fonts)
            languages: List of language codes (default: ['en'])
        """
        self.use_easyocr = use_easyocr
        self.languages = languages or ['en']
        self._easyocr_reader = None

        if use_easyocr:
            self._init_easyocr()

    def _init_easyocr(self):
        """Lazy-load EasyOCR reader."""
        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader(self.languages, gpu=False)
        except ImportError:
            logger.warning("EasyOCR not available, falling back to Tesseract")
            self.use_easyocr = False

    def decode_frame(self, frame_data: str) -> np.ndarray:
        """
        Decode a base64-encoded frame to numpy array.

        Args:
            frame_data: Base64 encoded image (with or without data URI prefix)

        Returns:
            OpenCV image (BGR format)
        """
        # Remove data URI prefix if present
        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        # Decode base64 to bytes
        img_bytes = base64.b64decode(frame_data)

        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)

        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        return img

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame for better stitching and OCR.

        Args:
            frame: Input BGR image

        Returns:
            Preprocessed image
        """
        # Resize if too large (for performance)
        max_height = 720
        if frame.shape[0] > max_height:
            scale = max_height / frame.shape[0]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        # Slight denoise while preserving edges
        frame = cv2.bilateralFilter(frame, 9, 75, 75)

        return frame

    def extract_bottle_region(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Extract the bottle region from the frame using edge detection.

        Args:
            frame: Input BGR image

        Returns:
            Tuple of (cropped region, bounding box dict)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return frame, {'x': 0, 'y': 0, 'w': frame.shape[1], 'h': frame.shape[0]}

        # Find the largest contour (likely the bottle)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)

        cropped = frame[y:y+h, x:x+w]

        return cropped, {'x': x, 'y': y, 'w': w, 'h': h}

    def stitch_frames(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Stitch multiple frames into a panoramic image.

        Uses OpenCV's Stitcher for robust panorama creation.

        Args:
            frames: List of preprocessed frames in rotation order

        Returns:
            Stitched panoramic image, or None if stitching fails
        """
        if len(frames) < 2:
            return frames[0] if frames else None

        # Try OpenCV stitcher first
        try:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            status, pano = stitcher.stitch(frames)

            if status == cv2.Stitcher_OK:
                logger.info(f"Stitching successful: {pano.shape}")
                return pano
            else:
                logger.warning(f"Stitcher failed with status {status}, trying manual approach")
        except Exception as e:
            logger.warning(f"Stitcher error: {e}, trying manual approach")

        # Fallback: Manual feature-based stitching
        return self._manual_stitch(frames)

    def _manual_stitch(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Manual stitching using feature matching.

        Args:
            frames: List of frames to stitch

        Returns:
            Stitched result or horizontal concatenation as fallback
        """
        if len(frames) < 2:
            return frames[0] if frames else None

        # Initialize feature detector
        orb = cv2.ORB_create(nfeatures=1000)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        result = frames[0].copy()

        for i in range(1, len(frames)):
            # Detect keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(result, None)
            kp2, des2 = orb.detectAndCompute(frames[i], None)

            if des1 is None or des2 is None:
                # No features found, just concatenate
                result = np.hstack([result, frames[i]])
                continue

            # Match features
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) < 10:
                # Not enough matches, concatenate
                result = np.hstack([result, frames[i]])
                continue

            # Get matching points
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

            # Find homography
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

            if H is None:
                result = np.hstack([result, frames[i]])
                continue

            # Warp and blend
            h1, w1 = result.shape[:2]
            h2, w2 = frames[i].shape[:2]

            # Calculate output size
            corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, H)

            all_corners = np.concatenate([
                np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
                transformed
            ])

            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

            # Limit size to prevent memory issues
            if x_max - x_min > 5000 or y_max - y_min > 2000:
                result = np.hstack([result, frames[i]])
                continue

            translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

            output_size = (x_max - x_min, y_max - y_min)
            warped = cv2.warpPerspective(frames[i], translation @ H, output_size)

            # Place original result
            warped[-y_min:-y_min+h1, -x_min:-x_min+w1] = result

            result = warped

        return result

    def enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image for better OCR results.

        Args:
            image: Input BGR image

        Returns:
            Enhanced grayscale image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Denoise
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Adaptive thresholding for text
        # We'll return both the enhanced gray and the thresholded for OCR to try
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        return thresh

    def run_ocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run OCR on the image and return structured results.

        Args:
            image: Preprocessed image (grayscale or BGR)

        Returns:
            List of text blocks with positions and confidence
        """
        if self.use_easyocr and self._easyocr_reader:
            return self._run_easyocr(image)
        else:
            return self._run_tesseract(image)

    def _run_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run Tesseract OCR."""
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)

        # Get detailed OCR data
        data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)

        results = []
        current_block = {'text': '', 'words': [], 'confidence': 0, 'block_num': -1}

        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            block_num = data['block_num'][i]

            if conf < 0:  # Skip invalid entries
                continue

            if block_num != current_block['block_num'] and current_block['text']:
                # Save previous block
                if current_block['words']:
                    current_block['confidence'] = sum(
                        w['confidence'] for w in current_block['words']
                    ) / len(current_block['words'])
                results.append(current_block)
                current_block = {'text': '', 'words': [], 'confidence': 0, 'block_num': block_num}

            current_block['block_num'] = block_num

            if text and conf > 30:  # Filter low confidence
                current_block['text'] += (' ' if current_block['text'] else '') + text
                current_block['words'].append({
                    'text': text,
                    'confidence': conf,
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i]
                })

        # Don't forget the last block
        if current_block['text']:
            if current_block['words']:
                current_block['confidence'] = sum(
                    w['confidence'] for w in current_block['words']
                ) / len(current_block['words'])
            results.append(current_block)

        return results

    def _run_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Run EasyOCR."""
        if self._easyocr_reader is None:
            self._init_easyocr()
            if self._easyocr_reader is None:
                return self._run_tesseract(image)

        # EasyOCR expects BGR or grayscale
        raw_results = self._easyocr_reader.readtext(image)

        results = []
        for bbox, text, conf in raw_results:
            if conf < 0.3:  # Filter low confidence
                continue

            # Convert bbox to x, y, w, h
            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x, y = min(x_coords), min(y_coords)
            w, h = max(x_coords) - x, max(y_coords) - y

            results.append({
                'text': text,
                'confidence': conf * 100,  # Match Tesseract scale
                'words': [{
                    'text': text,
                    'confidence': conf * 100,
                    'x': int(x),
                    'y': int(y),
                    'w': int(w),
                    'h': int(h)
                }]
            })

        return results

    def segment_text(self, ocr_results: List[Dict[str, Any]],
                     separator: str = "\n---\n") -> str:
        """
        Segment OCR results into logical text blocks.

        Args:
            ocr_results: List of OCR result blocks
            separator: String to use between segments

        Returns:
            Formatted text with segment separators
        """
        if not ocr_results:
            return ""

        # Sort by vertical position (top to bottom)
        sorted_results = sorted(ocr_results, key=lambda x:
            min(w['y'] for w in x['words']) if x['words'] else 0
        )

        segments = []
        current_segment = []
        last_y = None

        for block in sorted_results:
            if not block['words']:
                continue

            block_y = min(w['y'] for w in block['words'])
            block_h = max(w['h'] for w in block['words'])

            # Check if this is a new segment (large vertical gap)
            if last_y is not None:
                gap = block_y - last_y
                if gap > block_h * 2:  # Gap larger than 2x line height
                    if current_segment:
                        segments.append(' '.join(current_segment))
                        current_segment = []

            current_segment.append(block['text'])
            last_y = block_y + block_h

        # Add last segment
        if current_segment:
            segments.append(' '.join(current_segment))

        return separator.join(segments)

    def process_frames(self, frame_data_list: List[str],
                       separator: str = "\n---\n") -> Dict[str, Any]:
        """
        Main processing pipeline: decode, stitch, OCR, segment.

        Args:
            frame_data_list: List of base64-encoded frames
            separator: Text segment separator

        Returns:
            Dict with 'text', 'segments', 'raw_results', 'panorama_b64'
        """
        logger.info(f"Processing {len(frame_data_list)} frames")

        # Decode frames
        frames = []
        for i, frame_data in enumerate(frame_data_list):
            try:
                frame = self.decode_frame(frame_data)
                frame = self.preprocess_frame(frame)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to decode frame {i}: {e}")

        if not frames:
            return {
                'text': '',
                'segments': [],
                'raw_results': [],
                'panorama_b64': None,
                'error': 'No valid frames to process'
            }

        # Stitch frames
        panorama = self.stitch_frames(frames)

        if panorama is None:
            # Fallback: process individual frames
            panorama = frames[len(frames) // 2]  # Use middle frame

        # Enhance for OCR
        enhanced = self.enhance_for_ocr(panorama)

        # Run OCR
        ocr_results = self.run_ocr(panorama)  # Try original first

        if not ocr_results or sum(r.get('confidence', 0) for r in ocr_results) / max(len(ocr_results), 1) < 50:
            # Try enhanced version
            enhanced_results = self.run_ocr(enhanced)
            if len(enhanced_results) > len(ocr_results):
                ocr_results = enhanced_results

        # Segment text
        segmented_text = self.segment_text(ocr_results, separator)
        segments = segmented_text.split(separator) if segmented_text else []

        # Encode panorama for response
        _, buffer = cv2.imencode('.jpg', panorama, [cv2.IMWRITE_JPEG_QUALITY, 85])
        panorama_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'text': segmented_text,
            'segments': segments,
            'raw_results': ocr_results,
            'panorama_b64': f"data:image/jpeg;base64,{panorama_b64}",
            'frame_count': len(frames)
        }

    def process_single_frame(self, frame_data: str) -> Dict[str, Any]:
        """
        Process a single frame (for preview/testing).

        Args:
            frame_data: Base64-encoded frame

        Returns:
            OCR results for the single frame
        """
        frame = self.decode_frame(frame_data)
        frame = self.preprocess_frame(frame)

        ocr_results = self.run_ocr(frame)
        text = self.segment_text(ocr_results)

        return {
            'text': text,
            'segments': text.split("\n---\n") if text else [],
            'raw_results': ocr_results
        }
