"""
Flask Blueprint for Bottle OCR endpoints.

Provides REST API for bottle text recognition.
"""

import os
from flask import Blueprint, request, jsonify, send_from_directory, current_app
from flask_cors import cross_origin
from typing import Dict, Any
import logging

from .processor import BottleOCRProcessor
from .feedback import FrameAnalyzer

logger = logging.getLogger(__name__)

# Create blueprint
bottle_ocr_bp = Blueprint(
    'bottle_ocr',
    __name__,
    url_prefix='/bottle-ocr',
    static_folder='../static',
    static_url_path='/static'
)

# Global processor instance (initialized lazily)
_processor = None
_analyzers: Dict[str, FrameAnalyzer] = {}


def get_processor() -> BottleOCRProcessor:
    """Get or create the OCR processor."""
    global _processor
    if _processor is None:
        use_easyocr = current_app.config.get('BOTTLE_OCR_USE_EASYOCR', False)
        languages = current_app.config.get('BOTTLE_OCR_LANGUAGES', ['en'])
        _processor = BottleOCRProcessor(use_easyocr=use_easyocr, languages=languages)
    return _processor


def get_analyzer(session_id: str) -> FrameAnalyzer:
    """Get or create a frame analyzer for a session."""
    if session_id not in _analyzers:
        _analyzers[session_id] = FrameAnalyzer()
    return _analyzers[session_id]


# ============================================================================
# Static file routes
# ============================================================================

@bottle_ocr_bp.route('/')
@bottle_ocr_bp.route('/capture')
def capture_interface():
    """Serve the bottle capture web interface."""
    static_dir = os.path.join(os.path.dirname(__file__), '..', 'static')
    return send_from_directory(static_dir, 'bottle_capture.html')


# ============================================================================
# OCR Processing endpoints
# ============================================================================

@bottle_ocr_bp.route('/process', methods=['POST'])
@cross_origin()
def process_frames():
    """
    Process multiple frames and return OCR results.

    Request JSON:
    {
        "frames": ["base64_frame1", "base64_frame2", ...],
        "separator": "\\n---\\n"  // optional
    }

    Response JSON:
    {
        "success": true,
        "text": "Full extracted text with separators",
        "segments": ["segment1", "segment2", ...],
        "panorama": "data:image/jpeg;base64,...",
        "frame_count": 12,
        "raw_results": [...]  // detailed OCR data
    }
    """
    try:
        data = request.get_json(force=True)

        frames = data.get('frames', [])
        separator = data.get('separator', '\n---\n')

        if not frames:
            return jsonify({
                'success': False,
                'error': 'No frames provided'
            }), 400

        if len(frames) > 100:
            return jsonify({
                'success': False,
                'error': 'Too many frames (max 100)'
            }), 400

        logger.info(f"Processing {len(frames)} frames")

        processor = get_processor()
        result = processor.process_frames(frames, separator=separator)

        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400

        return jsonify({
            'success': True,
            'text': result['text'],
            'segments': result['segments'],
            'panorama': result['panorama_b64'],
            'frame_count': result['frame_count'],
            'raw_results': result['raw_results']
        })

    except Exception as e:
        logger.exception("Error processing frames")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bottle_ocr_bp.route('/process-single', methods=['POST'])
@cross_origin()
def process_single_frame():
    """
    Process a single frame (for testing/preview).

    Request JSON:
    {
        "frame": "base64_encoded_image"
    }

    Response JSON:
    {
        "success": true,
        "text": "Extracted text",
        "segments": ["segment1", ...],
        "raw_results": [...]
    }
    """
    try:
        data = request.get_json(force=True)

        frame = data.get('frame')
        if not frame:
            return jsonify({
                'success': False,
                'error': 'No frame provided'
            }), 400

        processor = get_processor()
        result = processor.process_single_frame(frame)

        return jsonify({
            'success': True,
            'text': result['text'],
            'segments': result['segments'],
            'raw_results': result['raw_results']
        })

    except Exception as e:
        logger.exception("Error processing single frame")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# Live feedback endpoints
# ============================================================================

@bottle_ocr_bp.route('/analyze-frame', methods=['POST'])
@cross_origin()
def analyze_frame():
    """
    Analyze a frame for capture quality (real-time feedback).

    Request JSON:
    {
        "frame": "base64_encoded_image",
        "session_id": "unique_session_id",
        "frame_index": 0,
        "total_frames_needed": 12
    }

    Response JSON:
    {
        "status": "ok|warning|error|complete",
        "message": "Overall status message",
        "feedbacks": [
            {"message": "...", "type": "warning", "action": "..."}
        ],
        "metrics": {
            "blur_score": 150.5,
            "brightness": 128.0,
            "contrast": 45.2,
            "text_regions": 15,
            "rotation_covered": 45.0,
            "bottle_detected": true
        },
        "progress": 25.0,
        "rotation_complete": false
    }
    """
    try:
        data = request.get_json(force=True)

        frame = data.get('frame')
        session_id = data.get('session_id', 'default')
        frame_index = data.get('frame_index', 0)
        total_frames = data.get('total_frames_needed', 12)

        if not frame:
            return jsonify({
                'success': False,
                'error': 'No frame provided'
            }), 400

        analyzer = get_analyzer(session_id)
        result = analyzer.analyze_frame(frame, frame_index, total_frames)

        return jsonify({
            'success': True,
            **result
        })

    except Exception as e:
        logger.exception("Error analyzing frame")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@bottle_ocr_bp.route('/reset-session', methods=['POST'])
@cross_origin()
def reset_session():
    """
    Reset a capture session's analyzer state.

    Request JSON:
    {
        "session_id": "unique_session_id"
    }
    """
    try:
        data = request.get_json(force=True)
        session_id = data.get('session_id', 'default')

        if session_id in _analyzers:
            _analyzers[session_id].reset()

        return jsonify({'success': True})

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# Health check
# ============================================================================

@bottle_ocr_bp.route('/health', methods=['GET'])
@cross_origin()
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'module': 'bottle-ocr',
        'version': '0.1.0'
    })


@bottle_ocr_bp.route('/config', methods=['GET'])
@cross_origin()
def get_config():
    """Get current OCR configuration."""
    processor = get_processor()
    return jsonify({
        'use_easyocr': processor.use_easyocr,
        'languages': processor.languages,
        'endpoints': {
            'capture_ui': '/bottle-ocr/capture',
            'process': '/bottle-ocr/process',
            'process_single': '/bottle-ocr/process-single',
            'analyze_frame': '/bottle-ocr/analyze-frame',
            'reset_session': '/bottle-ocr/reset-session'
        }
    })
