"""
Integration Example for eln_web_backend/app.py

This file shows how to integrate the Bottle OCR module into your existing
Flask application. Copy the relevant sections into your app.py.
"""

# =============================================================================
# STEP 1: Add to your imports (at the top of app.py)
# =============================================================================

# Add this import alongside your other imports:
# from bottle_ocr import bottle_ocr_bp

# =============================================================================
# STEP 2: Register the blueprint (after creating Flask app)
# =============================================================================

# Add these lines after `app = Flask(__name__)`:
#
# # Bottle OCR Configuration (optional)
# app.config['BOTTLE_OCR_USE_EASYOCR'] = False  # Set True for EasyOCR instead of Tesseract
# app.config['BOTTLE_OCR_LANGUAGES'] = ['en']   # Languages to recognize
#
# # Register the Bottle OCR blueprint
# app.register_blueprint(bottle_ocr_bp)

# =============================================================================
# FULL EXAMPLE: Modified app.py
# =============================================================================

"""
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import cross_origin
import print_handling
import pth_data
from datetime import datetime
import json
import search_process
import eln_packages_common.resourcemanage as resourcemanage
import label_creating

# Import Bottle OCR module
from bottle_ocr import bottle_ocr_bp

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "pth_data.csv")

# Bottle OCR Configuration
app.config['BOTTLE_OCR_USE_EASYOCR'] = False
app.config['BOTTLE_OCR_LANGUAGES'] = ['en']

# Register Bottle OCR endpoints
app.register_blueprint(bottle_ocr_bp)

# ... rest of your existing code ...
"""

# =============================================================================
# AVAILABLE ENDPOINTS AFTER INTEGRATION
# =============================================================================

"""
After integration, the following endpoints will be available:

CAPTURE INTERFACE:
  GET  /bottle-ocr/           - Web interface for capturing bottle images
  GET  /bottle-ocr/capture    - Same as above (alias)

PROCESSING:
  POST /bottle-ocr/process         - Process multiple frames, return OCR text
  POST /bottle-ocr/process-single  - Process a single frame (for testing)

LIVE FEEDBACK:
  POST /bottle-ocr/analyze-frame   - Analyze frame quality in real-time
  POST /bottle-ocr/reset-session   - Reset capture session state

INFO:
  GET  /bottle-ocr/health   - Health check
  GET  /bottle-ocr/config   - Current configuration

EXAMPLE API USAGE:

1. Process frames:
   POST /bottle-ocr/process
   Content-Type: application/json
   {
       "frames": ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."],
       "separator": "\\n---\\n"
   }

   Response:
   {
       "success": true,
       "text": "PRODUCT NAME\\n---\\nIngredients: ...",
       "segments": ["PRODUCT NAME", "Ingredients: ..."],
       "panorama": "data:image/jpeg;base64,...",
       "frame_count": 12
   }

2. Analyze frame (for live feedback):
   POST /bottle-ocr/analyze-frame
   Content-Type: application/json
   {
       "frame": "data:image/jpeg;base64,...",
       "session_id": "session_123",
       "frame_index": 0,
       "total_frames_needed": 12
   }

   Response:
   {
       "success": true,
       "status": "ok",
       "message": "Keep rotating (11 more frames suggested)",
       "feedbacks": [{"message": "...", "type": "warning", "action": "..."}],
       "progress": 8.3,
       "rotation_complete": false
   }
"""

# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == '__main__':
    """
    Run this file directly to test the Bottle OCR module standalone.
    """
    from flask import Flask
    from flask_cors import CORS
    from bottle_ocr import bottle_ocr_bp

    app = Flask(__name__, static_folder='static')
    CORS(app)

    # Configuration
    app.config['BOTTLE_OCR_USE_EASYOCR'] = False
    app.config['BOTTLE_OCR_LANGUAGES'] = ['en']

    # Register blueprint
    app.register_blueprint(bottle_ocr_bp)

    # Root redirect to capture interface
    @app.route('/')
    def index():
        return '<a href="/bottle-ocr/capture">Go to Bottle OCR Capture</a>'

    print("Starting Bottle OCR standalone server...")
    print("Open http://localhost:5000/bottle-ocr/capture in your browser")

    app.run(host='0.0.0.0', port=5000, debug=True)
