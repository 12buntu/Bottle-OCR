# Bottle OCR Module

A Flask-compatible module for recognizing text from round bottles using video frame capture and stitching.

## Features

- **360-degree text capture**: Captures multiple frames as user rotates bottle, stitches into panorama
- **Real-time feedback**: Provides live guidance ("turn left", "move away from camera", etc.)
- **Segment detection**: Automatically separates text by visual regions with configurable delimiters
- **Web interface**: Browser-based capture using device camera
- **Flask integration**: Drop-in blueprint for existing Flask applications

## Installation

### 1. Install system dependencies

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr libtesseract-dev

# macOS
brew install tesseract

# Windows: Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Integration with eln_web_backend

Add to your `app.py`:

```python
# Add import
from bottle_ocr import bottle_ocr_bp

# After app = Flask(__name__), add:
app.config['BOTTLE_OCR_USE_EASYOCR'] = False  # or True for EasyOCR
app.config['BOTTLE_OCR_LANGUAGES'] = ['en']

# Register blueprint
app.register_blueprint(bottle_ocr_bp)
```

## Usage

### Web Interface

Navigate to `/bottle-ocr/capture` in your browser to access the capture interface.

1. Allow camera access when prompted
2. Position the bottle in the frame (guide lines shown)
3. Click "Start Capture"
4. Slowly rotate the bottle 360 degrees
5. Click "Process" when done
6. View/copy extracted text

### API Endpoints

#### POST `/bottle-ocr/process`

Process multiple captured frames.

**Request:**
```json
{
    "frames": ["data:image/jpeg;base64,...", "..."],
    "separator": "\n---\n"
}
```

**Response:**
```json
{
    "success": true,
    "text": "PRODUCT NAME\n---\nIngredients: water, sodium...",
    "segments": ["PRODUCT NAME", "Ingredients: water, sodium..."],
    "panorama": "data:image/jpeg;base64,...",
    "frame_count": 12
}
```

#### POST `/bottle-ocr/analyze-frame`

Analyze a frame for live feedback.

**Request:**
```json
{
    "frame": "data:image/jpeg;base64,...",
    "session_id": "session_123",
    "frame_index": 0,
    "total_frames_needed": 12
}
```

**Response:**
```json
{
    "success": true,
    "status": "ok",
    "message": "Keep rotating (11 more frames suggested)",
    "feedbacks": [
        {"message": "Low contrast", "type": "warning", "action": "Improve lighting"}
    ],
    "metrics": {
        "blur_score": 150.5,
        "brightness": 128.0,
        "rotation_covered": 45.0
    },
    "progress": 8.3,
    "rotation_complete": false
}
```

#### POST `/bottle-ocr/process-single`

Process a single frame (for testing).

#### POST `/bottle-ocr/reset-session`

Reset capture session state.

#### GET `/bottle-ocr/health`

Health check endpoint.

## Configuration

| Config Key | Default | Description |
|------------|---------|-------------|
| `BOTTLE_OCR_USE_EASYOCR` | `False` | Use EasyOCR instead of Tesseract |
| `BOTTLE_OCR_LANGUAGES` | `['en']` | Languages to recognize |

## Standalone Testing

Run the module standalone for testing:

```bash
python integration_example.py
```

Then open http://localhost:5000/bottle-ocr/capture

## Project Structure

```
Bottle-OCR/
├── bottle_ocr/
│   ├── __init__.py      # Package exports
│   ├── processor.py     # Core OCR and stitching logic
│   ├── blueprint.py     # Flask blueprint with endpoints
│   └── feedback.py      # Real-time capture feedback
├── static/
│   └── bottle_capture.html  # Web capture interface
├── integration_example.py   # Integration guide and standalone runner
├── requirements.txt         # Python dependencies
└── README.md
```

## OCR Engines

### Tesseract (default)
- Fast, lightweight
- Good for clear, printed text
- Requires system installation

### EasyOCR (optional)
- Better for complex fonts and curved text
- Slower, larger model download on first use
- Set `BOTTLE_OCR_USE_EASYOCR = True`

## Troubleshooting

### "No text detected"
- Ensure good lighting (not too dark or bright)
- Hold bottle steady while capturing
- Rotate slowly (500ms+ between frames)
- Check that text is facing the camera

### "Stitching failed"
- Capture more frames (increase max frames)
- Rotate more slowly
- Ensure there's enough visual overlap between frames

### Camera not working
- Check browser permissions
- HTTPS is required for camera access (except localhost)
- Try a different browser

## License

MIT
