# PPE Vision Detector

A computer vision-powered safety compliance system that detects Personal Protective Equipment (PPE) in images and generates incident reports.

## 🎯 Features

- **Image Upload & Detection**: Upload images to detect PPE compliance
- **Multi-PPE Detection**: Detects helmets, safety vests, and masks
- **Real-time Annotations**: Bounding boxes with confidence scores and compliance status
- **PDF Report Generation**: Export detailed safety incident reports
- **Per-Person Analysis**: Individual compliance status for each detected person

## 🛠️ Tech Stack

| Component        | Technology                                                                                           |
| ---------------- | ---------------------------------------------------------------------------------------------------- |
| Backend          | FastAPI (Python)                                                                                     |
| Vision Model     | YOLOv8 ([Hansung-Cho/yolov8-ppe-detection](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection)) |
| Image Processing | OpenCV                                                                                               |
| PDF Generation   | ReportLab                                                                                            |

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:

```bash
git clone https://github.com/SaladStik/Hackathon.git
cd Hackathon
```

2. Install dependencies:

```bash
cd backend
pip install fastapi uvicorn opencv-python numpy ultralytics reportlab python-multipart
```

3. Download the model (already included in `backend/models/construction/best.pt`):

```bash
# Or manually download from Hugging Face:
python -c "from huggingface_hub import hf_hub_download; import shutil; path = hf_hub_download(repo_id='Hansung-Cho/yolov8-ppe-detection', filename='best.pt'); shutil.copy(path, 'models/construction/best.pt')"
```

4. Run the server:

```bash
cd backend
uvicorn app:app --reload --port 8000
```

5. Access the API at `http://localhost:8000`

## 🔌 API Endpoints

### `GET /`

Health check endpoint.

**Response:**

```json
{ "message": "PPE Vision Detector API.", "status": "running" }
```

### `GET /health`

Detailed health status.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-12-14T22:00:00"
}
```

### `POST /detect`

Detect PPE in an uploaded image.

**Request:**

- `Content-Type: multipart/form-data`
- `file`: Image file (JPEG, PNG)

**Response:**

```json
{
  "success": true,
  "total_persons": 1,
  "persons": [
    {
      "person_id": 1,
      "compliance": "compliant",
      "ppe_status": {
        "helmet": { "detected": true, "present": true, "confidence": 0.82 },
        "vest": { "detected": true, "present": true, "confidence": 0.84 },
        "mask": { "detected": true, "present": true, "confidence": 0.78 }
      },
      "summary": {
        "helmet": "DETECTED",
        "vest": "DETECTED",
        "mask": "DETECTED"
      }
    }
  ],
  "compliance_summary": {
    "compliant": 1,
    "partial": 0,
    "non_compliant": 0
  },
  "annotated_image": "<base64-encoded-image>",
  "timestamp": "2025-12-14T22:42:41.052672"
}
```

### `POST /report`

Generate a PDF safety incident report.

**Request:**

- `Content-Type: multipart/form-data`
- `file`: Image file (JPEG, PNG)

**Response:**

- PDF file download (`application/pdf`)

## 🎨 Detection Classes

The model detects the following classes:

| Class          | Type            |
| -------------- | --------------- |
| Hardhat        | PPE Present ✅  |
| NO-Hardhat     | PPE Missing ❌  |
| Safety Vest    | PPE Present ✅  |
| NO-Safety Vest | PPE Missing ❌  |
| Mask           | PPE Present ✅  |
| NO-Mask        | PPE Missing ❌  |
| Person         | Human Detection |

## 📊 Compliance Status

| Status            | Description                                    |
| ----------------- | ---------------------------------------------- |
| **Compliant**     | All required PPE detected (helmet, vest, mask) |
| **Partial**       | Some PPE detected, none explicitly missing     |
| **Non-Compliant** | One or more PPE items explicitly missing       |

## 📄 PDF Report Contents

Generated reports include:

- Report ID and timestamp
- Overall compliance status
- Per-person PPE analysis
- Detection confidence scores
- Annotated image with bounding boxes
- Recommended actions based on violations

## 🚀 Usage Example

### Using cURL:

```bash
# Detect PPE in an image
curl -X POST "http://localhost:8000/detect" \
  -F "file=@test_image.jpg"

# Generate PDF report
curl -X POST "http://localhost:8000/report" \
  -F "file=@test_image.jpg" \
  --output report.pdf
```

### Using Python:

```python
import requests

# Detect PPE
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect",
        files={"file": f}
    )
    result = response.json()
    print(f"Persons detected: {result['total_persons']}")
    print(f"Compliant: {result['compliant']}")

# Generate PDF report
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/report",
        files={"file": f}
    )
    with open("report.pdf", "wb") as pdf:
        pdf.write(response.content)
```

## 📁 Project Structure

```
Hackathon/
├── backend/
│   ├── app.py                 # FastAPI application
│   ├── pdf_generator.py       # PDF report generation
│   └── models/
│       └── construction/
│           └── best.pt        # YOLOv8 PPE detection model
├── README.md
└── requirements.txt
```

## 🔗 Model Attribution

This project uses the [YOLOv8 PPE Detection model](https://huggingface.co/Hansung-Cho/yolov8-ppe-detection) by Hansung-Cho, fine-tuned for construction site safety PPE detection.

## 📝 License

MIT License
