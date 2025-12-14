from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os
from pdf_generator import (
    generate_pdf_report,
)  # Assume this is a custom module for PDF generation
import base64
import tempfile
import uuid

app = FastAPI(title="PPE Vision Detector API")

# cors settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def calculate_iou(box1, box2):
    """calculate intersection over union of two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # calculate the intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = max(x2_1, x2_2)
    y2_i = max(y2_1, y2_2)

    if x2_1 < x1_1 or y2_i < y1_1:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # calculate the union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def box_overlap_percentage(ppe_box, person_box):
    """Calculate the percentage of the PPE box that overlaps with the person box"""
    x1_p, y1_p, x2_p, y2_p = ppe_box
    x1_per, y1_per, x2_per, y2_per = person_box

    # Calculate intersection
    x1_i = max(x1_p, x1_per)
    y1_i = max(y1_p, y1_per)
    x2_i = min(x2_p, x2_per)
    y2_i = min(y2_p, y2_per)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    ppe_area = (x2_p - x1_p) * (y2_p - y1_p)

    return intersection_area / ppe_area if ppe_area > 0 else 0.0


def is_ppe_near_person(ppe_box, person_box, margin_ratio=0.3):
    """Check if PPE is within or near a person's bounding box."""
    x1_per, y1_per, x2_per, y2_per = person_box
    person_width = x2_per - x1_per
    person_height = y2_per - y1_per

    # expand person box by margin
    margin_x = person_width * margin_ratio
    margin_y = person_height * margin_ratio

    expanded_box = [
        x1_per - margin_x,
        y1_per - margin_y,
        x2_per + margin_x,
        y2_per + margin_y,
    ]

    # Check if PPE center is within expanded person box
    ppe_center_x = (ppe_box[0] + ppe_box[2]) / 2
    ppe_center_y = (ppe_box[1] + ppe_box[3]) / 2

    return (
        expanded_box[0] <= ppe_center_x <= expanded_box[2]
        and expanded_box[1] <= ppe_center_y <= expanded_box[3]
    )


class PPEDetector:

    def __init__(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "models", "construction", "best.pt"
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        self.model = YOLO(model_path)
        print(f"Loaded model from {model_path}")
        self.class_names = self.model.names
        print(f"PPE Model Classes: {self.model.names}")

        # Positive PPE Detections
        self.positive_classes = {
            "hardhat": "helmet",
            "safety vest": "vest",
            "mask": "mask",
        }
        self.negative_classes = {
            "no-hardhat": "helmet",
            "no-safety vest": "vest",
            "no-mask": "mask",
        }

        # Colors for bounding boxes
        self.colors = {
            "compliant": (0, 200, 0),  # Green
            "partial": (0, 165, 255),  # Orange some ppe
            "non_compliant": (0, 0, 255),  # Red
            "ppe_present": (0, 255, 0),  # Green
            "ppe_missing": (0, 0, 255),  # Red
        }

    def detect(self, image: np.ndarray, confidence_threshold: float = 0.25):
        """Run PPE detection on an image."""
        results = self.model(image, conf=confidence_threshold, verbose=False)
        persons = []  # list of detected persons
        ppe_items = []  # list of detected ppe items
        missing_ppe = []  # list of detected missing ppe items

        # first pass: collect all detections
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls_id = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                class_name = self.class_names[cls_id]
                class_lower = class_name.lower()

                detection = {
                    "class": class_name,
                    "class_lower": class_lower,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                }
                if class_lower == "person":
                    persons.append(detection)
                elif class_lower in self.positive_classes:
                    detection["ppe_type"] = self.positive_classes[class_lower]
                    detection["is_present"] = True
                    ppe_items.append(detection)
                elif class_lower in self.negative_classes:
                    detection["ppe_type"] = self.negative_classes[class_lower]
                    detection["is_present"] = False
                    missing_ppe.append(detection)

            # if no persons detected but ppe detected, add virtual perso
        if len(persons) == 0 and (len(ppe_items) > 0 or len(missing_ppe) > 0):
            # assume single person and just use the image bounds
            h, w = image.shape[:2]
            persons.append(
                {
                    "class": "person",
                    "class_lower": "person",
                    "confidence": 1.0,
                    "bbox": [0, 0, w, h],
                    "virtual": True,
                }
            )

        # second pass: associate ppe to persons and determine compliance
        person_ppe_status = []
        for i, person in enumerate(persons):
            person_box = person["bbox"]
            # init ppe status
            ppe_status = {
                "helmet": {"detected": False, "present": None, "confidence": 0.0},
                "vest": {"detected": False, "present": None, "confidence": 0.0},
                "mask": {"detected": False, "present": None, "confidence": 0.0},
            }

            # check positive ppe items
            for ppe in ppe_items:
                if is_ppe_near_person(ppe["bbox"], person_box):
                    ppe_type = ppe["ppe_type"]

                    if (
                        not ppe_status[ppe_type]["detected"]
                        or ppe["confidence"] > ppe_status[ppe_type]["confidence"]
                    ):
                        ppe_status[ppe_type] = {
                            "detected": True,
                            "present": True,
                            "confidence": ppe["confidence"],
                            "bbox": ppe["bbox"],
                        }
            # check negative ppe items
            for ppe in missing_ppe:
                if is_ppe_near_person(ppe["bbox"], person_box):
                    ppe_type = ppe["ppe_type"]
                    # only update if not already present
                    if (
                        not ppe_status[ppe_type]["detected"]
                        or not ppe_status[ppe_type]["present"]
                    ):
                        if (
                            ppe_status[ppe_type]["detected"]
                            or ppe["confidence"] > ppe_status[ppe_type]["confidence"]
                        ):
                            ppe_status[ppe_type] = {
                                "detected": True,
                                "present": False,
                                "confidence": ppe["confidence"],
                                "bbox": ppe["bbox"],
                            }
            # determine compliance
            detected_count = sum(1 for v in ppe_status.values() if v["present"] is True)
            missing_count = sum(1 for v in ppe_status.values() if v["present"] is False)

            if missing_count > 0:
                compliance = "non_compliant"
            elif detected_count == 3:
                compliance = "compliant"
            elif detected_count > 0:
                compliance = "partial"
            else:
                compliance = "unknown"

            person_ppe_status.append(
                {
                    "person_id": i + 1,
                    "bbox": person_box,
                    "confidence": person["confidence"],
                    "virtual": person.get("virtual", False),
                    "ppe_status": ppe_status,
                    "compliance": compliance,
                    "detected_count": detected_count,
                    "missing_count": missing_count,
                }
            )

            # draw annoatations
            annoated_image = self._draw_annotations(
                image, person_ppe_status, ppe_items, missing_ppe
            )

            return {
                "persons": person_ppe_status,
                "total_persons": len(person_ppe_status),
                "annotated_image": annoated_image,
            }

    def _draw_annotations(self, image, person_ppe_status, ppe_items, missing_ppe):
        """Draw bounding boxes and labels on image."""
        annotated = image.copy()

        # draw each persons box with compliance colour
        for person in person_ppe_status:
            if person.get("virtual"):
                continue  # skip virtual persons

            bbox = person["bbox"]
            x1, y1, x2, y2 = bbox
            compliance = person["compliance"]

            # choose color based on compliance
            if compliance == "compliant":
                color = self.colors["compliant"]
                status_text = "COMPLIANT"
            elif compliance == "partial":
                color = self.colors["partial"]
                status_text = "PARTIAL"
            else:
                color = self.colors["non_compliant"]
                status_text = "NON-COMPLIANT"

            # draw person box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # draw person label with ppe summary
            label = f"Person {person['person_id']}: {status_text}"
            ppe_summary = []

            for ppe_type, status in person["ppe_status"].items():
                if status["present"] is True:
                    ppe_summary.append(f"{ppe_type[0].upper()}:Y")
                elif status["present"] is False:
                    ppe_summary.append(f"{ppe_type[0].upper()}:N")
                else:
                    ppe_summary.append(f"{ppe_type[0].upper()}:?")

            ppe_text = " | ".join(ppe_summary)

            # draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                annotated,
                (x1, y1 - 50),
                (x1 + max(label_size[0], 200) + 10, y1),
                color,
                -1,
            )

            # draw text
            text_color = (255, 255, 255)  # White text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                text_color,
                2,
            )
            cv2.putText(
                annotated,
                ppe_text,
                (x1 + 5, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
            )

            # draw PPE item boxes
            for ppe in ppe_items:
                x1, y1, x2, y2 = ppe["bbox"]
                cv2.rectangle(
                    annotated, (x1, y1), (x2, y2), self.colors["ppe_present"], 2
                )
                label = f"{ppe['class']} ({ppe['confidence']:.2f})"
                cv2.putText(
                    annotated,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    self.colors["ppe_present"],
                    2,
                )

                # draw missing PPE boxes
                for ppe in missing_ppe:
                    x1, y1, x2, y2 = ppe["bbox"]
                    cv2.rectangle(
                        annotated, (x1, y1), (x2, y2), self.colors["ppe_missing"], 2
                    )
                    label = f"{ppe['class']} ({ppe['confidence']:.2f})"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        self.colors["ppe_missing"],
                        2,
                    )
        return annotated


# global detector instance
detector = None


def get_detector():
    global detector
    if detector is None:
        detector = PPEDetector()
    return detector


@app.on_event("startup")
async def startup_event():
    """Initialize the PPE detector on startup."""
    try:
        get_detector()
        print("PPE Detector initialized successfully.")
    except Exception as e:
        print(e)


@app.get("/")
async def root():
    return {"message": "PPE Vision Detector API.", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "timestamp": datetime.utcnow(),
    }


@app.post("/detect")
async def detect_ppe(file: UploadFile = File(...)):
    """Endpoint to detect PPE in an uploaded image."""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400, content={"error": "Invalid image file."}
            )

        det = get_detector()
        results = det.detect(image)
        max_width = 1280
        # resize the image if its too big
        annotated = results["annotated_image"]
        h, w = annotated.shape[:2]
        if w > max_width / w:
            scale = max_width / w
            annotated = cv2.resize(
                annotated,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        # encode annotated image to base64
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        _, img_encoded = cv2.imencode(".jpg", annotated, encode_params)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")

        # build response with per person data
        persons_data = []
        total_compliant = 0
        total_partial = 0
        total_non_compliant = 0

        aggregate_ppe = {
            "helmet": {"detected": False, "present": None, "confidence": 0.0},
            "vest": {"detected": False, "present": None, "confidence": 0.0},
            "mask": {"detected": False, "present": None, "confidence": 0.0},
        }

        for person in results["persons"]:
            person_data = {
                "person_id": person["person_id"],
                "compliance": person["compliance"],
                "ppe_status": {},
                "summary": {},
            }

            for ppe_type, status in person["ppe_status"].items():
                person_data["ppe_status"][ppe_type] = {
                    "detected": status["detected"],
                    "present": status["present"],
                    "confidence": (
                        round(status["confidence"], 3) if status["confidence"] else None
                    ),
                }

                if status["present"] is True:
                    person_data["summary"][ppe_type] = "DETECTED"
                elif status["present"] is False:
                    person_data["summary"][ppe_type] = "MISSING"
                else:
                    person_data["summary"][ppe_type] = "NOT DETECTED"

                # update aggregate ppe status
                if (
                    status["detected"]
                    and status["confidence"] > aggregate_ppe[ppe_type]["confidence"]
                ):
                    aggregate_ppe[ppe_type] = status
            persons_data.append(person_data)

            if person["compliance"] == "compliant":
                total_compliant += 1
            elif person["compliance"] == "non_compliant":
                total_non_compliant += 1
            else:
                total_partial += 1

        summary = {}
        ppe_status = {}
        for ppe_type, status in aggregate_ppe.items():
            ppe_status[ppe_type] = {
                "detected": status["detected"],
                "present": status["present"],
                "confidence": (
                    round(status["confidence"], 3) if status["confidence"] else None
                ),
            }
            if status["present"] is True:
                summary[ppe_type] = "DETECTED"
            elif status["present"] is False:
                summary[ppe_type] = "MISSING"
            else:
                summary[ppe_type] = "NOT DETECTED"

        response = {
            "success": True,
            "total_persons": results["total_persons"],
            "persons": persons_data,
            "compliance_summary": {
                "compliant": total_compliant,
                "partial": total_partial,
                "non_compliant": total_non_compliant,
            },
            "ppe_status": ppe_status,
            "summary": summary,
            "compliant": total_non_compliant == 0 and total_partial == 0,
            "annotated_image": img_base64,
            "timestamp": datetime.utcnow().isoformat(),
        }

        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error.", "details": str(e)},
        )


@app.post("/report")
async def generate_report(file: UploadFile = File(...)):
    """Generate PDF incident Report with per-person PPE status"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse(
                status_code=400, content={"error": "Invalid image file."}
            )

        det = get_detector()

        results = det.detect(image)

        # prepare detection data for PDF
        detection_data = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "total_persons": results["total_persons"],
            "persons": results["persons"],
        }

        # save annotated image to temp file
        temp_dir = tempfile.gettempdir()
        detection_id = str(uuid.uuid4())[:8]
        annotated_path = os.path.join(temp_dir, f"annotated_{detection_id}.jpg")
        cv2.imwrite(annotated_path, results["annotated_image"])

        # generate PDF report
        pdf_path = os.path.join(temp_dir, f"ppe_report_{detection_id}.pdf")
        generate_pdf_report(
            output_path=pdf_path,
            detection_id=detection_id,
            annotated_image_path=annotated_path,
            detection_data=detection_data,
        )

        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"ppe_report_{detection_id}.pdf",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error.", "details": str(e)},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
