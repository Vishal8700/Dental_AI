from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import base64
import io
import os
import logging
import numpy as np
import cv2
import pydicom
import requests
from google import generativeai as genai
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Dental X-Ray Analysis API", version="1.0.0")

ROBOFLOW_MODEL_URL = "https://detect.roboflow.com/adr/6"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ROBOFLOW_API_KEY = 'XIx4bLKjnvzxKe6bCgsT'
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Constants
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB
CONFIDENCE_THRESHOLD = 0.30  # 30%
OVERLAP_THRESHOLD = 0.50  # 50%


class Detection(BaseModel):
    class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    severity: Optional[str] = None
    location_description: Optional[str] = None


class AnalysisResponse(BaseModel):
    detections: List[Detection]
    report: str
    original_image: str
    annotated_image: str
    success: bool
    message: str
    processing_time: Optional[str] = None
    summary: Optional[dict] = None


def process_roboflow_predictions(image_bytes: bytes) -> List[Detection]:
    """
    Get predictions from Roboflow API for dental pathologies.
    """
    try:
        # Make API request
        response = requests.post(
            ROBOFLOW_MODEL_URL,
            params={
                "api_key": ROBOFLOW_API_KEY,
                "confidence": CONFIDENCE_THRESHOLD,
                "overlap": OVERLAP_THRESHOLD
            },
            data=base64.b64encode(image_bytes).decode('utf-8'),
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=30
        )
        
        logger.info(f"Roboflow status code: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"Roboflow response: {response.text}")
            response.raise_for_status()

        result = response.json()
        predictions = result.get('predictions', [])

        detections = []
        for pred in predictions:
            detection = Detection(
                class_name=pred['class'],
                confidence=float(pred['confidence']),
                x=float(pred['x']),
                y=float(pred['y']),
                width=float(pred['width']),
                height=float(pred['height'])
            )
            detections.append(detection)

        logger.info(f"Found {len(detections)} detections")
        return detections

    except requests.exceptions.RequestException as e:
        logger.error(f"Roboflow API request error: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Roboflow API error: {str(e)}")
        return []


def draw_annotations(image_array: np.ndarray, detections: List[Detection]) -> np.ndarray:

    annotated_image = image_array.copy()

    pathology_info = {
        'caries': {'color': (0, 50, 255), 'name': 'Dental Caries'},  # Bright Red
        'cavity': {'color': (0, 50, 255), 'name': 'Dental Cavity'},  # Bright Red
        'infection': {'color': (0, 165, 255), 'name': 'Infection'},  # Orange Red
        'periapical': {'color': (0, 100, 255), 'name': 'Periapical Lesion'},  # Red Orange
        'fracture': {'color': (0, 255, 100), 'name': 'Tooth Fracture'},  # Lime Green
        'root_fracture': {'color': (0, 255, 50), 'name': 'Root Fracture'},  # Green
        'implant': {'color': (255, 200, 0), 'name': 'Dental Implant'},  # Cyan
        'crown': {'color': (255, 255, 0), 'name': 'Dental Crown'},  # Yellow
        'filling': {'color': (255, 150, 0), 'name': 'Dental Filling'},  # Light Blue
        'bone_loss': {'color': (100, 0, 255), 'name': 'Bone Loss'},  # Purple
        'impacted': {'color': (255, 0, 150), 'name': 'Impacted Tooth'},  # Magenta
        'default': {'color': (0, 255, 255), 'name': 'Unknown Pathology'}  # Yellow
    }

    # Get image dimensions for scaling
    img_height, img_width = annotated_image.shape[:2]
    scale_factor = min(img_width, img_height) / 800  # Normalize

    for i, detection in enumerate(detections):
        # Get bounding box coordinates
        x_center = int(detection.x)
        y_center = int(detection.y)
        width = int(detection.width)
        height = int(detection.height)

        # Calculate corner coordinates
        x1 = max(0, int(x_center - width / 2))
        y1 = max(0, int(y_center - height / 2))
        x2 = min(img_width, int(x_center + width / 2))
        y2 = min(img_height, int(y_center + height / 2))

        # Get pathology info
        pathology_key = detection.class_name.lower().replace(' ', '_')
        pathology = pathology_info.get(pathology_key, pathology_info['default'])
        color = pathology['color']
        pathology_name = pathology['name']

        # Draw main bounding box with thicker lines
        box_thickness = max(2, int(3 * scale_factor))
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, box_thickness)

        # Draw corner markers for better visibility
        corner_size = max(8, int(12 * scale_factor))
        corner_thickness = max(2, int(3 * scale_factor))

        # Top-left corner
        cv2.line(annotated_image, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
        cv2.line(annotated_image, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)

        # Top-right corner
        cv2.line(annotated_image, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
        cv2.line(annotated_image, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)

        # Bottom-left corner
        cv2.line(annotated_image, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
        cv2.line(annotated_image, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)

        # Bottom-right corner
        cv2.line(annotated_image, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
        cv2.line(annotated_image, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)

        # Create enhanced label with ID number
        confidence_percent = detection.confidence * 100
        label_main = f"#{i + 1} {pathology_name}"
        label_confidence = f"{confidence_percent:.1f}%"

        # Font settings based on image size
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.5, 0.8 * scale_factor)
        font_thickness = max(1, int(2 * scale_factor))

        # Get text dimensions
        (main_width, main_height), _ = cv2.getTextSize(label_main, font, font_scale, font_thickness)
        (conf_width, conf_height), _ = cv2.getTextSize(label_confidence, font, font_scale * 0.8, font_thickness)

        # Calculate label box dimensions
        label_width = max(main_width, conf_width) + 20
        label_height = main_height + conf_height + 25

        # Position label (try to keep it within image bounds)
        label_x = x1
        label_y = y1 - label_height - 5

        # Adjust if label goes outside image
        if label_y < 0:
            label_y = y2 + 5
        if label_x + label_width > img_width:
            label_x = img_width - label_width

        # Draw semi-transparent label background
        overlay = annotated_image.copy()
        cv2.rectangle(overlay,
                      (label_x - 5, label_y - 5),
                      (label_x + label_width, label_y + label_height),
                      color, -1)
        cv2.addWeighted(overlay, 0.8, annotated_image, 0.2, 0, annotated_image)

        # Draw label border
        cv2.rectangle(annotated_image,
                      (label_x - 5, label_y - 5),
                      (label_x + label_width, label_y + label_height),
                      color, 2)

        # Draw main label text
        cv2.putText(annotated_image, label_main,
                    (label_x + 5, label_y + main_height + 5),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Draw confidence text (slightly smaller)
        cv2.putText(annotated_image, label_confidence,
                    (label_x + 5, label_y + main_height + conf_height + 15),
                    font, font_scale * 0.8, (200, 255, 200), font_thickness, cv2.LINE_AA)

        # Add a center dot for precise location
        center_size = max(3, int(4 * scale_factor))
        cv2.circle(annotated_image, (x_center, y_center), center_size, (255, 255, 255), -1)
        cv2.circle(annotated_image, (x_center, y_center), center_size + 1, color, 2)

    return annotated_image


def generate_diagnostic_report(detections: List[Detection], image_base64: str = None) -> str:
    """
    Generate comprehensive diagnostic report using Gemini AI with detailed pathology analysis.
    """
    try:
        if not GEMINI_API_KEY:
            return generate_comprehensive_fallback_report(detections)

        model = genai.GenerativeModel('gemini-2.0-flash')

        # Create detailed findings summary for AI analysis
        if not detections:
            findings_summary = "No pathological findings detected by automated analysis."
            detection_details = "No abnormalities identified within the detection parameters."
        else:
            # Group detections by pathology type
            pathology_groups = {}
            for detection in detections:
                pathology = detection.class_name.lower()
                if pathology not in pathology_groups:
                    pathology_groups[pathology] = []
                pathology_groups[pathology].append(detection)

            findings_summary = f"Automated analysis detected {len(detections)} pathological finding(s) across {len(pathology_groups)} categories."

            detection_details = []
            for pathology, detections_list in pathology_groups.items():
                count = len(detections_list)
                avg_confidence = sum(d.confidence for d in detections_list) / count
                max_confidence = max(d.confidence for d in detections_list)

                if count == 1:
                    detection_details.append(
                        f"- {pathology.title()}: Single occurrence detected with {avg_confidence:.1%} confidence at coordinates ({detections_list[0].x:.0f}, {detections_list[0].y:.0f})"
                    )
                else:
                    detection_details.append(
                        f"- {pathology.title()}: {count} occurrences detected with average confidence {avg_confidence:.1%} (highest: {max_confidence:.1%})"
                    )

            detection_details = "\n".join(detection_details)

        # Enhanced prompt for comprehensive dental analysis
        prompt = f"""You are an experienced dental radiologist analyzing a dental X-ray. Please provide a comprehensive diagnostic report based on the following computer-assisted findings.

AUTOMATED DETECTION SUMMARY:
{findings_summary}

DETAILED FINDINGS:
{detection_details}

Please provide a thorough diagnostic report with the following sections:

1. CLINICAL FINDINGS:
   - Detailed description of each detected pathology
   - Location assessment (specify tooth numbers/quadrants when possible)
   - Severity assessment based on confidence levels and visual characteristics

2. PATHOLOGICAL ANALYSIS:
   - Disease etiology (causes) for each finding
   - Progression stage assessment
   - Risk factors and contributing conditions
   - Differential diagnosis considerations

3. CLINICAL SIGNIFICANCE:
   - Impact on patient's oral health
   - Potential complications if left untreated
   - Urgency level of treatment needed
   - Prognosis assessment

4. TREATMENT RECOMMENDATIONS:
   - Immediate treatment priorities
   - Comprehensive treatment planning
   - Alternative treatment options
   - Preventive measures

5. FOLLOW-UP PROTOCOL:
   - Recommended monitoring schedule
   - Additional diagnostic tests needed
   - Patient education points
   - Long-term care considerations

Please format your response professionally as a formal dental radiological report, using appropriate clinical terminology while remaining clear and comprehensive. Include specific recommendations based on the detected pathologies."""

        # Generate AI response
        if image_base64:
            # Include image analysis if base64 image is provided
            response = model.generate_content([prompt, f"data:image/png;base64,{image_base64}"])
        else:
            response = model.generate_content(prompt)

        return response.text

    except Exception as e:
        logger.error(f"AI report generation error: {str(e)}")
        return generate_comprehensive_fallback_report(detections)


def generate_comprehensive_fallback_report(detections: List[Detection]) -> str:
    """
    Generate a comprehensive rule-based diagnostic report when AI is not available.
    """
    if not detections:
        return """DENTAL RADIOGRAPHIC ANALYSIS REPORT

CLINICAL FINDINGS:
No significant pathological findings detected within the automated analysis parameters. The radiographic examination appears to show normal dental and periodontal structures within the detection thresholds.

PATHOLOGICAL ANALYSIS:
No active disease processes identified by automated screening. However, clinical correlation is essential as subtle pathologies may not be detected by automated systems.

CLINICAL SIGNIFICANCE:
While no obvious pathology was detected, this does not exclude the presence of early-stage disease or conditions outside the detection parameters.

TREATMENT RECOMMENDATIONS:
- Continue routine preventive dental care
- Regular professional cleanings and examinations
- Maintain optimal oral hygiene practices
- Consider fluoride supplementation if indicated

FOLLOW-UP PROTOCOL:
- Routine dental examination in 6 months
- Annual bitewing radiographs as per clinical guidelines
- Immediate consultation if symptoms develop
- Patient education on preventive care importance

NOTE: This automated analysis should be interpreted in conjunction with clinical examination and professional dental assessment. Manual review by a qualified dental professional is recommended for comprehensive diagnosis."""

    # Define pathology information database
    pathology_database = {
        'caries': {
            'name': 'Dental Caries',
            'causes': 'Bacterial demineralization due to Streptococcus mutans and Lactobacillus, poor oral hygiene, frequent sugar/acid exposure, reduced salivary flow',
            'severity': lambda conf: 'Advanced' if conf > 0.8 else 'Moderate' if conf > 0.6 else 'Early-stage',
            'treatment': 'Restorative treatment (composite/amalgam filling), possible root canal if pulp involvement, crown if extensive destruction',
            'complications': 'Pulpal infection, periapical abscess, tooth loss, systemic infection spread',
            'prevention': 'Fluoride use, dietary modification, improved oral hygiene, regular dental visits'
        },
        'cavity': {
            'name': 'Dental Cavity',
            'causes': 'Progressive enamel and dentin destruction by acid-producing bacteria, inadequate fluoride exposure, dietary factors',
            'severity': lambda
                conf: 'Deep cavity' if conf > 0.8 else 'Moderate cavity' if conf > 0.6 else 'Surface cavity',
            'treatment': 'Direct restoration, possible indirect restoration if extensive, endodontic therapy if pulp involved',
            'complications': 'Pulp exposure, irreversible pulpitis, periapical pathology, structural tooth compromise',
            'prevention': 'Preventive sealants, fluoride therapy, dietary counseling, antimicrobial rinses'
        },
        'infection': {
            'name': 'Dental Infection',
            'causes': 'Bacterial invasion through caries, trauma, or periodontal disease, compromised immune response, poor oral hygiene',
            'severity': lambda
                conf: 'Severe infection' if conf > 0.8 else 'Moderate infection' if conf > 0.6 else 'Localized infection',
            'treatment': 'Antibiotic therapy, source control (extraction/root canal), drainage if abscess present',
            'complications': 'Facial cellulitis, Ludwig\'s angina, osteomyelitis, septicemia, airway compromise',
            'prevention': 'Prompt caries treatment, trauma management, periodontal therapy, immune system support'
        },
        'periapical': {
            'name': 'Periapical Lesion',
            'causes': 'Necrotic pulp tissue, bacterial infection spread through root canal system, inadequate endodontic treatment',
            'severity': lambda
                conf: 'Large periapical lesion' if conf > 0.8 else 'Moderate lesion' if conf > 0.6 else 'Small periapical radiolucency',
            'treatment': 'Root canal therapy, apicoectomy if conventional treatment fails, extraction if non-restorable',
            'complications': 'Cyst formation, osteomyelitis, sinus tract formation, adjacent tooth involvement',
            'prevention': 'Early caries treatment, trauma prevention, proper endodontic procedures'
        },
        'fracture': {
            'name': 'Tooth Fracture',
            'causes': 'Traumatic injury, excessive occlusal forces, weakened tooth structure, thermal stress',
            'severity': lambda
                conf: 'Complete fracture' if conf > 0.8 else 'Partial fracture' if conf > 0.6 else 'Hairline fracture',
            'treatment': 'Bonding/restoration, crown placement, root canal if pulp exposed, extraction if non-restorable',
            'complications': 'Pulp exposure, bacterial invasion, tooth sensitivity, complete tooth loss',
            'prevention': 'Mouthguard use in sports, avoid hard foods, address bruxism, regular dental check-ups'
        },
        'bone_loss': {
            'name': 'Alveolar Bone Loss',
            'causes': 'Periodontal disease, trauma, tooth loss, aging, systemic diseases, medications',
            'severity': lambda
                conf: 'Severe bone loss' if conf > 0.8 else 'Moderate bone loss' if conf > 0.6 else 'Mild bone resorption',
            'treatment': 'Periodontal therapy, bone grafting, implant placement, prosthetic rehabilitation',
            'complications': 'Tooth mobility, tooth loss, compromised chewing function, facial profile changes',
            'prevention': 'Periodontal maintenance, prompt tooth replacement, systemic health management'
        }
    }

    # Group and analyze detections
    pathology_groups = {}
    for detection in detections:
        pathology_key = detection.class_name.lower().replace(' ', '_')
        if pathology_key not in pathology_groups:
            pathology_groups[pathology_key] = []
        pathology_groups[pathology_key].append(detection)

    # Generate report sections
    report_sections = ["COMPREHENSIVE DENTAL RADIOGRAPHIC ANALYSIS REPORT\n"]

    # Clinical Findings
    report_sections.append("CLINICAL FINDINGS:")
    for pathology_key, detections_list in pathology_groups.items():
        pathology_info = pathology_database.get(pathology_key, {
            'name': pathology_key.replace('_', ' ').title(),
            'causes': 'Multiple factors may contribute to this condition',
            'severity': lambda conf: 'Significant' if conf > 0.7 else 'Moderate',
            'treatment': 'Clinical evaluation and appropriate treatment recommended',
            'complications': 'Potential for progression if untreated',
            'prevention': 'Regular dental care and good oral hygiene'
        })

        count = len(detections_list)
        avg_confidence = sum(d.confidence for d in detections_list) / count
        max_confidence = max(d.confidence for d in detections_list)
        severity = pathology_info['severity'](max_confidence)

        report_sections.append(f"\n• {pathology_info['name']}:")
        report_sections.append(f"  - {count} location(s) identified with {severity.lower()} presentation")
        report_sections.append(
            f"  - Confidence range: {min(d.confidence for d in detections_list):.1%} - {max_confidence:.1%}")
        report_sections.append(f"  - Classification: {severity} pathology based on detection parameters")

    # Pathological Analysis
    report_sections.append("\n\nPATHOLOGICAL ANALYSIS:")
    for pathology_key, detections_list in pathology_groups.items():
        pathology_info = pathology_database.get(pathology_key, pathology_database.get('caries'))
        report_sections.append(f"\n• {pathology_info['name']} - Etiology and Risk Factors:")
        report_sections.append(f"  {pathology_info['causes']}")

    # Clinical Significance
    report_sections.append("\n\nCLINICAL SIGNIFICANCE:")
    urgency_level = "HIGH" if any(max(d.confidence for d in detections_list) > 0.8 for detections_list in
                                  pathology_groups.values()) else "MODERATE"
    report_sections.append(f"Treatment urgency level: {urgency_level}")

    for pathology_key, detections_list in pathology_groups.items():
        pathology_info = pathology_database.get(pathology_key, pathology_database.get('caries'))
        max_conf = max(d.confidence for d in detections_list)
        report_sections.append(f"\n• {pathology_info['name']} complications if untreated:")
        report_sections.append(f"  {pathology_info['complications']}")

    # Treatment Recommendations
    report_sections.append("\n\nTREATMENT RECOMMENDATIONS:")
    report_sections.append("Immediate priorities:")
    for pathology_key, detections_list in pathology_groups.items():
        pathology_info = pathology_database.get(pathology_key, pathology_database.get('caries'))
        report_sections.append(f"• {pathology_info['name']}: {pathology_info['treatment']}")

    # Follow-up Protocol
    report_sections.append("\n\nFOLLOW-UP PROTOCOL:")
    if urgency_level == "HIGH":
        report_sections.append("• Immediate dental consultation recommended (within 24-48 hours)")
        report_sections.append("• Pain management if symptomatic")
        report_sections.append("• Antibiotic therapy may be indicated for infectious processes")
    else:
        report_sections.append("• Dental consultation within 1-2 weeks recommended")
        report_sections.append("• Monitor for symptom development")

    report_sections.append("• Follow-up radiographs in 6-12 months post-treatment")
    report_sections.append("• Enhanced preventive care protocol implementation")
    report_sections.append("• Patient education on identified risk factors")

    # Prevention
    report_sections.append("\n\nPREVENTIVE RECOMMENDATIONS:")
    for pathology_key, detections_list in pathology_groups.items():
        pathology_info = pathology_database.get(pathology_key, pathology_database.get('caries'))
        report_sections.append(f"• {pathology_info['name']}: {pathology_info['prevention']}")

    report_sections.append("\n\nDISCLAIMER:")
    report_sections.append("This automated analysis provides preliminary findings based on algorithmic detection. ")
    report_sections.append(
        "Clinical correlation with comprehensive dental examination is essential for definitive diagnosis ")
    report_sections.append(
        "and treatment planning. Manual review by a qualified dental professional is strongly recommended.")

    return "\n".join(report_sections)


def convert_to_image_bytes(image_array: np.ndarray) -> bytes:
    """Convert numpy array to PNG bytes."""
    success, buffer = cv2.imencode('.png', image_array)
    if not success:
        raise Exception("Failed to convert image")
    return buffer.tobytes()


def process_dicom(file_content: bytes) -> np.ndarray:
    """
    Process DICOM or RVG file content and return a numpy array.
    """
    try:
        # First try to read as DICOM
        try:
            buffer = io.BytesIO(file_content)
            ds = pydicom.dcmread(buffer)
            image_array = ds.pixel_array
            logger.info("Successfully processed as DICOM file")
        except:
            # If DICOM reading fails, try processing as RVG or regular image
            nparr = np.frombuffer(file_content, np.uint8)
            image_array = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            logger.info("Successfully processed as image file")

        if image_array is None:
            raise ValueError("Failed to decode image data")

        # Ensure proper bit depth and normalize if needed
        if image_array.dtype != np.uint8:
            # Normalize to 8-bit depth
            image_array = cv2.normalize(
                image_array,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_8U
            )

        # Convert to 3-channel if grayscale
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

        logger.info(f"Processed image shape: {image_array.shape}")
        return image_array

    except Exception as e:
        logger.error(f"Error processing image file: {str(e)}")
        raise Exception(f"Failed to process image file: {str(e)}")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_xray(file: UploadFile = File(...)):
    start_time = time.time()

    try:
        # Validate file type
        allowed_extensions = ('.dcm', '.rvg', '.png', '.jpg', '.jpeg')
        if not file.filename.lower().endswith(allowed_extensions):
            raise HTTPException(
                status_code=400,
                detail=f"File must be one of: {', '.join(allowed_extensions)}"
            )

        # Read file content
        file_content = await file.read()

        # Check file size
        if len(file_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum limit of {MAX_UPLOAD_SIZE / 1024 / 1024}MB"
            )

        # Process DICOM/RVG file
        image_array = process_dicom(file_content)

        # Convert to PNG for API calls
        image_bytes = convert_to_image_bytes(image_array)
        original_image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Get predictions from Roboflow
        detections = process_roboflow_predictions(image_bytes)

        # Create annotated image
        annotated_array = draw_annotations(image_array, detections)
        annotated_bytes = convert_to_image_bytes(annotated_array)
        annotated_image_base64 = base64.b64encode(annotated_bytes).decode('utf-8')

        # Generate diagnostic report with image data for enhanced analysis
        report = generate_diagnostic_report(detections, original_image_base64)

        processing_time = f"{time.time() - start_time:.2f}s"

        # Create summary statistics
        summary = {
            "total_detections": len(detections),
            "pathology_types": len(set(d.class_name for d in detections)),
            "high_confidence_detections": len([d for d in detections if d.confidence > 0.8]),
            "medium_confidence_detections": len([d for d in detections if 0.6 <= d.confidence <= 0.8]),
            "low_confidence_detections": len([d for d in detections if d.confidence < 0.6]),
            "average_confidence": sum(d.confidence for d in detections) / len(detections) if detections else 0,
            "pathology_distribution": {
                pathology: len([d for d in detections if d.class_name == pathology])
                for pathology in set(d.class_name for d in detections)
            } if detections else {}
        }

        return AnalysisResponse(
            detections=detections,
            report=report,
            original_image=original_image_base64,
            annotated_image=annotated_image_base64,
            success=True,
            message=f"Analysis completed successfully. Found {len(detections)} pathological findings.",
            processing_time=processing_time,
            summary=summary
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return AnalysisResponse(
            detections=[],
            report=f"Analysis failed: {str(e)}",
            original_image="",
            annotated_image="",
            success=False,
            message=f"Analysis failed: {str(e)}",
            processing_time=f"{time.time() - start_time:.2f}s"
        )


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Dental X-Ray Analysis API",
        "version": "1.0.0",
        "timestamp": time.time()
    }


@app.get("/")
async def root():
    return {
        "message": "Dental X-Ray Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/analyze": "POST - Upload and analyze dental X-ray images",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)