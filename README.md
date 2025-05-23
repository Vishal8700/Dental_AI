# DentalAI – AI-Powered Dental X-Ray Analysis

**DentalAI** is a full-stack application that leverages cutting-edge AI technologies to automate the analysis and diagnosis of dental X-rays. It offers a powerful backend for image processing and diagnostic report generation, and a modern, interactive frontend for clinicians and dental professionals.

---

## 🌐 Live Demo
> (Add your deployment URL here if hosted)

---

## 🧠 Features

### 🔬 Backend (FastAPI)
- **Automated Pathology Detection** via Roboflow AI:
  - Dental Caries, Cavities, Infections, Periapical Lesions
  - Tooth Fractures, Bone Loss, Implants, Crowns & Fillings
- **Gemini AI Diagnostic Reports**:
  - Clinical Findings, Significance, and Treatment Recommendations
- **DICOM + Image Processing**:
  - Handles DICOM, RVG, PNG, JPG, JPEG
  - Enhances images, annotates pathologies with bounding boxes and labels
- **Robust API**:
  - RESTful endpoints for file analysis, health check, and Swagger UI

### 💻 Frontend (React.js)
- **Theme Toggle**:
  - Light/Dark mode with system preference detection
- **Drag & Drop Upload**:
  - Intuitive and modern interface for uploading dental X-rays
- **Interactive Dashboard**:
  - Detection metrics, risk categories, pathology distribution
- **AI-Powered Reports**:
  - Downloadable diagnostic summaries with treatment insights
- **Real-Time Feedback**:
  - Toast notifications and performance metrics

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **AI Services**: Roboflow (Object Detection), Google Gemini (Text Generation)
- **Libraries**: OpenCV, NumPy, Pydicom, Requests, Pydantic

### Frontend
- **Framework**: React.js
- **Icons**: Lucide React
- **Styling**: CSS Modules / Custom CSS
- **State Management**: React useState & useEffect
- **UX**: Responsive, accessible, animated

---

## 📁 Project Structure

dental-ai/
├── backend/
│ ├── main.py
│ ├── utils/...
│ ├── requirements.txt
│ └── .env
├── frontend/
│ ├── public/
│ ├── src/
│ │ ├── components/
│ │ ├── pages/
│ │ ├── App.js
│ │ └── index.js
│ └── package.json



---

## ⚙️ Backend Setup

### 🔐 Environment Variables
Create a `.env` file in the backend root:

```env
ROBOFLOW_API_KEY=your_roboflow_api_key
GEMINI_API_KEY=your_gemini_api_key
```
--- 
##📦 Install Dependencies

cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

## 🚀 Start API Server

uvicorn main:app --host 0.0.0.0 --port 8000

| Method | Endpoint   | Description                       |
| ------ | ---------- | --------------------------------- |
| GET    | `/`        | API info and welcome message      |
| GET    | `/health`  | Server health check               |
| GET    | `/docs`    | Swagger UI (interactive API docs) |
| POST   | `/analyze` | Upload and analyze dental X-rays  |


{
  "detections": [...],
  "report": "string",
  "original_image": "base64",
  "annotated_image": "base64",
  "success": true,
  "message": "Success",
  "processing_time": "2.3s",
  "summary": {
    "total_detections": 5,
    "pathology_types": 3,
    "high_confidence_detections": 2,
    ...
  }
}
## 🌐 Frontend Setup

#📦 Install & Run
```
cd frontend
npm install
npm run dev
```
# 🔑 Configuration
Update the backend URL in your API utility or .env file:
```
VITE_API_URL=http://localhost:8000

```
#📂 Key Components
Header: Branding and theme toggle

Upload: Drag & drop file upload with preview

Results: Annotated images, charts, and risk levels

Notifications: Toasts for success/errors

Reports: AI-generated recommendations and downloadable reports

#🔐 Security & Validation
CORS middleware enabled (FastAPI)

File size limit (10MB)

Input format restrictions

API key protected inference endpoints

#⚠️ Limitations
File size max: 10MB

Supported file types: DICOM, RVG, PNG, JPG, JPEG

Requires internet for Roboflow/Gemini AI APIs

#📌 Future Enhancements
 Batch upload and analysis

 Historical comparison view

 Multi-language support for reports

 PDF report generation

 Dental practice system integration

#📜 License
MIT License (or specify your license)

#👥 Contributors
Vishal kumar – Developer

[Contributors list if applicable]

#🆘 Support
For issues or contributions, feel free to open an Issue or submit a PR.

Contact: [vishalkumar09837@gmail.com]
