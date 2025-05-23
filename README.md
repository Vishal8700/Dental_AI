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

