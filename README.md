# 📝 AI Based Answer Script Evaluation System

> An intelligent system that automates the evaluation of handwritten or typed answer scripts using AI and image processing techniques.

---

## 📌 Project Info

| Field | Details |
|---|---|
| **College** | St. Martin's Engineering College, Secunderabad |
| **Department** | Computer Science & Engineering (AI & ML) |
| **Degree** | B.Tech – CSE (AI & ML), 2023–2027 |
| **Guide** | T. Kanakamma, Professor, Dept. of CSE |
| **Type** | Industrial Oriented Mini Project |

### 👨‍💻 Team Members

| Name | Roll Number |
|---|---|
| B. Praise Nancy | 23K81A66D4 |
| G. Manogna | 23K81A66D9 |
| M. Pavithra | 23K81A66J3 |

---

## 📖 About the Project

The **AI Based Answer Script Evaluation System** is designed to automate the process of evaluating student answer scripts. Traditional manual evaluation is time-consuming, prone to bias, and inconsistent. This system leverages AI and image processing to read, understand, and grade answer scripts accurately and efficiently.

The system can:
- Accept scanned or photographed answer scripts as input
- Extract text using OCR (Optical Character Recognition)
- Compare student answers against a model answer using NLP techniques
- Assign scores based on semantic similarity and keyword matching
- Generate detailed evaluation reports

---

## 🎯 Objectives

- Automate the grading process to reduce examiner workload
- Ensure consistent and unbiased evaluation
- Support both handwritten and typed answer scripts
- Provide detailed feedback to students on their answers
- Reduce evaluation turnaround time significantly

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | HTML, CSS, JavaScript / React |
| **Backend** | Python, Flask / FastAPI |
| **AI / ML** | TensorFlow / PyTorch, Hugging Face Transformers |
| **OCR** | Tesseract OCR / Google Vision API |
| **NLP** | BERT, Sentence Transformers, spaCy |
| **Database** | MySQL / PostgreSQL |
| **Image Processing** | OpenCV, PIL |

---

## 🔄 System Workflow

```
Input Answer Script (Image/PDF)
        ↓
  Image Preprocessing (OpenCV)
        ↓
  Text Extraction (OCR - Tesseract)
        ↓
  Text Cleaning & Normalization (NLP)
        ↓
  Semantic Comparison with Model Answer
        ↓
  Score Generation & Feedback
        ↓
  Evaluation Report (PDF/Dashboard)
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip
- Tesseract OCR installed on your system
- Node.js (if using React frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-answer-evaluation.git
cd ai-answer-evaluation

# Create virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Install Tesseract OCR

- **Windows:** Download installer from https://github.com/UB-Mannheim/tesseract/wiki
- **Linux:** `sudo apt install tesseract-ocr`
- **Mac:** `brew install tesseract`

### Run the Application

```bash
# Start backend
python app.py

# Frontend (if separate)
cd frontend
npm install
npm start
```

Open your browser and go to `http://localhost:5000`

---

## 📁 Project Structure

```
ai-answer-evaluation/
│
├── app.py                   # Main Flask/FastAPI application
├── requirements.txt         # Python dependencies
├── README.md
│
├── model/
│   ├── evaluator.py         # Core evaluation logic
│   ├── ocr.py               # OCR text extraction
│   └── similarity.py        # NLP similarity scoring
│
├── preprocessing/
│   ├── image_cleaner.py     # Image enhancement
│   └── text_normalizer.py   # Text cleaning
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── static/                  # Static files
├── templates/               # HTML templates
└── uploads/                 # Uploaded answer scripts
```

---

## 📊 Features

- ✅ Upload answer scripts (JPG, PNG, PDF)
- ✅ Automatic text extraction via OCR
- ✅ AI-powered semantic answer matching
- ✅ Keyword-based scoring
- ✅ Multi-subject support
- ✅ Detailed score breakdown per question
- ✅ Downloadable evaluation report
- ✅ Admin dashboard for managing question papers and model answers

---

## 🧪 How to Use

1. **Admin/Teacher:** Upload the question paper and model answers
2. **Upload Script:** Submit the student's answer script (scanned image or PDF)
3. **Processing:** The system extracts text and evaluates each answer
4. **Results:** View the score, feedback, and download the evaluation report

---

## 📈 Future Enhancements

- Support for multiple languages
- Handwriting recognition with deeper CNN models
- Integration with LMS platforms (Moodle, Google Classroom)
- Mobile app for on-the-go evaluation
- Real-time evaluation dashboard for institutions

---

## 📜 License

This project is developed for academic purposes at St. Martin's Engineering College under the B.Tech program in CSE (AI & ML).

---

## 🙏 Acknowledgements

We express our sincere gratitude to:
- **T. Kanakamma** – Project Guide & Professor, Dept. of CSE (AI & ML)
- **Dr. Gowtham Mamidisetti** – Head of the Department, CSE (AI & ML)
- **Dr. M. Sreenivasa Rao** – Principal, St. Martin's Engineering College
- **Dr. Kasa Ravindra** – Director, St. Martin's Engineering College

---

*St. Martin's Engineering College | UGC Autonomous | Affiliated to JNTUH | NBA & NAAC A+ Accredited*
