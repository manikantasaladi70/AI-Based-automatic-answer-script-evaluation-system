"""
SMEC — AI Paper Evaluation System
Full backend with SQLite + PDF support + sub-question (2a/2b) handling.
"""

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import re
import io
import os
import base64
import json
import requests
from PIL import Image, ImageEnhance, ImageFilter
import torch
import nltk
from sentence_transformers import SentenceTransformer, util
from skimage.feature import canny

# PaddleOCR — better handwriting recognition than EasyOCR, fully free & local
try:
    from paddleocr import PaddleOCR
    _paddle = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    PADDLE_AVAILABLE = True
    print("PaddleOCR loaded.")
except Exception as _e:
    PADDLE_AVAILABLE = False
    print(f"PaddleOCR not available ({_e}), using EasyOCR fallback.")
    import easyocr as _easyocr
    _easy_reader = _easyocr.Reader(['en'])
from datetime import datetime
import uuid

# PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("WARNING: PyMuPDF not installed. PDF support disabled. Run: pip install pymupdf")

# =========================================================
# CONFIG
# =========================================================
SCORE_PER_QUESTION     = 10   # bumped to 10 for longer academic answers
ANTHROPIC_API_KEY      = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY         = os.environ.get("GEMINI_API_KEY", "")
FULL_MARKS_THRESHOLD   = 0.60
PARTIAL_HIGH_THRESHOLD = 0.42
PARTIAL_LOW_THRESHOLD  = 0.25
AI_GRADER_LOW          = 0.38
AI_GRADER_HIGH         = 0.65

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =========================================================
# FLASK + DATABASE
# =========================================================
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'smec.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "smec-secret-change-in-production")
db = SQLAlchemy(app)


# =========================================================
# DATABASE MODELS
# =========================================================
class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    uid           = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    username      = db.Column(db.String(80),  nullable=False, unique=True)
    email         = db.Column(db.String(120), nullable=False, unique=True)
    password_hash = db.Column(db.String(256), nullable=False)
    role          = db.Column(db.String(20),  default="teacher")
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)
    is_active     = db.Column(db.Boolean,     default=True)
    evaluations   = db.relationship("Evaluation", backref="evaluator", lazy=True)

    def set_password(self, pw):   self.password_hash = generate_password_hash(pw)
    def check_password(self, pw): return check_password_hash(self.password_hash, pw)
    def to_dict(self):
        return {"id": self.id, "uid": self.uid, "username": self.username,
                "email": self.email, "role": self.role,
                "createdAt": self.created_at.isoformat(), "isActive": self.is_active}


class Student(db.Model):
    __tablename__ = "students"
    id         = db.Column(db.Integer, primary_key=True)
    uid        = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    name       = db.Column(db.String(120), nullable=False)
    roll_no    = db.Column(db.String(40),  nullable=False, unique=True)
    email      = db.Column(db.String(120))
    class_name = db.Column(db.String(40))
    section    = db.Column(db.String(10))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    evaluations= db.relationship("Evaluation", backref="student", lazy=True)

    def to_dict(self):
        return {"id": self.id, "uid": self.uid, "name": self.name,
                "rollNo": self.roll_no, "email": self.email,
                "class": self.class_name, "section": self.section,
                "createdAt": self.created_at.isoformat()}


class AnswerKey(db.Model):
    __tablename__ = "answer_keys"
    id          = db.Column(db.Integer, primary_key=True)
    uid         = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    subject     = db.Column(db.String(80),  nullable=False)
    exam_name   = db.Column(db.String(120))
    questions   = db.Column(db.Text, nullable=False)
    total_marks = db.Column(db.Integer)
    created_by  = db.Column(db.Integer, db.ForeignKey("users.id"))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    evaluations = db.relationship("Evaluation", backref="answer_key", lazy=True)

    def get_questions(self):         return json.loads(self.questions)
    def set_questions(self, q_list):
        self.questions   = json.dumps(q_list)
        self.total_marks = len(q_list) * SCORE_PER_QUESTION

    def to_dict(self):
        return {"id": self.id, "uid": self.uid, "subject": self.subject,
                "examName": self.exam_name, "questions": self.get_questions(),
                "totalMarks": self.total_marks, "createdAt": self.created_at.isoformat()}


class Evaluation(db.Model):
    __tablename__   = "evaluations"
    id              = db.Column(db.Integer, primary_key=True)
    uid             = db.Column(db.String(36), unique=True, default=lambda: str(uuid.uuid4()))
    student_id      = db.Column(db.Integer, db.ForeignKey("students.id"))
    answer_key_id   = db.Column(db.Integer, db.ForeignKey("answer_keys.id"))
    evaluator_id    = db.Column(db.Integer, db.ForeignKey("users.id"))
    subject         = db.Column(db.String(80))
    total_marks     = db.Column(db.Integer)
    obtained_marks  = db.Column(db.Integer)
    percentage      = db.Column(db.Float)
    remark          = db.Column(db.String(40))
    ai_grader_used  = db.Column(db.Boolean, default=False)
    evaluated_at    = db.Column(db.DateTime, default=datetime.utcnow)
    answers         = db.relationship("EvalAnswer", backref="evaluation",
                                      lazy=True, cascade="all, delete-orphan")

    def to_dict(self, include_answers=True):
        d = {"id": self.id, "uid": self.uid, "studentId": self.student_id,
             "studentName": self.student.name    if self.student else None,
             "rollNo":      self.student.roll_no if self.student else None,
             "subject": self.subject, "totalMarks": self.total_marks,
             "obtainedMarks": self.obtained_marks, "percentage": self.percentage,
             "remark": self.remark, "aiGraderUsed": self.ai_grader_used,
             "evaluatedAt": self.evaluated_at.isoformat()}
        if include_answers:
            d["answers"] = [a.to_dict() for a in self.answers]
        return d


class EvalAnswer(db.Model):
    __tablename__   = "eval_answers"
    id              = db.Column(db.Integer, primary_key=True)
    evaluation_id   = db.Column(db.Integer, db.ForeignKey("evaluations.id"), nullable=False)
    question_no     = db.Column(db.String(10), nullable=False)   # "1", "2a", "3b" etc.
    key_answer      = db.Column(db.Text)
    student_answer  = db.Column(db.Text)
    marks           = db.Column(db.Integer)
    max_marks       = db.Column(db.Integer)
    feedback        = db.Column(db.Text)
    score_pct       = db.Column(db.Integer)

    def to_dict(self):
        return {"question": self.question_no, "keyAnswer": self.key_answer,
                "studentAnswer": self.student_answer, "marks": self.marks,
                "maxMarks": self.max_marks, "feedback": self.feedback,
                "scorePct": self.score_pct}


# =========================================================
# NLTK
# =========================================================
for _pkg in ['punkt', 'punkt_tab']:
    try:    nltk.data.find(f'tokenizers/{_pkg}')
    except: nltk.download(_pkg, quiet=True)


# =========================================================
# AI MODELS
# =========================================================

# TrOCR removed — EasyOCR handles handwriting well and is 10x faster on CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading semantic model (all-mpnet-base-v2)...")
semantic_model = SentenceTransformer("all-mpnet-base-v2")

print("System ready.")

# ── Embedding cache (avoids re-encoding same text repeatedly) ──
_emb_cache = {}
def _get_embedding(text):
    if text not in _emb_cache:
        _emb_cache[text] = semantic_model.encode(text, convert_to_tensor=True)
        if len(_emb_cache) > 500:          # cap cache size
            _emb_cache.pop(next(iter(_emb_cache)))
    return _emb_cache[text]


# =========================================================
# PDF → IMAGES
# =========================================================
def pdf_to_images(pdf_bytes):
    """Convert each PDF page to a PIL Image."""
    if not PDF_SUPPORT:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")
    doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []
    for page in doc:
        mat = fitz.Matrix(2.0, 2.0)          # 2x zoom → ~150 dpi → better OCR
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


# =========================================================
# IMAGE PREPROCESSING — aggressive pipeline for handwriting
# =========================================================
def preprocess_for_ocr(img_pil):
    """Adaptive threshold + denoise — makes cursive handwriting crisp."""
    w, h = img_pil.size
    if w < 1200:
        scale   = 1200 / w
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img_np = np.array(img_pil.convert("L"))
    img_np = cv2.fastNlMeansDenoising(img_np, h=10)
    img_np = cv2.adaptiveThreshold(
        img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    return Image.fromarray(img_np).convert("RGB")

def preprocess_for_tesseract(img_pil):
    """Otsu threshold tuned for Tesseract."""
    w, h = img_pil.size
    if w < 1400:
        scale   = 1400 / w
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    img_np = np.array(img_pil.convert("L"))
    img_np = cv2.fastNlMeansDenoising(img_np, h=15)
    _, img_np = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_np = cv2.medianBlur(img_np, 3)
    return Image.fromarray(img_np)


# =========================================================
# OCR ENGINE 1 — EasyOCR
# =========================================================
def ocr_easyocr(img_pil):
    processed = preprocess_for_ocr(img_pil)
    img_np    = np.array(processed)
    results   = _easy_reader.readtext(img_np, detail=1, paragraph=False,
                                     width_ths=0.7, add_margin=0.1)
    if not results:
        return ""
    heights   = [(r[0][2][1] - r[0][0][1]) for r in results if r[0][2][1] > r[0][0][1]]
    avg_h     = sum(heights) / len(heights) if heights else 20
    threshold = max(avg_h * 0.55, 10)
    results.sort(key=lambda r: (r[0][0][1] + r[0][2][1]) / 2)
    lines, current_line, last_y = [], [], None
    for bbox, text, conf in results:
        cy = (bbox[0][1] + bbox[2][1]) / 2
        if last_y is None or abs(cy - last_y) > threshold:
            if current_line:
                current_line.sort(key=lambda x: x[0])
                lines.append(" ".join(w for _, w in current_line))
            current_line = [(bbox[0][0], text)]
            last_y = cy
        else:
            current_line.append((bbox[0][0], text))
    if current_line:
        current_line.sort(key=lambda x: x[0])
        lines.append(" ".join(w for _, w in current_line))
    return "\n".join(lines)


# =========================================================
# OCR ENGINE 2 — Tesseract (free, offline, great for handwriting)
# =========================================================
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    if os.name == "nt":
        _tp = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(_tp):
            pytesseract.pytesseract.tesseract_cmd = _tp
    print("Tesseract OCR: available")
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Tesseract OCR: not installed (run: pip install pytesseract  +  install Tesseract app)")

def ocr_tesseract(img_pil):
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        processed = preprocess_for_tesseract(img_pil)
        config    = "--psm 6 --oem 1 -c preserve_interword_spaces=1"
        return pytesseract.image_to_string(processed, config=config, lang="eng").strip()
    except Exception as e:
        print(f"[Tesseract error] {e}")
        return ""


# =========================================================
# GEMINI VISION OCR — FREE, 1500 requests/day
# =========================================================
def ocr_with_gemini(img_pil):
    """Use Google Gemini Vision (FREE) to transcribe handwriting accurately."""
    import base64
    if not GEMINI_API_KEY:
        return ""
    w, h = img_pil.size
    # Shrink to max 1000px wide — smaller = faster upload, still readable
    max_w = 1000
    if w > max_w:
        img_pil = img_pil.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=75)  # lower quality = smaller file
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        "You are a handwriting transcription expert. "
        "Carefully read ALL the handwritten text in this image EXACTLY as written. "
        "IMPORTANT: Keep numbered questions (1., 2., 2a., 2b., 3a., 3b. etc) on their own lines. "
        "Keep each sub-point and bullet on a separate line. "
        "Do NOT summarise, skip, or add anything. "
        "Output ONLY the transcribed text, nothing else."
    )
    try:
        url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
        body = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    {"text": prompt}
                ]
            }],
            "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.1}
        }
        resp   = requests.post(url, json=body, timeout=120)
        result = resp.json()

        # Debug: print full response if candidates missing
        if "candidates" not in result:
            print(f"[Gemini OCR] unexpected response: {json.dumps(result)[:300]}")
            return ""

        # Handle both response structures
        candidate = result["candidates"][0]
        if "content" in candidate:
            text = candidate["content"]["parts"][0]["text"].strip()
        elif "output" in candidate:
            text = candidate["output"].strip()
        else:
            print(f"[Gemini OCR] unknown candidate format: {list(candidate.keys())}")
            return ""

        print(f"[Gemini OCR] transcribed {len(text.split())} words ✓")
        return text
    except Exception as e:
        print(f"[Gemini OCR error] {type(e).__name__}: {e}")
        return ""


# =========================================================
# CLAUDE VISION OCR — fallback if Anthropic key has credits
# =========================================================
def ocr_with_claude(img_pil):
    """Use Claude Vision to transcribe handwriting."""
    import base64
    if not ANTHROPIC_API_KEY:
        return ""
    w, h = img_pil.size
    if w > 1600:
        img_pil = img_pil.resize((1600, int(h * 1600 / w)), Image.LANCZOS)
    buf = io.BytesIO()
    img_pil.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    prompt = (
        "Transcribe ALL handwritten text exactly. "
        "Keep numbered questions on separate lines. "
        "Output ONLY transcribed text."
    )
    try:
        resp   = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-opus-4-6", "max_tokens": 4096,
                  "messages": [{"role": "user", "content": [
                      {"type": "image", "source": {"type": "base64",
                       "media_type": "image/jpeg", "data": b64}},
                      {"type": "text", "text": prompt}
                  ]}]},
            timeout=60
        )
        result = resp.json()
        if "content" in result:
            text = result["content"][0]["text"].strip()
            print(f"[Claude OCR] transcribed {len(text.split())} words ✓")
            return text
        print(f"[Claude OCR] error: {result.get('error',{}).get('message','')}")
        return ""
    except Exception as e:
        print(f"[Claude OCR error] {e}")
        return ""


# =========================================================
# MAIN OCR — Gemini (free) → Claude → Local fallback
# =========================================================
def ocr_image(img_pil):
    """
    Priority: 1) Gemini Vision (free, accurate)
              2) Claude Vision (if credits available)
              3) EasyOCR + Tesseract (local, less accurate)
    """
    # 1. Try Gemini first (FREE, best for handwriting)
    if GEMINI_API_KEY:
        text = ocr_with_gemini(img_pil)
        if text and len(text.split()) > 5:
            return text

    # 2. Try Claude (if API key has credits)
    if ANTHROPIC_API_KEY:
        text = ocr_with_claude(img_pil)
        if text and len(text.split()) > 5:
            return text

    # 3. Local OCR fallback
    print("[OCR] Using local OCR (accuracy limited for handwriting)")
    easy_text = ocr_easyocr(img_pil)
    tess_text = ocr_tesseract(img_pil)

    def word_score(t):
        return len([w for w in t.split() if len(w) > 2 and re.search(r"[a-zA-Z]", w)])

    easy_score = word_score(easy_text)
    tess_score = word_score(tess_text)
    print(f"[OCR] EasyOCR={easy_score} words, Tesseract={tess_score} words")
    if tess_score > easy_score * 1.2 and tess_score > 10:
        return tess_text
    return easy_text if easy_text else tess_text


# =========================================================
# EXTRACT TEXT FROM FILE (image OR pdf)
# =========================================================
def extract_text_from_bytes(file_bytes, filename=""):
    """Auto-detect PDF vs image. Returns (full_text, last_page_np_image)."""
    fname  = filename.lower()
    is_pdf = fname.endswith(".pdf") or file_bytes[:4] == b"%PDF"
    if is_pdf:
        pages   = pdf_to_images(file_bytes)
        texts, last_np = [], None
        for i, pg in enumerate(pages):
            texts.append(ocr_image(pg))
            last_np = np.array(pg)
            if GEMINI_API_KEY and (i+1) % 4 == 0 and i+1 < len(pages):
                import time; print(f"[OCR] Pausing 5s for rate limit (page {i+1}/{len(pages)})..."); time.sleep(5)
        return "\n".join(texts), last_np
    else:
        img_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return ocr_image(img_pil), np.array(img_pil)


# =========================================================
# TEXT CLEANING
# =========================================================
OCR_CORRECTIONS = {
    "intelligencee":"intelligence","staus":"status","compluters":"computers",
    "acceleartion":"acceleration","accceleration":"acceleration",
    "photosythesis":"photosynthesis","artifical":"artificial",
    "unifrom":"uniform","motoin":"motion","langauge":"language",
    "represenation":"representation","sentance":"sentence",
    "knowlege":"knowledge","ontolgy":"ontology",
}

def clean_text(text):
    for k, v in OCR_CORRECTIONS.items():
        text = text.replace(k, v)
    return "\n".join(" ".join(l.split()) for l in text.split("\n"))

def remove_headings(text):
    HEADINGS = {"ANSWER KEY","STUDENT ANSWER","STUDENT ANSWERS",
                "ANSWER SHEET","ANSWERS","MODEL ANSWER","MODEL ANSWERS"}
    return "\n".join(l for l in text.split("\n") if l.strip().upper() not in HEADINGS)


# =========================================================
# QUESTION SPLITTING — supports 1, 2a, 2b, 3a … formats
# =========================================================
def split_questions(text):
    """
    Parses questions in these formats:
      1.   1)   1:
      2a.  2a)  2a:
      3b.  3b)
    Returns OrderedDict: { "1": "...", "2a": "...", "2b": "...", ... }
    """
    lines    = text.split("\n")
    questions = {}
    order     = []
    current_key  = None
    current_text = []

    # Pattern: digit(s) + optional letter + separator
    QPAT = re.compile(r'^(\d{1,2}[a-eA-E]?)[.):\s]\s*(.*)', re.IGNORECASE)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = QPAT.match(line)
        if m:
            qkey = m.group(1).lower()   # "1", "2a", "3b" etc.
            rest = m.group(2).strip()
            # Validate: main number must be 1–20
            main_num = int(re.match(r'\d+', qkey).group())
            if 1 <= main_num <= 20:
                if current_key is not None:
                    questions[current_key] = " ".join(current_text).strip()
                current_key  = qkey
                current_text = [rest] if rest else []
                if qkey not in order:
                    order.append(qkey)
                continue
        if current_key is not None:
            current_text.append(line)

    if current_key is not None:
        questions[current_key] = " ".join(current_text).strip()

    if not questions:
        return {"1": text.strip()}, ["1"]

    return questions, order


# =========================================================
# SUB-POINT SPLITTING
# =========================================================
def split_subpoints(text):
    parts = re.split(r'(?<!\d)\s+(?=\d+[.)]\s)', text)
    parts = [re.sub(r'^\d+[.)]\s*', '', p).strip() for p in parts if len(p.strip()) > 4]
    if len(parts) > 1:
        return parts
    parts = [p.strip() for p in re.split(r'\b(\d+)\.\s+', text)
             if not re.match(r'^\d+$', p) and len(p.strip()) > 4]
    return parts if len(parts) > 1 else [text.strip()]


# =========================================================
# KEYWORDS + SCORING
# =========================================================
STOP_WORDS = {
    'the','a','an','is','are','was','were','be','been','being','by','of',
    'in','to','for','on','at','it','its','that','this','with','and','or',
    'but','how','when','which','what','from','they','their','there','has',
    'have','had','will','would','can','does','do','did','not','also',
    'than','then','into','about','its','we','us','our'
}

def extract_keywords(text):
    try:    tokens = nltk.word_tokenize(text.lower())
    except: tokens = text.lower().split()
    return list({w for w in tokens if len(w) > 3 and w.isalpha() and w not in STOP_WORDS})

def is_math(text):
    t = text.strip()
    return len(t) <= 60 and bool(re.search(r'[=+\-*/]', t)) and \
           bool(re.search(r'\d|\b[a-zA-Z]\s*=', t))

def normalize_math(text):
    t = re.sub(r'\s+', '', text.lower())
    return t.replace('×','*').replace('÷','/').replace('²','^2')

def semantic_score(a, b):
    if not a.strip() or not b.strip(): return 0.0
    e1 = _get_embedding(a)
    e2 = _get_embedding(b)
    return float(util.cos_sim(e1, e2).item())

def keyword_score(key, stu):
    k = extract_keywords(key); s = extract_keywords(stu)
    return len(set(k) & set(s)) / len(k) if k else 0.0

def length_factor(key, stu):
    if not stu.strip(): return 0.0
    ratio = len(stu.strip()) / max(len(key.strip()), 1)
    return 1.0 if ratio >= 0.30 else ratio / 0.30

def combined_score(key, stu):
    raw = (0.72 * semantic_score(key, stu)) + (0.28 * keyword_score(key, stu))
    return raw * length_factor(key, stu)


# =========================================================
# CLAUDE AI GRADER
# =========================================================
def ai_grade(key_answer, student_answer, max_marks):
    if not ANTHROPIC_API_KEY:
        return None, None
    prompt = (
        f'You are a strict but fair university exam evaluator.\n\n'
        f'Answer Key: "{key_answer}"\nStudent Answer: "{student_answer}"\n'
        f'Maximum marks: {max_marks}\n\n'
        f'Award marks for correct concepts even if wording differs. '
        f'Penalise only if key concepts are missing or wrong.\n'
        f'Respond ONLY with valid JSON (no markdown):\n'
        f'{{"marks": <0 to {max_marks}>, "feedback": "<one sentence>"}}'
    )
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY,
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "claude-haiku-4-5-20251001", "max_tokens": 150,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=12
        )
        raw    = re.sub(r'^```json|```$', '',
                        resp.json()["content"][0]["text"].strip(),
                        flags=re.MULTILINE).strip()
        result = json.loads(raw)
        m      = max(0, min(max_marks, int(result.get("marks", 0))))
        return m / max_marks, f"[AI] {result.get('feedback', '')}"
    except Exception as e:
        print(f"[AI grader error] {e}")
        return None, None


# =========================================================
# COMPARE + GRADE
# =========================================================
def compare(key, stu, max_marks=SCORE_PER_QUESTION):
    key, stu = key.strip(), stu.strip()
    if not stu:
        return 0.0, "No answer provided."

    if is_math(key) or is_math(stu):
        if normalize_math(key) == normalize_math(stu):
            return 1.0, "Correct"
        kn = set(re.findall(r'\d+', key)); sn = set(re.findall(r'\d+', stu))
        if kn == sn:
            return 0.85, "Correct (equivalent form)"
        return len(kn & sn) / max(len(kn), 1) * 0.5, "Partially correct"

    score = combined_score(key, stu)

    # Borderline → Claude AI
    if AI_GRADER_LOW <= score <= AI_GRADER_HIGH and ANTHROPIC_API_KEY:
        ai_s, ai_fb = ai_grade(key, stu, max_marks)
        if ai_s is not None:
            return 0.40 * score + 0.60 * ai_s, ai_fb

    return score, None


def grade_answer(key, stu, max_marks=SCORE_PER_QUESTION, diagram_present=False):
    if not stu.strip():
        return 0, "No answer provided."

    key_parts = split_subpoints(key)
    stu_parts = split_subpoints(stu)

    if len(key_parts) > 1:
        marks, part_max, feedbacks = 0.0, max_marks / len(key_parts), []
        for j, kp in enumerate(key_parts):
            sp = stu_parts[j] if len(stu_parts) > 1 and j < len(stu_parts) else stu
            score, fb = compare(kp, sp, part_max)
            if score >= 0.58:
                m = part_max;         feedbacks.append(f"Sub-point {j+1}: correct")
            elif score >= 0.36:
                m = part_max * 0.55;  feedbacks.append(f"Sub-point {j+1}: partial")
            else:
                m = 0;                feedbacks.append(f"Sub-point {j+1}: missing/incorrect")
            marks += m
        return round(marks), "; ".join(feedbacks)

    score, ai_fb = compare(key, stu, max_marks)

    if score >= FULL_MARKS_THRESHOLD:
        marks, feedback = max_marks, ai_fb or "Correct"
    elif score >= PARTIAL_HIGH_THRESHOLD:
        marks, feedback = int(max_marks * 0.70), ai_fb or "Good answer — most key concepts present"
    elif score >= PARTIAL_LOW_THRESHOLD:
        marks, feedback = int(max_marks * 0.40), ai_fb or "Partial — some concepts missing"
    else:
        marks, feedback = 0, ai_fb or "Incorrect or insufficient"

    if diagram_present:
        marks = min(max_marks, marks + 1)
    return marks, feedback


def _detect_diagram(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return np.sum(canny(gray.astype(float))) / gray.size > 0.05
    except: return False

def remark(p):
    if p >= 90: return "Excellent"
    if p >= 75: return "Very Good"
    if p >= 60: return "Good"
    if p >= 40: return "Needs Improvement"
    return "Poor"

def err(msg, code=400):
    return jsonify({"error": msg}), code


# =========================================================
# AUTH ROUTES
# =========================================================
@app.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    email    = (data.get("email")    or "").strip()
    password = (data.get("password") or "").strip()
    if not username or not email or not password:
        return err("username, email and password are required.")
    if User.query.filter_by(username=username).first():
        return err("Username already exists.", 409)
    if User.query.filter_by(email=email).first():
        return err("Email already registered.", 409)
    user = User(username=username, email=email, role=data.get("role","teacher"))
    user.set_password(password)
    db.session.add(user); db.session.commit()
    return jsonify({"message": "User registered.", "user": user.to_dict()}), 201

@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    user = User.query.filter_by(username=(data.get("username") or "").strip()).first()
    if not user or not user.check_password((data.get("password") or "").strip()):
        return err("Invalid credentials.", 401)
    if not user.is_active:
        return err("Account disabled.", 403)
    return jsonify({"message": "Login successful.", "user": user.to_dict()})

@app.route("/auth/users", methods=["GET"])
def list_users():
    return jsonify([u.to_dict() for u in User.query.all()])

@app.route("/auth/users/<int:uid>", methods=["DELETE"])
def delete_user(uid):
    u = User.query.get_or_404(uid)
    db.session.delete(u); db.session.commit()
    return jsonify({"message": "User deleted."})


# =========================================================
# STUDENT ROUTES
# =========================================================
@app.route("/students", methods=["POST"])
def create_student():
    data = request.get_json() or {}
    name    = (data.get("name")   or "").strip()
    roll_no = (data.get("rollNo") or "").strip()
    if not name or not roll_no:
        return err("name and rollNo are required.")
    if Student.query.filter_by(roll_no=roll_no).first():
        return err(f"Roll number '{roll_no}' already exists.", 409)
    s = Student(name=name, roll_no=roll_no,
                email=data.get("email",""), class_name=data.get("class",""),
                section=data.get("section",""))
    db.session.add(s); db.session.commit()
    return jsonify({"message": "Student created.", "student": s.to_dict()}), 201

@app.route("/students", methods=["GET"])
def list_students():
    q      = request.args.get("q","")
    cls    = request.args.get("class","")
    query  = Student.query
    if q:   query = query.filter(Student.name.ilike(f"%{q}%") | Student.roll_no.ilike(f"%{q}%"))
    if cls: query = query.filter_by(class_name=cls)
    return jsonify([s.to_dict() for s in query.order_by(Student.name).all()])

@app.route("/students/<int:sid>", methods=["GET"])
def get_student(sid):
    s = Student.query.get_or_404(sid)
    d = s.to_dict()
    d["evaluationHistory"] = [e.to_dict(include_answers=False) for e in s.evaluations]
    return jsonify(d)

@app.route("/students/<int:sid>", methods=["PUT"])
def update_student(sid):
    s    = Student.query.get_or_404(sid)
    data = request.get_json() or {}
    for field, col in [("name","name"),("email","email"),("class","class_name"),("section","section")]:
        if field in data: setattr(s, col, data[field])
    db.session.commit()
    return jsonify({"message": "Updated.", "student": s.to_dict()})

@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    s = Student.query.get_or_404(sid)
    db.session.delete(s); db.session.commit()
    return jsonify({"message": "Student deleted."})


# =========================================================
# ANSWER KEY ROUTES
# =========================================================
@app.route("/answer-keys", methods=["POST"])
def create_answer_key():
    if request.content_type and "multipart" in request.content_type:
        f         = request.files.get("keyImage")
        if not f: return err("keyImage required.")
        file_bytes = f.read()
        filename   = f.filename or ""
        key_text, _= extract_text_from_bytes(file_bytes, filename)
        key_text   = remove_headings(clean_text(key_text))
        qdict, order = split_questions(key_text)
        questions  = [qdict.get(k, "") for k in order]
        subject    = request.form.get("subject","")
        exam_name  = request.form.get("examName","")
        created_by = request.form.get("createdBy")
    else:
        data      = request.get_json() or {}
        subject   = data.get("subject","")
        exam_name = data.get("examName","")
        questions = data.get("questions",[])
        created_by= data.get("createdBy")

    if not subject:   return err("subject is required.")
    if not questions: return err("No questions found.")

    ak = AnswerKey(subject=subject, exam_name=exam_name, created_by=created_by)
    ak.set_questions(questions)
    db.session.add(ak); db.session.commit()
    return jsonify({"message": "Answer key saved.", "answerKey": ak.to_dict()}), 201

@app.route("/answer-keys", methods=["GET"])
def list_answer_keys():
    subject = request.args.get("subject","")
    query   = AnswerKey.query
    if subject: query = query.filter(AnswerKey.subject.ilike(f"%{subject}%"))
    return jsonify([k.to_dict() for k in query.order_by(AnswerKey.created_at.desc()).all()])

@app.route("/answer-keys/<int:kid>", methods=["GET"])
def get_answer_key(kid):
    return jsonify(AnswerKey.query.get_or_404(kid).to_dict())

@app.route("/answer-keys/<int:kid>", methods=["DELETE"])
def delete_answer_key(kid):
    ak = AnswerKey.query.get_or_404(kid)
    db.session.delete(ak); db.session.commit()
    return jsonify({"message": "Deleted."})


# =========================================================
# EVALUATE — main route (PDF + image, sub-questions)
# =========================================================
@app.route("/evaluate", methods=["POST"])
def evaluate_api():
    stu_file      = request.files.get("studentScript")
    key_file      = request.files.get("answerKey")
    answer_key_id = request.form.get("answerKeyId")

    name         = request.form.get("studentName", "Unknown")
    roll_no      = request.form.get("rollNo",       "NA")
    subject      = request.form.get("subject",      "NA")
    evaluator_id = request.form.get("evaluatorId")

    if not stu_file: return err("studentScript is required.")

    # ── Resolve answer key ──
    if answer_key_id:
        ak   = AnswerKey.query.get(int(answer_key_id))
        if not ak: return err(f"Answer key {answer_key_id} not found.", 404)
        keys  = ak.get_questions()
        qkeys = [str(i+1) for i in range(len(keys))]
    elif key_file:
        key_bytes        = key_file.read()
        key_text, _      = extract_text_from_bytes(key_bytes, key_file.filename or "")
        key_text         = remove_headings(clean_text(key_text))
        qdict, qkeys     = split_questions(key_text)
        keys             = [qdict.get(k, "") for k in qkeys]
        ak               = None
    else:
        return err("Provide answerKeyId or answerKey file.")

    # ── OCR student script ──
    stu_bytes            = stu_file.read()
    stu_text, img        = extract_text_from_bytes(stu_bytes, stu_file.filename or "")
    stu_text             = remove_headings(clean_text(stu_text))
    sdict, _             = split_questions(stu_text)
    stus                 = [sdict.get(k, "") for k in qkeys]

    # ── Grade ──
    diagram  = _detect_diagram(img)
    total    = len(keys) * SCORE_PER_QUESTION
    obtained = 0
    details  = []

    for i, key in enumerate(keys):
        stu    = stus[i] if i < len(stus) else ""
        qno    = qkeys[i] if i < len(qkeys) else str(i+1)
        marks, feedback = grade_answer(key, stu, SCORE_PER_QUESTION, diagram)
        obtained += marks
        details.append({
            "question": qno, "keyAnswer": key, "studentAnswer": stu,
            "marks": marks, "maxMarks": SCORE_PER_QUESTION,
            "feedback": feedback,
            "scorePct": round((marks / SCORE_PER_QUESTION) * 100)
        })

    percent = round((obtained / total) * 100, 2) if total > 0 else 0
    rmk     = remark(percent)

    # ── Save student record ──
    student = Student.query.filter_by(roll_no=roll_no).first()
    if not student and roll_no != "NA":
        student = Student(name=name, roll_no=roll_no)
        db.session.add(student); db.session.flush()

    # ── Save evaluation ──
    evaluation = Evaluation(
        student_id    = student.id if student else None,
        answer_key_id = ak.id if ak else None,
        evaluator_id  = int(evaluator_id) if evaluator_id else None,
        subject=subject, total_marks=total, obtained_marks=obtained,
        percentage=percent, remark=rmk, ai_grader_used=bool(ANTHROPIC_API_KEY)
    )
    db.session.add(evaluation); db.session.flush()

    for d in details:
        db.session.add(EvalAnswer(
            evaluation_id=evaluation.id, question_no=str(d["question"]),
            key_answer=d["keyAnswer"], student_answer=d["studentAnswer"],
            marks=d["marks"], max_marks=d["maxMarks"],
            feedback=d["feedback"], score_pct=d["scorePct"]
        ))
    db.session.commit()

    return jsonify({
        "evaluationId": evaluation.id, "studentName": name,
        "rollNo": roll_no, "subject": subject,
        "totalMarks": total, "obtainedMarks": obtained,
        "percentage": percent, "remark": rmk,
        "answers": details, "aiGraderUsed": bool(ANTHROPIC_API_KEY),
        "_debug": {"keyQuestions": dict(zip(qkeys, keys)),
                   "stuQuestions": dict(zip(qkeys, stus))}
    })


# =========================================================
# RESULTS
# =========================================================
@app.route("/results", methods=["GET"])
def list_results():
    subject  = request.args.get("subject","")
    roll_no  = request.args.get("rollNo","")
    page     = int(request.args.get("page", 1))
    per_page = int(request.args.get("perPage", 20))
    query    = Evaluation.query
    if subject: query = query.filter(Evaluation.subject.ilike(f"%{subject}%"))
    if roll_no: query = query.join(Student).filter(Student.roll_no == roll_no)
    paginated = query.order_by(Evaluation.evaluated_at.desc()).paginate(
                    page=page, per_page=per_page, error_out=False)
    return jsonify({"total": paginated.total, "page": page, "perPage": per_page,
                    "pages": paginated.pages,
                    "results": [e.to_dict(include_answers=False) for e in paginated.items]})

@app.route("/results/<int:eid>", methods=["GET"])
def get_result(eid):
    return jsonify(Evaluation.query.get_or_404(eid).to_dict(include_answers=True))

@app.route("/results/<int:eid>", methods=["DELETE"])
def delete_result(eid):
    ev = Evaluation.query.get_or_404(eid)
    db.session.delete(ev); db.session.commit()
    return jsonify({"message": "Deleted."})

@app.route("/results/stats", methods=["GET"])
def stats():
    subject = request.args.get("subject","")
    query   = Evaluation.query
    if subject: query = query.filter(Evaluation.subject.ilike(f"%{subject}%"))
    evals   = query.all()
    if not evals: return jsonify({"message": "No evaluations found."})
    pcts    = [e.percentage for e in evals]
    return jsonify({
        "totalEvaluations": len(evals),
        "averageScore":     round(sum(pcts)/len(pcts), 2),
        "highestScore":     max(pcts), "lowestScore": min(pcts),
        "passRate":         round(sum(1 for p in pcts if p >= 40)/len(evals)*100, 2),
        "gradeDistribution": {
            "Excellent":        sum(1 for p in pcts if p >= 90),
            "VeryGood":         sum(1 for p in pcts if 75 <= p < 90),
            "Good":             sum(1 for p in pcts if 60 <= p < 75),
            "NeedsImprovement": sum(1 for p in pcts if 40 <= p < 60),
            "Poor":             sum(1 for p in pcts if p < 40),
        }
    })


# =========================================================
# DEBUG + HEALTH
# =========================================================
@app.route("/debug", methods=["POST"])
def debug_api():
    result = {}
    for field, key in [('answerKey','key'), ('studentScript','stu')]:
        f = request.files.get(field)
        if f:
            text, _ = extract_text_from_bytes(f.read(), f.filename or "")
            clean   = remove_headings(clean_text(text))
            qdict, order = split_questions(clean)
            result[f"{key}Raw"]       = text
            result[f"{key}Clean"]     = clean
            result[f"{key}Questions"] = qdict
            result[f"{key}Order"]     = order
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "pdfSupport": PDF_SUPPORT,
                    "claudeOCR": bool(ANTHROPIC_API_KEY),
                    "aiGrader": bool(ANTHROPIC_API_KEY),
                    "ocrEngine": "PaddleOCR" if PADDLE_AVAILABLE else "EasyOCR (fallback)",
                    "semModel": "all-mpnet-base-v2",
                    "db": "SQLite — smec.db"})


# =========================================================
# INIT + RUN
# =========================================================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        if not User.query.first():
            admin = User(username="admin", email="admin@smec.local", role="admin")
            admin.set_password("admin123")
            db.session.add(admin); db.session.commit()
            print("✓ Default admin created  →  username: admin  |  password: admin123")
        print("✓ Database ready  →  smec.db")
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)