
Integrated AI Evaluator added.

What I changed:
- Added ai-service/ (Python Flask evaluator using Tesseract + sentence-transformers)
- Included app.py and requirements.txt in ai-service/
- Included a Node helper (ai-service/node_integration_example.js) that shows how to forward files from Node to Python.

How to run Python AI service:
1. Install Python 3.9+
2. Install Tesseract (https://github.com/tesseract-ocr/tesseract). Ensure `tesseract` is in PATH or set pytesseract.pytesseract.tesseract_cmd in ai-service/app.py.
3. Create venv and install requirements:
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r ai-service/requirements.txt
4. Run:
   python ai-service/app.py
   -> Service available at http://localhost:5000/evaluate

How to integrate into your Node project:
- Use the provided node_integration_example.js code.
- In your Node route, after receiving uploaded files (e.g. with express-fileupload or multer),
  call forwardToAI(req.files) and return the result to the frontend.

Cleaning:
- node_modules and common caches removed from the packaged zip.

If you want, I can now automatically modify your Node server to include a working route that forwards uploads to the AI service. Ask me to "patch node server" and I will edit specific files.
