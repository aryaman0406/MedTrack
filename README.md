# 🏥 Medical Tracking & AI Health Assistant

A Flask-based web application for medication tracking, health analytics, and AI-powered medical assistance. Features include medication reminders, symptom logging, AI chatbot, symptom checker, drug interaction analysis, and health data export.

## ✨ Key Features
- **Medication Management**: Schedule and track medications with adherence analytics
- **AI Health Tools**: Medical chatbot, symptom checker, and drug interaction analysis powered by GROQ and XAI APIs
- **Health Insights**: Visual charts for trends, BMI/BMR calculations, and PDF/CSV reports
- **Document Handling**: OCR for prescriptions and secure medical document storage

## 🛠️ Tech Stack
- **Backend**: Flask, SQLAlchemy, Flask-Login, WeasyPrint
- **AI/ML**: LangChain, GROQ API, XAI API, HuggingFace, FAISS, Scikit-learn
- **Frontend**: HTML5, Tailwind CSS, Chart.js, JavaScript
- **Other**: OpenCV, Tesseract, python-dotenv

## 🚀 Quick Start
1. Clone repo: `git clone https://github.com/yourusername/medical-tracking-app.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Set up `.env` with GROQ and XAI API keys
4. Run: `python app.py`
5. Access: `http://127.0.0.1:5000`

## 📋 Requirements
- Python 3.8+
- Git
- GROQ/XAI API keys

## 🔒 Security
- Password hashing with Werkzeug
- Secure session management with Flask-Login
- SQL injection protection via SQLAlchemy
- Validated file uploads

## 🤝 Contributing
Fork, create a feature branch, commit changes, and open a pull request. Follow PEP 8 and update `requirements.txt` for new dependencies.

## 📝 License
MIT License

## ⚠️ Disclaimer
For informational purposes only. Consult healthcare providers for medical decisions.

---

**Built with ❤️ for better health management**
