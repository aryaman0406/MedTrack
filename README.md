# 💊 MedTrack: Intelligent Health Companion (API-Free)

![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-green.svg)
![Privacy](https://img.shields.io/badge/privacy-100%25_Local-red.svg)

**MedTrack** is a premium, privacy-focused health management system built to solve real-world medical challenges without relying on external paid APIs. It combines advanced web scraping, machine learning, and a stunning "human-made" design to provide a world-class health companion.

---

## 🌟 Proprietary Features

### 🍎 **Diet-Medication Safety Scanner (New)**
A proactive safety tool that identifies dangerous interactions between your medications and common foods/drinks (e.g., Grapefruit, Dairy, Leafy Greens). It uses verified medical databases and real-time scraping to keep you safe.

### 🤖 **API-Free Medical AI Suite**
*   **🧠 Health AI Chatbot**: Instant answers to medical queries, powered by dynamic scraping from **Wikipedia** & **MedlinePlus**.
*   **🔍 Advanced Symptom Analyzer**: Hybrid analysis (Local DB + Web NLP) to assess risk levels and provide common-sense recommendations.
*   **💊 Smart Drug Interaction Checker**: Scan multiple medications to detect potentially harmful combinations.

### 📈 **Smart Tracking & Compliance**
*   **📋 Personalized Routine**: Manage medications, dosages, and schedules with intelligent "taken" logic.
*   **� Predictive Adherence**: Uses Logistic Regression to identify "High Risk" times when you're likely to miss a dose.
*   **� Premium Hydration Tracker**: Goal-oriented water tracking based on your weight and health profile.
*   **🤕 Side Effect Logger**: Track patterns in medication reactions to discuss with your doctor.

### 🛡️ **Organization & Accessibility**
*   **🆘 Emergency Hub**: Print a physical **Emergency QR Card** with your vital information and contacts.
*   **� OCR Vault**: Scan physical prescriptions and reports using Tesseract OCR to digitize your medical history.
*   **�‍👩‍👧‍👦 Family Suite**: Dedicated profiles for family members to track their specific conditions and allergies.
*   **📄 Clinical PDF Export**: Generate professional health summaries for your next doctor's visit.

---

## 🚀 One-Click Launch (Windows)

MedTrack is designed to be plug-and-play.
1.  **Clone the repository**.
2.  Double-click **`run_app.bat`**.
    *   *This automates Python checks, venv creation, dependency installation, and server startup.*
3.  Access your dashboard at `http://127.0.0.1:5000`.

---

## 🛠️ Technology Stack

*   **Logic**: Python / Flask / SQLAlchemy
*   **AI/ML**: `scikit-learn` (Predictive Models), `BeautifulSoup4` (Data Extraction)
*   **Vision**: `pytesseract` & `OpenCV` (OCR Digitization)
*   **Design**: Modern CSS Design System (Glassmorphism, Dark/Light Mode), Outfit & Inter Typography

---

## ⚠️ Medical Disclaimer
**MedTrack is for informational purposes only.** It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider before making medical decisions.
