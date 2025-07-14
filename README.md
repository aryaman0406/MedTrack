# 🏥 Medical Tracking & AI Health Assistant

A comprehensive Flask-based web application that combines medication tracking, health analytics, and AI-powered medical assistance. Features include intelligent symptom checking, drug interaction analysis, and a medical chatbot powered by advanced language models.

## ✨ Features

### 📊 **Core Health Tracking**
- **Medication Reminders**: Set, track, and manage daily medication schedules
- **Health Analytics**: Visual charts showing medication adherence and trends
- **Symptom Logging**: Track symptoms with severity ratings over time
- **Medical History**: Maintain comprehensive health records
- **Document Storage**: Upload and organize medical documents by category

### 🤖 **AI-Powered Features**
- **Medical Chatbot**: Get instant answers to health questions using medical knowledge base
- **Symptom Checker**: AI-powered symptom analysis with risk assessment
- **Drug Interaction Checker**: Check for dangerous medication combinations
- **Predictive Analytics**: ML-based prediction of missed medication reminders

### 📱 **Additional Tools**
- **OCR Text Extraction**: Extract text from medical documents/prescriptions
- **PDF Report Generation**: Export health data as professional reports
- **Profile Management**: BMI, BMR, and hydration calculations
- **CSV Export**: Export medication data for external analysis

## 🛠️ Technology Stack

### **Backend**
- **Flask**: Web framework with session management
- **SQLAlchemy**: Database ORM with SQLite
- **Flask-Login**: User authentication and authorization
- **WeasyPrint**: PDF generation for health reports

### **AI & Machine Learning**
- **LangChain**: AI application framework
- **GROQ API**: Primary LLM for fast responses (Llama 3)
- **XAI API**: Fallback LLM (Grok model)
- **HuggingFace**: Embeddings (sentence-transformers)
- **FAISS**: Vector database for medical knowledge retrieval
- **Scikit-learn**: Predictive analytics for medication adherence

### **Frontend**
- **HTML5/CSS3**: Responsive design
- **Tailwind CSS**: Modern styling framework
- **Chart.js**: Interactive data visualizations
- **JavaScript**: Dynamic UI interactions

### **Additional Libraries**
- **OpenCV**: Image processing for OCR
- **Tesseract**: Optical character recognition
- **python-dotenv**: Environment variable management

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git
- API keys for GROQ and/or XAI

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/medical-tracking-app.git
   cd medical-tracking-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   XAI_API_KEY=your_xai_api_key_here
   ```

4. **Initialize the database**
   ```bash
   python app.py
   ```
   The database will be created automatically on first run.

5. **Access the application**
   Open your browser and navigate to `http://127.0.0.1:5000`

## 📋 API Setup Guide

### GROQ API (Recommended - Free Tier)
1. Visit [GROQ Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Generate a new API key
5. Add to your `.env` file

**Rate Limits**: 30 requests/minute, 14,400 requests/day (free tier)

### XAI API (Fallback)
1. Visit [X.AI Platform](https://console.x.ai/)
2. Create an account and get API access
3. Generate API key
4. Add to your `.env` file

## 📁 Project Structure

```
medical-tracking-app/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (create this)
├── README.md             # This file
├── 
├── medical-chatbot/      # AI knowledge base
│   ├── vectorstore/      # FAISS vector database
│   ├── data/            # Medical reference documents
│   └── *.py             # Chatbot configuration files
├── 
├── templates/           # HTML templates
│   ├── base.html        # Base template
│   ├── dashboard.html   # Main dashboard
│   ├── chatbot.html     # AI chatbot interface
│   ├── analytics.html   # Health analytics
│   └── *.html          # Other pages
├── 
├── static/             # Static assets
│   ├── css/           # Stylesheets
│   └── js/            # JavaScript files
├── 
├── uploads/           # User uploaded files
└── instance/         # Database files
    └── meds.db       # SQLite database
```

## 🎯 Usage Examples

### Setting Up Medication Reminders
1. Register/Login to your account
2. Navigate to Dashboard
3. Add medication with name, dosage, and time
4. Mark as taken when you take your medication
5. View analytics to track adherence

### Using the AI Chatbot
1. Go to the Chatbot page
2. Ask health-related questions like:
   - "What are the side effects of ibuprofen?"
   - "How to manage high blood pressure?"
   - "What foods are good for diabetes?"

### Symptom Checker
1. Navigate to Symptom Checker
2. Select your symptoms from the list
3. Specify duration and severity
4. Get AI-powered analysis with risk assessment

### Drug Interaction Check
1. Go to Drug Interactions page
2. Enter multiple medications (comma-separated)
3. Get analysis of potential interactions
4. View safety recommendations

## 🔧 Configuration

### Database Models
- **User**: Store user profiles and health data
- **Reminder**: Medication schedules and tracking
- **SymptomLog**: Symptom history with severity
- **ChatHistory**: AI conversation logs
- **Document**: Uploaded medical documents

### AI Configuration
The app uses a dual-API system for reliability:
- **Primary**: GROQ API (faster, free tier available)
- **Fallback**: XAI API (backup when GROQ is unavailable)

### Customization
- Modify `medical-chatbot/` for custom knowledge base
- Update CSS in `static/css/` for styling changes
- Extend database models in `app.py` for additional features

## 🧪 Testing

### Test AI Components
Visit `/debug/test-ai` to verify:
- API key validation
- Vector database connectivity
- Search functionality
- LLM connection status

### Manual Testing
1. **Registration/Login**: Test user authentication
2. **Medication Tracking**: Add/modify/track medications
3. **AI Features**: Test chatbot, symptom checker, drug interactions
4. **Data Export**: Generate PDF reports and CSV exports

## 🔒 Security Features

- **Password Hashing**: Werkzeug security for password protection
- **Session Management**: Flask-Login for secure user sessions
- **File Upload Security**: Secure filename handling
- **SQL Injection Protection**: SQLAlchemy ORM prevents injection attacks
- **Input Validation**: Form validation on frontend and backend

## 📊 Analytics & Insights

### Health Metrics
- Medication adherence percentage
- Daily/weekly medication trends
- Symptom severity tracking over time
- BMI, BMR, and hydration calculations

### Predictive Features
- ML-based prediction of missed medications
- Risk assessment for symptom combinations
- Drug interaction severity analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add comments for complex functions
- Update requirements.txt for new dependencies
- Test AI features with both API providers

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Troubleshooting

### Common Issues

**AI Features Not Working**
- Check API keys in `.env` file
- Verify GROQ key is 50+ characters
- Test with `/debug/test-ai` route

**Database Errors**
- Delete `instance/meds.db` and restart app
- Check file permissions in project directory

**OCR Not Working**
- Install Tesseract OCR on your system
- Verify OpenCV installation

### Getting Help
- Check the [Issues](https://github.com/yourusername/medical-tracking-app/issues) page
- Review API setup guides: `GROQ_API_SETUP.md`
- Test system status with debug routes

## 🙏 Acknowledgments

- **GROQ**: Fast LLM inference for medical responses
- **XAI**: Backup LLM service for reliability
- **HuggingFace**: Embedding models for semantic search
- **LangChain**: AI application development framework
- **Medical Encyclopedia**: Knowledge base for medical information

## 🔮 Future Enhancements

- [ ] Mobile app development (React Native/Flutter)
- [ ] Integration with wearable devices
- [ ] Telemedicine appointment scheduling
- [ ] Prescription refill reminders
- [ ] Healthcare provider integration
- [ ] Multi-language support
- [ ] Voice-activated medication reminders

---

**⚠️ Medical Disclaimer**: This application is for informational purposes only and should not replace professional medical advice. Always consult with healthcare providers for medical decisions.

**🔗 Live Demo**: [Add your deployment URL here]

**📧 Contact**: [Your contact information]

---

<div align="center">
  <strong>Built with ❤️ for better health management</strong>
</div>
