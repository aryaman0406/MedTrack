from flask import Flask, render_template, redirect, request, url_for, flash, jsonify, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import pytesseract
import cv2
import numpy as np
import os
import csv
import time  
import random
import traceback

from openai import OpenAI  
from dotenv import load_dotenv
from flask import make_response
from weasyprint import HTML

# Medibot imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

app = Flask(__name__)

# Production-ready configuration
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///meds.db')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

load_dotenv()
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found in environment variables. Please set it in the .env file.")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)

# Medibot configuration
DB_FAISS_PATH = "medical-chatbot/vectorstore/db_faiss"

# Cache for vectorstore (Flask doesn't have streamlit's cache_resource, so we'll use a global variable)
_vectorstore_cache = None

def get_llm_client():
    """Get the best available LLM client (GROQ preferred, XAI as fallback)"""
    
    # Try GROQ first
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and len(groq_api_key) >= 50:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model_name="llama3-8b-8192",
                temperature=0.0,
                groq_api_key=groq_api_key,
                timeout=30,  # Add timeout
                max_retries=2
            )
            print("GROQ client created successfully")
            return llm, "GROQ"
        except Exception as e:
            print(f"GROQ failed: {e}")
    
    # Fallback to XAI (using OpenAI interface)
    xai_api_key = os.getenv("XAI_API_KEY")
    if xai_api_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="grok-beta",
                temperature=0.0,
                openai_api_key=xai_api_key,
                openai_api_base="https://api.x.ai/v1",
                timeout=30,  # Add timeout
                max_retries=2
            )
            print("XAI client created successfully")
            return llm, "XAI"
        except Exception as e:
            print(f"XAI failed: {e}")
    
    print("Both GROQ and XAI failed")
    return None, "None"

def get_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is None:
        try:
            print("Loading vectorstore...")
            
            # Check if vectorstore path exists
            if not os.path.exists(DB_FAISS_PATH):
                print(f"Vectorstore path does not exist: {DB_FAISS_PATH}")
                return None
            
            print("Creating embedding model...")
            embedding_model = HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={'device': 'cpu'},  # Force CPU to avoid GPU issues
                encode_kwargs={'normalize_embeddings': True}  # Better performance
            )
            print("Embedding model created")
            
            print("Loading FAISS vectorstore...")
            _vectorstore_cache = FAISS.load_local(
                DB_FAISS_PATH, 
                embedding_model, 
                allow_dangerous_deserialization=True
            )
            print("Vectorstore loaded successfully")
            
        except Exception as e:
            print(f"Error loading vectorstore: {e}")
            print("Continuing without vectorstore - will use direct LLM responses")
            import traceback
            traceback.print_exc()
            return None
    return _vectorstore_cache

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def format_sources(source_documents):
    formatted_sources = ""
    for i, doc in enumerate(source_documents, 1):
        content = doc.page_content.strip()
        source_link = doc.metadata.get("source", "No source link available")

        # Check for valid URL
        if source_link.startswith("http"):
            formatted_sources += f"\n📄 **Source {i}:**\n{content}\n🔗 [View Source {i}]({source_link})\n"
        else:
            formatted_sources += f"\n📄 **Source {i}:**\n{content}\n🔗 Source Path: `{source_link}`\n"
    return formatted_sources

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    name = db.Column(db.String(150))
    age = db.Column(db.Integer)
    height_cm = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    blood_group = db.Column(db.String(10))
    conditions = db.Column(db.Text)
    reminders = db.relationship('Reminder', backref='user', lazy=True)

class Reminder(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150))
    dosage = db.Column(db.String(100))
    time = db.Column(db.String(10))
    taken = db.Column(db.Boolean, default=False)
    time_taken = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor = db.Column(db.String(100))
    reason = db.Column(db.String(200))
    date = db.Column(db.String(20))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class MedicalHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.Text)
    date = db.Column(db.String(20))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200))
    category = db.Column(db.String(50))  
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class SymptomLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.String(20))
    symptom = db.Column(db.String(200))
    severity = db.Column(db.Integer)  
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sources = db.Column(db.Text)  # JSON string of source documents

def predict_missed_reminders(reminders):
    if not reminders:
        return []

    features, labels = [], []
    le_dosage = LabelEncoder()
    all_dosages = list({r.dosage for r in reminders})
    le_dosage.fit(all_dosages)

    for r in reminders:
        hour = int(r.time.split(":")[0])
        weekday = r.created_at.weekday() if r.created_at else 0
        dosage_encoded = le_dosage.transform([r.dosage])[0]
        features.append([hour, weekday, dosage_encoded])
        labels.append(int(r.taken))

    if len(set(labels)) < 2:
        return []

    model = LogisticRegression()
    model.fit(features, labels)

    high_risk = []
    for r in reminders:
        if not r.taken:
            hour = int(r.time.split(":")[0])
            weekday = date.today().weekday()
            dosage_encoded = le_dosage.transform([r.dosage])[0]
            pred = model.predict_proba([[hour, weekday, dosage_encoded]])[0][1]
            if pred < 0.5:
                high_risk.append((r, pred))
    return high_risk

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        if User.query.filter_by(username=username).first():
            flash("Username already exists", "error")
            return redirect(url_for('register'))
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for("dashboard"))
        flash("Invalid credentials", "error")
    return render_template("login.html")

@app.route("/export/pdf")
@login_required
def export_pdf():
    user = current_user
    reminders = Reminder.query.filter_by(user_id=user.id).all()
    symptoms = SymptomLog.query.filter_by(user_id=user.id).all()
    rendered = render_template("report_pdf.html", user=user, reminders=reminders, symptoms=symptoms)
    pdf = HTML(string=rendered).write_pdf()
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'inline; filename=health_report.pdf'
    return response

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        reminder = Reminder(
            name=request.form['name'],
            dosage=request.form['dosage'],
            time=request.form['time'],
            user=current_user
        )
        db.session.add(reminder)
        db.session.commit()
        return redirect(url_for("dashboard"))

    filter_option = request.args.get("filter", "all")
    query = Reminder.query.filter_by(user_id=current_user.id)

    if filter_option == "taken":
        query = query.filter_by(taken=True)
    elif filter_option == "pending":
        query = query.filter_by(taken=False)

    reminders = query.order_by(Reminder.time).all()
    risk_alerts = predict_missed_reminders(reminders) if filter_option != "taken" else []
    total = len(reminders)
    taken = sum(1 for r in reminders if r.taken)
    progress_percent = int((taken / total) * 100) if total else 0

    reminders_json = [
        {
            "id": r.id,
            "name": r.name,
            "dosage": r.dosage,
            "time": r.time,
            "taken": r.taken
        } for r in reminders
    ]

    return render_template(
        "dashboard.html",
        reminders=reminders,
        reminders_json=reminders_json,
        risk_alerts=risk_alerts,
        filter=filter_option,
        progress_percent=progress_percent
    )

@app.route("/reminder/<int:reminder_id>/mark", methods=["POST"])
@login_required
def mark_reminder_taken(reminder_id):
    reminder = Reminder.query.get_or_404(reminder_id)
    if reminder.user_id != current_user.id:
        flash("Unauthorized", "error")
        return redirect(url_for("dashboard"))

    reminder.taken = True
    reminder.time_taken = datetime.now().strftime("%H:%M")
    db.session.commit()
    flash(f"{reminder.name} marked as taken.", "success")
    return redirect(url_for("dashboard"))

@app.route("/reminder/<int:reminder_id>/undo", methods=["POST"])
@login_required
def undo_reminder_taken(reminder_id):
    reminder = Reminder.query.get_or_404(reminder_id)
    if reminder.user_id == current_user.id:
        reminder.taken = False
        reminder.time_taken = None
        db.session.commit()
        flash(f"{reminder.name} marked as pending.", "info")
    return redirect(url_for("dashboard", filter=request.args.get('filter', 'all')))

@app.route("/chatbot", methods=["GET", "POST"])
@login_required
def chatbot():
    reply = None
    if request.method == "POST":
        question = request.form.get('question', '').strip()
        
        if not question:
            reply = "⚠️ Please enter a question."
            return render_template("chatbot.html", reply=reply)

        try:
            print(f"Processing question: {question}")  # Debug log
            
            # Get the best available LLM
            llm_client, llm_source = get_llm_client()
            if llm_client is None:
                reply = "⚠️ No valid API keys found. Please check your GROQ_API_KEY or XAI_API_KEY configuration."
                return render_template("chatbot.html", reply=reply)
            
            print(f"Using {llm_source} for LLM")
            
            # Try to use vectorstore, but fallback to direct LLM if it fails
            vectorstore = get_vectorstore()
            
            if vectorstore is not None:
                print("Using vectorstore-based response")
                CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you don't know the answer, just say that you don't know, don't try to make up an answer.
                Don't provide anything out of the given context.

                Context: {context}
                Question: {question}

                Start the answer directly. No small talk please.
                """
                
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm_client,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': question})
                result = response["result"]
                source_documents = response["source_documents"]
                
                reply = f"🧠 **Answer** (via {llm_source}):\n{result}\n\n🔍 **Source Documents:**\n{format_sources(source_documents)}"
                sources_text = format_sources(source_documents)
            else:
                print("Vectorstore unavailable, using direct LLM response")
                # Direct LLM response without vectorstore
                medical_prompt = f"""You are a helpful medical information assistant. Please provide accurate, general medical information about: {question}

                Important: This is for educational purposes only and should not replace professional medical advice. Always recommend consulting healthcare providers for specific medical concerns.
                
                Question: {question}
                
                Please provide a helpful, accurate response:"""
                
                response = llm_client.invoke(medical_prompt)
                result = response.content if hasattr(response, 'content') else str(response)
                
                reply = f"🧠 **Answer** (via {llm_source} - Direct Mode):\n{result}\n\n⚠️ **Note**: Medical knowledge base temporarily unavailable. This response is from the AI model directly."
                sources_text = "Direct AI response - no medical database sources"

            # Save to chat history
            chat_entry = ChatHistory(
                user_id=current_user.id,
                question=question,
                answer=result,
                sources=sources_text
            )
            db.session.add(chat_entry)
            db.session.commit()

            print("Successfully processed query")

        except Exception as e:
            reply = f"⚠️ Error processing your question: {str(e)}"
            print(f"Medibot error: {e}")
            import traceback
            traceback.print_exc()
    
    return render_template("chatbot.html", reply=reply)


@app.route("/ocr-upload", methods=["GET", "POST"])
@login_required
def ocr_upload():
    extracted_text = ""
    if request.method == "POST":
        f = request.files['image']
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        image = cv2.imread(filepath)
        extracted_text = pytesseract.image_to_string(image)
    return render_template("ocr_upload.html", extracted_text=extracted_text)


@app.route("/analytics")
@login_required
def analytics():
    reminders = Reminder.query.filter_by(user_id=current_user.id).all()
    dosage_counts = Counter([r.dosage for r in reminders])
    total = len(reminders)
    taken = sum(1 for r in reminders if r.taken)
    missed = total - taken

    today = date.today()
    day_labels = [(today - timedelta(days=i)).strftime('%d %b') for i in reversed(range(7))]
    taken_by_day = {label: 0 for label in day_labels}
    missed_by_day = {label: 0 for label in day_labels}

    for r in reminders:
        if r.created_at:
            label = r.created_at.date().strftime('%d %b')
            # If label not in day_labels, you can add it or skip
            if label in taken_by_day:
                if r.taken:
                    taken_by_day[label] += 1
                else:
                    missed_by_day[label] += 1
            else:
                # Optionally, add to the first label (oldest day)
                if r.taken:
                    taken_by_day[day_labels[0]] += 1
                else:
                    missed_by_day[day_labels[0]] += 1

    return render_template("analytics.html",
                           dosage_counts=dosage_counts,
                           total=total,
                           taken=taken,
                           missed=missed,
                           day_labels=day_labels,
                           taken_data=[taken_by_day[d] for d in day_labels],
                           missed_data=[missed_by_day[d] for d in day_labels])


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = current_user
    bmi = bmr = hydration = None
    if user.height_cm and user.weight_kg:
        height_m = user.height_cm / 100
        bmi = round(user.weight_kg / (height_m ** 2), 2)
        bmr = round(10 * user.weight_kg + 6.25 * user.height_cm - 5 * user.age + 5, 2)
        hydration = round(user.weight_kg * 0.033, 2)
    if request.method == "POST":
        user.name = request.form['name']
        user.age = request.form['age']
        user.height_cm = request.form['height_cm']
        user.weight_kg = request.form['weight_kg']
        user.blood_group = request.form['blood_group']
        user.conditions = request.form['conditions']
        db.session.commit()
        flash("Profile updated", "success")
    return render_template("profile.html", user=user, bmi=bmi, bmr=bmr, hydration=hydration)

@app.route("/symptoms", methods=["GET", "POST"])
@login_required
def symptoms():
    if request.method == "POST":
        log = SymptomLog(
            date=request.form['date'],
            symptom=request.form['symptom'],
            severity=request.form['severity'],
            user_id=current_user.id
        )
        db.session.add(log)
        db.session.commit()
    logs = SymptomLog.query.filter_by(user_id=current_user.id).order_by(SymptomLog.date.desc()).all()
    return render_template("symptoms.html", logs=logs)

@app.route("/uploads", methods=["GET", "POST"])
@login_required
def uploads():
    if request.method == "POST":
        f = request.files['document']
        category = request.form['category']
        if f:
            filename = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            doc = Document(filename=filename, category=category, user_id=current_user.id)
            db.session.add(doc)
            db.session.commit()
    files = Document.query.filter_by(user_id=current_user.id).all()
    return render_template("uploads.html", files=files)

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/export/reminders")
@login_required
def export_reminders():
    reminders = Reminder.query.filter_by(user_id=current_user.id).all()
    def generate():
        yield "Name,Dosage,Time,Taken,Time Taken\n"
        for r in reminders:
            yield f"{r.name},{r.dosage},{r.time},{r.taken},{r.time_taken or ''}\n"
    return Response(generate(), mimetype='text/csv',
                    headers={"Content-Disposition": "attachment; filename=reminders.csv"})

@app.route("/symptom-checker", methods=["GET", "POST"])
@login_required
def symptom_checker():
    diagnosis = None
    risk_level = None
    recommendations = []
    
    if request.method == "POST":
        symptoms = request.form.getlist('symptoms')
        duration = request.form.get('duration')
        severity = request.form.get('severity')
        age = current_user.age or 25
        
        if symptoms:
            # Create a symptom analysis prompt
            symptoms_text = ", ".join(symptoms)
            symptom_query = f"""
            A {age}-year-old patient presents with the following symptoms: {symptoms_text}.
            Duration: {duration}. Severity: {severity}/10.
            
            Please provide:
            1. Possible conditions or diagnoses
            2. Risk level (Low/Medium/High)
            3. Recommended next steps
            4. When to seek immediate medical attention
            
            Be thorough but emphasize the need for professional medical evaluation.
            """
            
            try:
                llm_client, llm_source = get_llm_client()
                if llm_client:
                    vectorstore = get_vectorstore()
                    
                    if vectorstore:
                        # Use vectorstore-based analysis
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm_client,
                            chain_type="stuff",
                            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                            return_source_documents=True,
                            chain_type_kwargs={'prompt': set_custom_prompt("""
                                Use the medical information to analyze the symptoms and provide guidance.
                                
                                Context: {context}
                                Question: {question}
                                
                                Provide a structured response with possible conditions, risk assessment, and recommendations.
                            """)}
                        )
                        
                        response = qa_chain.invoke({'query': symptom_query})
                        diagnosis = response["result"]
                    else:
                        # Direct LLM analysis without vectorstore
                        medical_prompt = f"""You are a medical AI assistant. Analyze these symptoms and provide guidance:

{symptom_query}

Please provide:
1. Possible conditions that might cause these symptoms
2. General recommendations
3. When to seek medical attention

Important: This is for informational purposes only. Always recommend consulting healthcare providers."""
                        
                        response = llm_client.invoke(medical_prompt)
                        diagnosis = response.content if hasattr(response, 'content') else str(response)
                        diagnosis += "\n\n⚠️ **Note**: Medical knowledge base temporarily unavailable. This analysis is from the AI model directly."
                    
                    # Simple risk assessment based on symptoms
                    high_risk_symptoms = ['chest pain', 'difficulty breathing', 'severe headache', 'loss of consciousness']
                    if any(symptom.lower() in symptoms_text.lower() for symptom in high_risk_symptoms):
                        risk_level = "High"
                    elif int(severity) >= 7:
                        risk_level = "Medium"
                    else:
                        risk_level = "Low"
                    
                    # Generate recommendations
                    if risk_level == "High":
                        recommendations = [
                            "Seek immediate medical attention",
                            "Consider visiting emergency room",
                            "Contact your healthcare provider immediately"
                        ]
                    elif risk_level == "Medium":
                        recommendations = [
                            "Schedule an appointment with your doctor",
                            "Monitor symptoms closely",
                            "Rest and stay hydrated"
                        ]
                    else:
                        recommendations = [
                            "Monitor symptoms for 24-48 hours",
                            "Consider over-the-counter remedies if appropriate",
                            "Contact doctor if symptoms worsen"
                        ]
                else:
                    diagnosis = "⚠️ No valid API keys found. Please check your GROQ_API_KEY or XAI_API_KEY configuration."
                    risk_level = "Unknown"
                            
            except Exception as e:
                diagnosis = f"Error analyzing symptoms: {str(e)}"
                risk_level = "Unknown"
    
    common_symptoms = [
        'Headache', 'Fever', 'Cough', 'Sore throat', 'Nausea', 'Fatigue',
        'Chest pain', 'Difficulty breathing', 'Abdominal pain', 'Dizziness',
        'Back pain', 'Joint pain', 'Rash', 'Loss of appetite', 'Insomnia'
    ]
    
    return render_template("symptom_checker.html", 
                         diagnosis=diagnosis, 
                         risk_level=risk_level,
                         recommendations=recommendations,
                         common_symptoms=common_symptoms)

@app.route("/drug-interactions", methods=["GET", "POST"])
@login_required
def drug_interactions():
    interactions = None
    risk_level = None
    
    if request.method == "POST":
        medications = [med.strip() for med in request.form.get('medications', '').split(',') if med.strip()]
        
        if len(medications) >= 2:
            # Create drug interaction query
            meds_text = ", ".join(medications)
            interaction_query = f"""
            Check for drug interactions between these medications: {meds_text}
            
            Please provide:
            1. Known interactions between these drugs
            2. Severity of interactions (Minor/Moderate/Major)
            3. Potential side effects or complications
            4. Recommendations for safe use
            5. When to consult a healthcare provider
            
            If no interactions are found in the medical literature, state that clearly.
            """
            
            try:
                llm_client, llm_source = get_llm_client()
                if llm_client:
                    vectorstore = get_vectorstore()
                    
                    if vectorstore:
                        # Use vectorstore-based analysis
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm_client,
                            chain_type="stuff",
                            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                            return_source_documents=True,
                            chain_type_kwargs={'prompt': set_custom_prompt("""
                                Use the medical information to check for drug interactions.
                                
                                Context: {context}
                                Question: {question}
                                
                                Focus on drug interactions, contraindications, and safety information.
                            """)}
                        )
                        
                        response = qa_chain.invoke({'query': interaction_query})
                        interactions = response["result"]
                    else:
                        # Direct LLM analysis without vectorstore
                        medical_prompt = f"""You are a medical AI assistant specialized in drug interactions. Analyze these medications:

{interaction_query}

Please provide information about:
1. Known interactions between these medications
2. Severity levels
3. Safety recommendations
4. When to consult healthcare providers

Important: This is for informational purposes only. Always consult pharmacists or doctors for medication advice."""
                        
                        response = llm_client.invoke(medical_prompt)
                        interactions = response.content if hasattr(response, 'content') else str(response)
                        interactions += "\n\n⚠️ **Note**: Medical knowledge base temporarily unavailable. This analysis is from the AI model directly."
                    
                    # Simple risk assessment based on keywords
                    if any(word in interactions.lower() for word in ['major', 'severe', 'contraindicated', 'dangerous']):
                        risk_level = "High"
                    elif any(word in interactions.lower() for word in ['moderate', 'caution', 'monitor']):
                        risk_level = "Medium"
                    else:
                        risk_level = "Low"
                else:
                    interactions = "⚠️ No valid API keys found. Please check your GROQ_API_KEY or XAI_API_KEY configuration."
                    risk_level = "Unknown"
                            
            except Exception as e:
                interactions = f"Error checking interactions: {str(e)}"
                risk_level = "Unknown"
        else:
            interactions = "Please enter at least 2 medications to check for interactions."
    
    # Get user's current medications
    user_medications = []
    try:
        reminders = Reminder.query.filter_by(user_id=current_user.id).all()
        user_medications = list(set([r.name for r in reminders if r.name]))
    except:
        pass
    
    return render_template("drug_interactions.html", 
                         interactions=interactions,
                         risk_level=risk_level,
                         user_medications=user_medications)


@app.route("/debug/test-ai", methods=["GET"])
@login_required
def test_ai_components():
    """Test route to verify AI components are working"""
    results = {
        "groq_api_key": "❌ Not found",
        "vectorstore": "❌ Failed to load",
        "groq_connection": "❌ Failed",
        "search_test": "❌ Failed"
    }
    
    try:
        # Test GROQ API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and len(groq_api_key) >= 50:
            results["groq_api_key"] = f"✅ Found ({len(groq_api_key)} chars)"
        else:
            results["groq_api_key"] = f"❌ Invalid or too short ({len(groq_api_key) if groq_api_key else 0} chars)"
        
        # Test vectorstore
        vectorstore = get_vectorstore()
        if vectorstore:
            results["vectorstore"] = "✅ Loaded successfully"
            
            # Test search functionality
            try:
                docs = vectorstore.similarity_search("blood pressure", k=2)
                results["search_test"] = f"✅ Found {len(docs)} documents"
            except Exception as e:
                results["search_test"] = f"❌ Search failed: {str(e)}"
        else:
            results["vectorstore"] = "❌ Failed to load"
        
        # Test GROQ connection
        if groq_api_key and len(groq_api_key) >= 50:
            try:
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model_name="llama3-8b-8192",
                    temperature=0.0,
                    groq_api_key=groq_api_key,
                )
                # Don't actually call the API, just test initialization
                results["groq_connection"] = "✅ Client created successfully"
            except Exception as e:
                results["groq_connection"] = f"❌ Failed: {str(e)}"
    
    except Exception as e:
        results["general_error"] = f"❌ Unexpected error: {str(e)}"
    
    # Return as simple HTML page
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Components Test</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .success {{ color: green; }}
            .error {{ color: red; }}
            .result {{ margin: 10px 0; padding: 10px; border: 1px solid #ccc; }}
        </style>
    </head>
    <body>
        <h1>AI Components Test Results</h1>
        <div class="result">
            <strong>GROQ API Key:</strong> {results["groq_api_key"]}
        </div>
        <div class="result">
            <strong>Vectorstore:</strong> {results["vectorstore"]}
        </div>
        <div class="result">
            <strong>GROQ Connection:</strong> {results["groq_connection"]}
        </div>
        <div class="result">
            <strong>Search Test:</strong> {results["search_test"]}
        </div>
        {"<div class='result'><strong>General Error:</strong> " + results.get("general_error", "") + "</div>" if "general_error" in results else ""}
        <br>
        <a href="/chatbot">← Back to Chatbot</a>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    with app.app_context():
        db.create_all()
    
    # Production vs Development settings
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
