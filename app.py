from flask import Flask, render_template, redirect, request, url_for, flash, jsonify, send_from_directory, Response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, date, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import os
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from flask import make_response

# Optional imports for OCR and PDF (may not work in all environments)
try:
    import pytesseract
    import cv2
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ OCR not available: pytesseract/cv2 not installed")

try:
    from weasyprint import HTML
    PDF_AVAILABLE = True
except (ImportError, OSError) as e:
    PDF_AVAILABLE = False
    print(f"⚠️ PDF export not available: {e}")

# API-Free Web Scraper for real-time medical data
from web_scraper import (
    MedicalWebScraper, 
    get_medical_info, 
    check_drug_interactions as scraper_check_drug_interactions,
    analyze_symptoms as scraper_analyze_symptoms,
    MEDICAL_DATABASE,
    check_food_interactions
)

app = Flask(__name__)

# Production-ready configuration
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')

# Database configuration (supports Supabase PostgreSQL)
database_url = os.getenv('DATABASE_URL', 'sqlite:///meds.db')
# Fix for Supabase/Heroku postgres:// URLs (SQLAlchemy requires postgresql://)
if database_url.startswith('postgres://'):
    database_url = database_url.replace('postgres://', 'postgresql://', 1)
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Initialize the API-free medical web scraper
medical_scraper = MedicalWebScraper()
print(f"🚀 MedTrack: Web Scraper Initialized (Wikipedia/MedlinePlus)")

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
    expiry_date = db.Column(db.Date, nullable=True)  # Medication expiry tracking
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
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

class SideEffect(db.Model):
    """Track medication side effects"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    medication_name = db.Column(db.String(150))
    side_effect = db.Column(db.String(200))
    severity = db.Column(db.Integer)  # 1=Mild, 2=Moderate, 3=Severe
    time_after_dose = db.Column(db.String(50))  # e.g., "2 hours", "30 minutes"
    notes = db.Column(db.Text)
    logged_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    question = db.Column(db.Text)
    answer = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    sources = db.Column(db.Text)  # JSON string of source documents

# ============ NEW FEATURES ============

class WaterLog(db.Model):
    """Track daily water intake"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.Date, default=date.today)
    glasses = db.Column(db.Integer, default=0)  # Number of glasses (250ml each)
    goal = db.Column(db.Integer, default=8)  # Daily goal
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Appointment(db.Model):
    """Doctor appointments scheduler"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    doctor_name = db.Column(db.String(150))
    specialty = db.Column(db.String(100))
    date = db.Column(db.Date)
    time = db.Column(db.String(10))
    location = db.Column(db.String(200))
    notes = db.Column(db.Text)
    reminder_sent = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class EmergencyContact(db.Model):
    """Emergency/ICE contacts"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    name = db.Column(db.String(150))
    relationship = db.Column(db.String(50))
    phone = db.Column(db.String(20))
    email = db.Column(db.String(150))
    is_primary = db.Column(db.Boolean, default=False)

class FamilyMember(db.Model):
    """Family member profiles for health tracking"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))  # Parent account
    name = db.Column(db.String(150))
    relationship = db.Column(db.String(50))
    age = db.Column(db.Integer)
    blood_group = db.Column(db.String(10))
    conditions = db.Column(db.Text)
    allergies = db.Column(db.Text)

class UserStreak(db.Model):
    """Gamification - track medication adherence streaks"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    current_streak = db.Column(db.Integer, default=0)
    longest_streak = db.Column(db.Integer, default=0)
    total_days_tracked = db.Column(db.Integer, default=0)
    last_activity_date = db.Column(db.Date)
    badges = db.Column(db.Text)  # JSON string of earned badges
    points = db.Column(db.Integer, default=0)

class HealthNews(db.Model):
    """Cached health news articles"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300))
    summary = db.Column(db.Text)
    source = db.Column(db.String(100))
    url = db.Column(db.String(500))
    category = db.Column(db.String(50))
    fetched_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables on startup (works with gunicorn in production)
# Must be after all models are defined
with app.app_context():
    try:
        db.create_all()
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")

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
        try:
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
        except Exception as e:
            db.session.rollback()
            print(f"Registration error: {e}")
            flash("Registration failed. Please try again.", "error")
            return redirect(url_for('register'))
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
    if not PDF_AVAILABLE:
        flash("PDF export is not available in this environment.", "error")
        return redirect(url_for("dashboard"))
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
    return redirect(url_for("index"))

@app.route("/", methods=["GET", "POST"])
def index():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("landing"))

@app.route("/landing")
def landing():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return render_template("landing.html")

@app.route("/dashboard", methods=["GET", "POST"])
@login_required
def dashboard():
    if request.method == "POST":
        expiry_date_str = request.form.get('expiry_date')
        expiry_date = datetime.strptime(expiry_date_str, '%Y-%m-%d').date() if expiry_date_str else None
        
        reminder = Reminder(
            name=request.form['name'],
            dosage=request.form['dosage'],
            time=request.form['time'],
            expiry_date=expiry_date,
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
    
    # Check for expiring medications
    today = date.today()
    expiry_warnings = []
    for r in reminders:
        if r.expiry_date:
            days_until_expiry = (r.expiry_date - today).days
            if days_until_expiry < 0:
                expiry_warnings.append((r, 'expired', abs(days_until_expiry)))
            elif days_until_expiry <= 7:
                expiry_warnings.append((r, 'expiring_soon', days_until_expiry))
            elif days_until_expiry <= 30:
                expiry_warnings.append((r, 'expiring_month', days_until_expiry))
    
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
        expiry_warnings=expiry_warnings,
        filter=filter_option,
        progress_percent=progress_percent
    )

@app.route("/reminder/<int:reminder_id>/mark", methods=["POST"])
@login_required
def mark_reminder_taken(reminder_id):
    reminder = Reminder.query.get_or_404(reminder_id)
    if reminder.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    
    reminder.taken = True
    reminder.time_taken = datetime.now().strftime("%I:%M %p")
    db.session.commit()
    
    # Update streak
    streak = UserStreak.query.filter_by(user_id=current_user.id).first()
    if streak:
        today = date.today()
        if streak.last_activity_date != today:
            if streak.last_activity_date == today - timedelta(days=1):
                streak.current_streak += 1
            else:
                streak.current_streak = 1
            streak.last_activity_date = today
            streak.total_days_tracked += 1
            if streak.current_streak > streak.longest_streak:
                streak.longest_streak = streak.current_streak
            db.session.commit()
    
    return redirect(url_for("dashboard"))

@app.route("/side-effects", methods=["GET", "POST"])
@login_required
def side_effects():
    if request.method == "POST":
        side_effect = SideEffect(
            user_id=current_user.id,
            medication_name=request.form['medication_name'],
            side_effect=request.form['side_effect'],
            severity=int(request.form['severity']),
            time_after_dose=request.form.get('time_after_dose', ''),
            notes=request.form.get('notes', '')
        )
        db.session.add(side_effect)
        db.session.commit()
        flash('Side effect logged successfully!', 'success')
        return redirect(url_for('side_effects'))
    
    # Get all side effects
    effects = SideEffect.query.filter_by(user_id=current_user.id).order_by(SideEffect.logged_at.desc()).all()
    
    # Pattern detection: group by medication and side effect
    patterns = {}
    for effect in effects:
        key = (effect.medication_name, effect.side_effect)
        if key not in patterns:
            patterns[key] = {'count': 0, 'severity_avg': 0, 'severities': []}
        patterns[key]['count'] += 1
        patterns[key]['severities'].append(effect.severity)
    
    # Calculate averages and identify frequent patterns
    frequent_patterns = []
    for (med, side_eff), data in patterns.items():
        if data['count'] >= 2:  # Show patterns that occurred at least twice
            avg_severity = sum(data['severities']) / len(data['severities'])
            frequent_patterns.append({
                'medication': med,
                'side_effect': side_eff,
                'occurrences': data['count'],
                'avg_severity': round(avg_severity, 1)
            })
    
    # Get user's medications for dropdown
    medications = Reminder.query.filter_by(user_id=current_user.id).all()
    
    return render_template('side_effects.html', 
                         effects=effects, 
                         patterns=frequent_patterns,
                         medications=medications)


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
            print(f"Processing question: {question}")
            
            # Use API-free web scraper for real-time medical information
            print("🌐 Fetching real-time data from medical websites...")
            result = get_medical_info(question, medical_scraper)
            reply = f"🩺 **Medical Information** (Real-time Web Data):\n\n{result}"
            sources_text = "Wikipedia, MedlinePlus, Medical Database"
            print("✅ Successfully fetched medical information")
            
            # Save to chat history
            try:
                chat_entry = ChatHistory(
                    user_id=current_user.id,
                    question=question,
                    answer=result,
                    sources=sources_text
                )
                db.session.add(chat_entry)
                db.session.commit()
                print("Chat history saved successfully")
            except Exception as db_error:
                print(f"Database save failed: {db_error}")

        except Exception as e:
            print(f"Error in chatbot: {e}")
            traceback.print_exc()
            reply = f"⚠️ Sorry, there was an error processing your question. Please try again."
    
    return render_template("chatbot.html", reply=reply)


@app.route("/ocr-upload", methods=["GET", "POST"])
@login_required
def ocr_upload():
    extracted_text = ""
    if request.method == "POST":
        if not OCR_AVAILABLE:
            flash("OCR is not available in this environment.", "error")
            return render_template("ocr_upload.html", extracted_text=extracted_text)
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
        
        if symptoms:
            try:
                print(f"🌐 Analyzing symptoms: {symptoms}")
                
                # Use API-free web scraper for symptom analysis
                diagnosis, risk_level, recommendations = scraper_analyze_symptoms(
                    symptoms, duration, severity, scraper=medical_scraper
                )
                
                # Try to get additional real-time info from Wikipedia
                try:
                    for symptom in symptoms[:2]:  # Limit to avoid too many requests
                        wiki_info = medical_scraper.search_wikipedia(f"{symptom} medical symptom")
                        if wiki_info:
                            diagnosis += f"\n\n📖 **Additional Info ({wiki_info['title']}):**\n{wiki_info['content'][:500]}..."
                except Exception as web_error:
                    print(f"Web scraping error: {web_error}")
                
                print(f"✅ Symptom analysis complete - Risk: {risk_level}")
                
            except Exception as e:
                print(f"Symptom analysis error: {e}")
                traceback.print_exc()
                diagnosis = "⚠️ Error analyzing symptoms. Please try again."
                risk_level = "Unknown"
                recommendations = ["Please consult a healthcare professional"]
    
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
            try:
                print(f"🌐 Checking drug interactions for: {medications}")
                
                # Use API-free web scraper for drug interaction analysis
                interactions, risk_level = scraper_check_drug_interactions(medications)
                
                # Try to get additional drug info from Wikipedia
                try:
                    for med in medications[:2]:  # Limit to avoid too many requests
                        drug_info = medical_scraper.get_drug_info_drugbank(med)
                        if drug_info:
                            interactions += f"\n\n📖 **Drug Info ({med.title()}):**\n{drug_info['info'][:400]}..."
                except Exception as web_error:
                    print(f"Web scraping error: {web_error}")
                
                print(f"✅ Drug interaction check complete - Risk: {risk_level}")
                
            except Exception as e:
                print(f"Drug interaction analysis error: {e}")
                traceback.print_exc()
                interactions = "⚠️ Error checking drug interactions. Please try again."
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


# ============ NEW FEATURE ROUTES ============

# --- WATER TRACKER ---
@app.route("/water-tracker", methods=["GET", "POST"])
@login_required
def water_tracker():
    today = date.today()
    water_log = WaterLog.query.filter_by(user_id=current_user.id, date=today).first()
    
    if not water_log:
        # Calculate goal based on user weight
        goal = 8  # Default 8 glasses
        if current_user.weight_kg:
            goal = int(current_user.weight_kg * 0.033 * 4)  # ~4 glasses per liter
        water_log = WaterLog(user_id=current_user.id, date=today, glasses=0, goal=goal)
        db.session.add(water_log)
        db.session.commit()
    
    if request.method == "POST":
        action = request.form.get('action')
        if action == 'add':
            water_log.glasses += 1
        elif action == 'remove' and water_log.glasses > 0:
            water_log.glasses -= 1
        elif action == 'set_goal':
            water_log.goal = int(request.form.get('goal', 8))
        db.session.commit()
        return redirect(url_for('water_tracker'))
    
    # Get weekly history
    week_data = []
    for i in range(7):
        d = today - timedelta(days=i)
        log = WaterLog.query.filter_by(user_id=current_user.id, date=d).first()
        week_data.append({
            'date': d.strftime('%a'),
            'glasses': log.glasses if log else 0,
            'goal': log.goal if log else 8
        })
    week_data.reverse()
    
    return render_template("water_tracker.html", 
                         water_log=water_log, 
                         week_data=week_data,
                         progress=int((water_log.glasses / water_log.goal) * 100) if water_log.goal else 0)


# --- APPOINTMENTS ---
@app.route("/appointments", methods=["GET", "POST"])
@login_required
def appointments():
    if request.method == "POST":
        appointment = Appointment(
            user_id=current_user.id,
            doctor_name=request.form['doctor_name'],
            specialty=request.form.get('specialty', ''),
            date=datetime.strptime(request.form['date'], '%Y-%m-%d').date(),
            time=request.form['time'],
            location=request.form.get('location', ''),
            notes=request.form.get('notes', '')
        )
        db.session.add(appointment)
        db.session.commit()
        flash("Appointment scheduled successfully!", "success")
        return redirect(url_for('appointments'))
    
    upcoming = Appointment.query.filter(
        Appointment.user_id == current_user.id,
        Appointment.date >= date.today()
    ).order_by(Appointment.date, Appointment.time).all()
    
    past = Appointment.query.filter(
        Appointment.user_id == current_user.id,
        Appointment.date < date.today()
    ).order_by(Appointment.date.desc()).limit(10).all()
    
    return render_template("appointments.html", upcoming=upcoming, past=past)


@app.route("/appointment/<int:id>/delete", methods=["POST"])
@login_required
def delete_appointment(id):
    appointment = Appointment.query.get_or_404(id)
    if appointment.user_id == current_user.id:
        db.session.delete(appointment)
        db.session.commit()
        flash("Appointment deleted", "info")
    return redirect(url_for('appointments'))


# --- EMERGENCY CONTACTS ---
@app.route("/emergency-contacts", methods=["GET", "POST"])
@login_required
def emergency_contacts():
    if request.method == "POST":
        # If setting as primary, unset others
        if request.form.get('is_primary'):
            EmergencyContact.query.filter_by(user_id=current_user.id).update({'is_primary': False})
        
        contact = EmergencyContact(
            user_id=current_user.id,
            name=request.form['name'],
            relationship=request.form['relationship'],
            phone=request.form['phone'],
            email=request.form.get('email', ''),
            is_primary=bool(request.form.get('is_primary'))
        )
        db.session.add(contact)
        db.session.commit()
        flash("Emergency contact added!", "success")
        return redirect(url_for('emergency_contacts'))
    
    contacts = EmergencyContact.query.filter_by(user_id=current_user.id).order_by(EmergencyContact.is_primary.desc()).all()
    return render_template("emergency_contacts.html", contacts=contacts)


@app.route("/emergency-contacts/<int:id>/delete", methods=["POST"])
@login_required
def delete_emergency_contact(id):
    contact = EmergencyContact.query.get_or_404(id)
    if contact.user_id == current_user.id:
        db.session.delete(contact)
        db.session.commit()
        flash("Contact deleted", "info")
    return redirect(url_for('emergency_contacts'))


@app.route("/emergency/card")
@login_required
def emergency_card():
    contacts = EmergencyContact.query.filter_by(user_id=current_user.id).all()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    return render_template("emergency_card.html", user=current_user, contacts=contacts, now=now)


# --- FAMILY PROFILES ---
@app.route("/family", methods=["GET", "POST"])
@login_required
def family_profiles():
    if request.method == "POST":
        member = FamilyMember(
            user_id=current_user.id,
            name=request.form['name'],
            relationship=request.form['relationship'],
            age=int(request.form.get('age', 0)) if request.form.get('age') else None,
            blood_group=request.form.get('blood_group', ''),
            conditions=request.form.get('conditions', ''),
            allergies=request.form.get('allergies', '')
        )
        db.session.add(member)
        db.session.commit()
        flash(f"Added {member.name} to family profiles!", "success")
        return redirect(url_for('family_profiles'))
    
    members = FamilyMember.query.filter_by(user_id=current_user.id).all()
    return render_template("family_profiles.html", members=members)


@app.route("/family/<int:id>/delete", methods=["POST"])
@login_required
def delete_family_member(id):
    member = FamilyMember.query.get_or_404(id)
    if member.user_id == current_user.id:
        db.session.delete(member)
        db.session.commit()
        flash("Family member removed", "info")
    return redirect(url_for('family_profiles'))


# --- GAMIFICATION & STREAKS ---
def update_user_streak(user_id):
    """Update user's medication streak and points"""
    today = date.today()
    streak = UserStreak.query.filter_by(user_id=user_id).first()
    
    if not streak:
        streak = UserStreak(user_id=user_id, current_streak=0, longest_streak=0, 
                           total_days_tracked=0, points=0, badges='[]')
        db.session.add(streak)
    
    # Check if user took all medications today
    today_reminders = Reminder.query.filter_by(user_id=user_id).all()
    all_taken = all(r.taken for r in today_reminders) if today_reminders else False
    
    if all_taken:
        if streak.last_activity_date == today - timedelta(days=1):
            streak.current_streak += 1
        elif streak.last_activity_date != today:
            streak.current_streak = 1
        
        streak.last_activity_date = today
        streak.total_days_tracked += 1
        streak.points += 10  # Points for completing daily meds
        
        # Update longest streak
        if streak.current_streak > streak.longest_streak:
            streak.longest_streak = streak.current_streak
        
        # Award badges
        import json
        badges = json.loads(streak.badges) if streak.badges else []
        
        if streak.current_streak >= 7 and 'week_warrior' not in badges:
            badges.append('week_warrior')
            streak.points += 50
        if streak.current_streak >= 30 and 'month_master' not in badges:
            badges.append('month_master')
            streak.points += 200
        if streak.total_days_tracked >= 100 and 'century_club' not in badges:
            badges.append('century_club')
            streak.points += 500
        
        streak.badges = json.dumps(badges)
    
    db.session.commit()
    return streak


@app.route("/streaks")
@login_required
def streaks():
    streak = update_user_streak(current_user.id)
    import json
    badges = json.loads(streak.badges) if streak and streak.badges else []
    
    badge_info = {
        'week_warrior': {'name': '7-Day Warrior', 'icon': '🏆', 'desc': 'Complete 7 days streak'},
        'month_master': {'name': 'Month Master', 'icon': '🥇', 'desc': 'Complete 30 days streak'},
        'century_club': {'name': 'Century Club', 'icon': '💯', 'desc': 'Track 100 total days'},
        'early_bird': {'name': 'Early Bird', 'icon': '🌅', 'desc': 'Take morning meds on time'},
        'night_owl': {'name': 'Night Owl', 'icon': '🦉', 'desc': 'Never miss evening meds'},
    }
    
    return render_template("streaks.html", 
                         streak=streak, 
                         badges=badges, 
                         badge_info=badge_info)


# --- DOSAGE CALCULATOR ---
@app.route("/dosage-calculator", methods=["GET", "POST"])
@login_required
def dosage_calculator():
    result = None
    if request.method == "POST":
        calc_type = request.form.get('calc_type')
        
        if calc_type == 'weight_based':
            weight = float(request.form.get('weight', 0))
            dose_per_kg = float(request.form.get('dose_per_kg', 0))
            result = {
                'type': 'Weight-based Dosage',
                'calculation': f"{weight} kg × {dose_per_kg} mg/kg",
                'result': f"{weight * dose_per_kg:.2f} mg",
                'notes': 'Always verify with your healthcare provider'
            }
        
        elif calc_type == 'liquid':
            desired_dose = float(request.form.get('desired_dose', 0))
            concentration = float(request.form.get('concentration', 1))
            result = {
                'type': 'Liquid Medication',
                'calculation': f"{desired_dose} mg ÷ {concentration} mg/mL",
                'result': f"{desired_dose / concentration:.2f} mL",
                'notes': 'Use accurate measuring device'
            }
        
        elif calc_type == 'bmi':
            weight = float(request.form.get('weight', current_user.weight_kg or 70))
            height = float(request.form.get('height', current_user.height_cm or 170))
            bmi = weight / ((height / 100) ** 2)
            category = 'Normal' if 18.5 <= bmi < 25 else ('Underweight' if bmi < 18.5 else ('Overweight' if bmi < 30 else 'Obese'))
            result = {
                'type': 'BMI Calculator',
                'calculation': f"{weight} kg ÷ ({height/100:.2f} m)²",
                'result': f"{bmi:.1f} ({category})",
                'notes': 'BMI categories: <18.5 Underweight, 18.5-24.9 Normal, 25-29.9 Overweight, 30+ Obese'
            }
    
    return render_template("dosage_calculator.html", result=result, user=current_user)


# --- HEALTH NEWS (Scraped) ---
@app.route("/health-news")
@login_required
def health_news():
    # Try to get fresh news from web scraping
    news_items = []
    try:
        # Use our medical scraper to get health topics
        topics = ['diabetes prevention', 'heart health tips', 'mental wellness', 
                  'nutrition advice', 'exercise benefits', 'sleep quality']
        
        for topic in topics[:4]:  # Limit to avoid too many requests
            wiki_result = medical_scraper.search_wikipedia(topic)
            if wiki_result:
                news_items.append({
                    'title': wiki_result['title'],
                    'summary': wiki_result['content'][:300] + '...',
                    'source': 'Wikipedia',
                    'url': wiki_result['source'],
                    'category': topic.split()[0].title()
                })
    except Exception as e:
        print(f"Error fetching health news: {e}")
    
    # Also get some health tips from local database
    health_tips = [
        {'title': 'Stay Hydrated', 'tip': 'Drink at least 8 glasses of water daily for optimal health.', 'icon': '💧'},
        {'title': 'Regular Exercise', 'tip': '150 minutes of moderate exercise per week reduces disease risk.', 'icon': '🏃'},
        {'title': 'Quality Sleep', 'tip': 'Adults need 7-9 hours of sleep for proper body recovery.', 'icon': '😴'},
        {'title': 'Balanced Diet', 'tip': 'Include fruits, vegetables, and whole grains in every meal.', 'icon': '🥗'},
        {'title': 'Mental Health', 'tip': 'Practice mindfulness and take breaks to reduce stress.', 'icon': '🧘'},
        {'title': 'Regular Checkups', 'tip': 'Annual health screenings catch problems early.', 'icon': '🏥'},
    ]
    
    return render_template("health_news.html", news_items=news_items, health_tips=health_tips)


# --- WEEKLY HEALTH SUMMARY ---
@app.route("/health-summary")
@login_required
def health_summary():
    today = date.today()
    week_start = today - timedelta(days=7)
    
    # Medication adherence
    week_reminders = Reminder.query.filter_by(user_id=current_user.id).all()
    total_meds = len(week_reminders)
    taken_meds = sum(1 for r in week_reminders if r.taken)
    adherence = int((taken_meds / total_meds * 100)) if total_meds else 0
    
    # Water intake
    water_logs = WaterLog.query.filter(
        WaterLog.user_id == current_user.id,
        WaterLog.date >= week_start
    ).all()
    avg_water = sum(w.glasses for w in water_logs) / len(water_logs) if water_logs else 0
    
    # Symptoms logged
    symptoms = SymptomLog.query.filter_by(user_id=current_user.id).order_by(SymptomLog.date.desc()).limit(7).all()
    
    # Upcoming appointments
    upcoming_appts = Appointment.query.filter(
        Appointment.user_id == current_user.id,
        Appointment.date >= today,
        Appointment.date <= today + timedelta(days=7)
    ).all()
    
    # Streak info
    streak = UserStreak.query.filter_by(user_id=current_user.id).first()
    
    summary = {
        'adherence': adherence,
        'total_meds': total_meds,
        'taken_meds': taken_meds,
        'avg_water': round(avg_water, 1),
        'symptom_count': len(symptoms),
        'upcoming_appts': len(upcoming_appts),
        'current_streak': streak.current_streak if streak else 0,
        'points': streak.points if streak else 0
    }
    
    return render_template("health_summary.html", summary=summary, symptoms=symptoms, appointments=upcoming_appts)


# --- THEME TOGGLE API ---
@app.route("/api/theme", methods=["POST"])
def toggle_theme():
    """API endpoint for theme toggle (handled by JS/localStorage)"""
    return jsonify({'status': 'ok'})


# --- FOOD SAFETY SCANNER ---
@app.route("/food-safety", methods=["GET", "POST"])
@login_required
def food_safety():
    results = None
    medications_list = []
    
    # Get user's current medications
    try:
        reminders = Reminder.query.filter_by(user_id=current_user.id).all()
        medications_list = list(set([r.name for r in reminders if r.name]))
    except Exception as e:
        print(f"Error fetching medications: {e}")
        
    if request.method == "POST":
        input_text = request.form.get('medications', '').strip()
        search_meds = []
        
        if input_text:
            search_meds = [m.strip() for m in input_text.split(',') if m.strip()]
        else:
            search_meds = medications_list
            
        if search_meds:
            try:
                print(f"🌐 Checking food interactions for: {search_meds}")
                results = check_food_interactions(search_meds, scraper=medical_scraper)
                print(f"✅ Food safety check complete - {len(results)} findings")
            except Exception as e:
                print(f"Food safety analysis error: {e}")
                traceback.print_exc()
        else:
            flash("Please enter medications or add them to your dashboard first.", "info")
            
    return render_template("food_safety.html", 
                         results=results, 
                         user_medications=medications_list)


if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    with app.app_context(): 
        db.create_all()
    
    # Production vs Development settings
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)