"""
API-Free Medical Information Web Scraper
Fetches real-time medical data from free public websites
"""

import requests
from bs4 import BeautifulSoup
import re
import json
from urllib.parse import quote_plus
import time

# User agent to mimic browser requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}

# Request timeout in seconds
TIMEOUT = 10


class MedicalWebScraper:
    """Scrapes medical information from free public websites"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.cache = {}  # Simple in-memory cache
        self.cache_expiry = 3600  # 1 hour cache
    
    def _get_cached(self, key):
        """Get cached response if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_expiry:
                return data
        return None
    
    def _set_cache(self, key, data):
        """Cache response with timestamp"""
        self.cache[key] = (data, time.time())
    
    def _make_request(self, url, cache_key=None):
        """Make HTTP request with error handling and caching"""
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        
        try:
            response = self.session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            if cache_key:
                self._set_cache(cache_key, response.text)
            return response.text
        except requests.RequestException as e:
            print(f"Request error for {url}: {e}")
            return None
    
    def search_wikipedia(self, query):
        """Search Wikipedia for medical information"""
        cache_key = f"wiki_{query}"
        
        try:
            # Wikipedia API for search
            search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={quote_plus(query)}&format=json"
            response = self._make_request(search_url, cache_key + "_search")
            
            if not response:
                return None
            
            data = json.loads(response)
            if not data.get('query', {}).get('search'):
                return None
            
            # Get the first result's page
            first_result = data['query']['search'][0]
            page_title = first_result['title']
            
            # Get page content
            content_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={quote_plus(page_title)}&prop=extracts&exintro=1&explaintext=1&format=json"
            content_response = self._make_request(content_url, cache_key + "_content")
            
            if not content_response:
                return None
            
            content_data = json.loads(content_response)
            pages = content_data.get('query', {}).get('pages', {})
            
            for page_id, page_info in pages.items():
                if 'extract' in page_info:
                    return {
                        'title': page_info.get('title', query),
                        'content': page_info['extract'],
                        'source': f"https://en.wikipedia.org/wiki/{quote_plus(page_title)}"
                    }
            
            return None
            
        except Exception as e:
            print(f"Wikipedia search error: {e}")
            return None
    
    
    
    def search_health_topic(self, topic):
        """Search for general health topics across multiple sources"""
        results = []
        
        # Try Wikipedia first (most reliable)
        wiki_result = self.search_wikipedia(f"{topic} health medicine")
        if wiki_result:
            results.append({
                'source': 'Wikipedia',
                'title': wiki_result['title'],
                'content': wiki_result['content'][:1500],
                'url': wiki_result['source']
            })
        
        return results
    
    


# Comprehensive local medical database for offline use
MEDICAL_DATABASE = {
    'symptoms': {
        'headache': {
            'causes': ['Tension/stress', 'Dehydration', 'Eye strain', 'Sinus issues', 'Migraine', 'High blood pressure', 'Lack of sleep'],
            'treatment': ['Rest in dark room', 'Stay hydrated', 'Over-the-counter pain relievers', 'Cold/warm compress', 'Reduce screen time'],
            'seek_help': ['Sudden severe headache', 'Headache after head injury', 'Fever with stiff neck', 'Vision changes', 'Confusion']
        },
        'fever': {
            'causes': ['Viral infection', 'Bacterial infection', 'Inflammatory conditions', 'Heat exhaustion', 'Some medications'],
            'treatment': ['Rest', 'Fluids', 'Fever reducers (acetaminophen/ibuprofen)', 'Cool compresses', 'Light clothing'],
            'seek_help': ['Temperature above 103°F (39.4°C)', 'Fever lasting more than 3 days', 'Difficulty breathing', 'Severe headache', 'Confusion']
        },
        'cough': {
            'causes': ['Common cold', 'Flu', 'Allergies', 'Asthma', 'GERD', 'Bronchitis', 'Pneumonia'],
            'treatment': ['Honey', 'Warm liquids', 'Humidifier', 'Cough drops', 'Rest', 'Stay hydrated'],
            'seek_help': ['Coughing blood', 'Difficulty breathing', 'High fever', 'Cough lasting more than 3 weeks', 'Wheezing']
        },
        'sore throat': {
            'causes': ['Viral infection', 'Bacterial infection (strep)', 'Allergies', 'Dry air', 'GERD', 'Muscle strain'],
            'treatment': ['Warm salt water gargle', 'Lozenges', 'Warm liquids', 'Rest voice', 'Humidifier'],
            'seek_help': ['Difficulty breathing', 'Difficulty swallowing', 'Severe pain', 'High fever', 'Rash']
        },
        'fatigue': {
            'causes': ['Lack of sleep', 'Stress', 'Anemia', 'Thyroid problems', 'Depression', 'Diabetes', 'Dehydration'],
            'treatment': ['Adequate sleep (7-9 hours)', 'Regular exercise', 'Balanced diet', 'Stress management', 'Reduce caffeine'],
            'seek_help': ['Fatigue lasting more than 2 weeks', 'Accompanied by unexplained weight loss', 'Fever', 'Shortness of breath', 'Severe pain']
        },
        'nausea': {
            'causes': ['Food poisoning', 'Motion sickness', 'Pregnancy', 'Medications', 'Migraine', 'Gastritis', 'Anxiety'],
            'treatment': ['Small sips of clear liquids', 'Ginger or ginger tea', 'Rest', 'Avoid strong odors', 'Eat bland foods (BRAT diet)'],
            'seek_help': ['Vomiting blood', 'Severe abdominal pain', 'High fever', 'Signs of dehydration', 'Confusion']
        },
        'chest pain': {
            'causes': ['Heart conditions', 'Muscle strain', 'Acid reflux', 'Anxiety', 'Respiratory infections', 'Costochondritis'],
            'treatment': ['Seek immediate medical attention for new chest pain', 'Rest if muscle-related', 'Antacids for acid reflux'],
            'seek_help': ['ANY new chest pain should be evaluated', 'Pain spreading to arm/jaw', 'Shortness of breath', 'Sweating', 'Nausea']
        },
        'dizziness': {
            'causes': ['Dehydration', 'Low blood pressure', 'Inner ear problems', 'Anemia', 'Medication side effects', 'Low blood sugar'],
            'treatment': ['Sit or lie down immediately', 'Stay hydrated', 'Avoid sudden movements', 'Eat regular meals'],
            'seek_help': ['Persistent dizziness', 'Accompanied by chest pain', 'Fainting', 'Severe headache', 'Numbness']
        },
        'back pain': {
            'causes': ['Muscle strain', 'Poor posture', 'Herniated disc', 'Arthritis', 'Osteoporosis', 'Kidney problems'],
            'treatment': ['Rest', 'Ice/heat therapy', 'Gentle stretching', 'Good posture', 'Over-the-counter pain relievers', 'Physical therapy'],
            'seek_help': ['Pain after injury', 'Numbness or tingling', 'Loss of bladder/bowel control', 'Fever with back pain', 'Unexplained weight loss']
        },
        'joint pain': {
            'causes': ['Arthritis', 'Injury', 'Overuse', 'Gout', 'Bursitis', 'Tendinitis', 'Infection'],
            'treatment': ['Rest', 'Ice/heat', 'Compression', 'Elevation', 'Gentle exercise', 'Over-the-counter pain relievers'],
            'seek_help': ['Severe swelling', 'Redness and warmth', 'Inability to move joint', 'Fever', 'Recent injury']
        }
    },
    'conditions': {
        'diabetes': {
            'description': 'A chronic condition affecting how the body processes blood sugar (glucose).',
            'types': ['Type 1 - Body doesn\'t produce insulin', 'Type 2 - Body doesn\'t use insulin properly', 'Gestational - Develops during pregnancy'],
            'symptoms': ['Increased thirst', 'Frequent urination', 'Fatigue', 'Blurred vision', 'Slow healing wounds', 'Unexplained weight loss'],
            'management': ['Blood sugar monitoring', 'Healthy diet', 'Regular exercise', 'Medication/insulin as prescribed', 'Regular checkups']
        },
        'hypertension': {
            'description': 'High blood pressure - a condition where blood pressure against artery walls is too high.',
            'levels': ['Normal: Less than 120/80 mmHg', 'Elevated: 120-129 systolic', 'Stage 1: 130-139/80-89', 'Stage 2: 140+/90+'],
            'symptoms': ['Often no symptoms (silent killer)', 'Severe cases: headaches, shortness of breath, nosebleeds'],
            'management': ['Reduce sodium intake', 'Regular exercise', 'Healthy weight', 'Limit alcohol', 'Manage stress', 'Medication if prescribed']
        },
        'asthma': {
            'description': 'A condition in which airways narrow and swell, producing extra mucus.',
            'triggers': ['Allergens', 'Exercise', 'Cold air', 'Smoke', 'Stress', 'Respiratory infections'],
            'symptoms': ['Shortness of breath', 'Chest tightness', 'Wheezing', 'Coughing', 'Difficulty sleeping due to breathing'],
            'management': ['Identify and avoid triggers', 'Use medications as prescribed', 'Monitor breathing', 'Have an action plan', 'Regular checkups']
        },
        'anxiety': {
            'description': 'A mental health disorder characterized by feelings of worry, anxiety, or fear.',
            'types': ['Generalized anxiety disorder', 'Panic disorder', 'Social anxiety', 'Phobias'],
            'symptoms': ['Excessive worry', 'Restlessness', 'Fatigue', 'Difficulty concentrating', 'Sleep problems', 'Physical symptoms (racing heart, sweating)'],
            'management': ['Therapy (CBT)', 'Medication if prescribed', 'Regular exercise', 'Stress management', 'Adequate sleep', 'Limit caffeine/alcohol']
        },
        'depression': {
            'description': 'A mood disorder that causes persistent feelings of sadness and loss of interest.',
            'symptoms': ['Persistent sad mood', 'Loss of interest', 'Sleep changes', 'Appetite changes', 'Fatigue', 'Difficulty concentrating', 'Thoughts of death/suicide'],
            'management': ['Therapy', 'Medication if prescribed', 'Regular exercise', 'Social support', 'Healthy routine', 'Professional help'],
            'crisis': 'If you or someone you know is in crisis, call 988 (Suicide & Crisis Lifeline) or go to nearest emergency room.'
        }
    },
    'medications': {
        'common_drugs': {
            'acetaminophen': {
                'uses': 'Pain relief, fever reduction',
                'brands': ['Tylenol', 'Panadol'],
                'warnings': 'Avoid alcohol, don\'t exceed 3000mg/day, can cause liver damage in high doses',
                'interactions': ['Warfarin', 'Alcohol']
            },
            'ibuprofen': {
                'uses': 'Pain relief, fever reduction, inflammation',
                'brands': ['Advil', 'Motrin'],
                'warnings': 'Take with food, avoid if kidney problems, can increase bleeding risk',
                'interactions': ['Aspirin', 'Blood thinners', 'ACE inhibitors', 'Diuretics']
            },
            'aspirin': {
                'uses': 'Pain relief, fever reduction, blood thinner',
                'brands': ['Bayer', 'Bufferin'],
                'warnings': 'Can cause stomach bleeding, avoid in children (Reye\'s syndrome)',
                'interactions': ['Warfarin', 'Ibuprofen', 'Blood thinners']
            },
            'omeprazole': {
                'uses': 'Acid reflux, heartburn, stomach ulcers',
                'brands': ['Prilosec', 'Losec'],
                'warnings': 'Long-term use may affect nutrient absorption',
                'interactions': ['Clopidogrel', 'Some antifungals']
            },
            'metformin': {
                'uses': 'Type 2 diabetes management',
                'brands': ['Glucophage', 'Fortamet'],
                'warnings': 'May cause GI upset, avoid in kidney disease',
                'interactions': ['Alcohol', 'Contrast dye for imaging']
            },
            'lisinopril': {
                'uses': 'High blood pressure, heart failure',
                'brands': ['Zestril', 'Prinivil'],
                'warnings': 'May cause cough, avoid in pregnancy',
                'interactions': ['Potassium supplements', 'NSAIDs', 'Other blood pressure medications']
            },
            'atorvastatin': {
                'uses': 'High cholesterol',
                'brands': ['Lipitor'],
                'warnings': 'May cause muscle pain, monitor liver function',
                'interactions': ['Grapefruit', 'Some antibiotics', 'Other cholesterol medications']
            },
            'amoxicillin': {
                'uses': 'Bacterial infections',
                'brands': ['Amoxil', 'Trimox'],
                'warnings': 'Complete full course, check for allergies',
                'interactions': ['Birth control (may reduce effectiveness)', 'Warfarin']
            }
        }
    },
    'drug_interactions': {
        ('aspirin', 'warfarin'): {'risk': 'High', 'effect': 'Increased bleeding risk'},
        ('aspirin', 'ibuprofen'): {'risk': 'Medium', 'effect': 'Reduced aspirin effectiveness, increased GI bleeding'},
        ('ibuprofen', 'warfarin'): {'risk': 'High', 'effect': 'Increased bleeding risk'},
        ('ibuprofen', 'lisinopril'): {'risk': 'Medium', 'effect': 'Reduced blood pressure control'},
        ('acetaminophen', 'warfarin'): {'risk': 'Medium', 'effect': 'May increase bleeding risk'},
        ('acetaminophen', 'alcohol'): {'risk': 'High', 'effect': 'Liver damage risk'},
        ('metformin', 'alcohol'): {'risk': 'High', 'effect': 'Increased lactic acidosis risk'},
        ('lisinopril', 'potassium'): {'risk': 'Medium', 'effect': 'High potassium levels'},
        ('atorvastatin', 'grapefruit'): {'risk': 'Medium', 'effect': 'Increased medication levels, side effects'},
        ('amoxicillin', 'warfarin'): {'risk': 'Medium', 'effect': 'May increase bleeding risk'},
    },
    'food_interactions': {
        'grapefruit': {
            'medications': ['atorvastatin', 'simvastatin', 'lovastatin', 'nifedipine', 'felodipine', 'buspirone', 'sertraline'],
            'effect': 'Blocks enzymes that break down these drugs, causing dangerously high levels in the blood.',
            'severity': 'High'
        },
        'leafy greens': {
            'medications': ['warfarin', 'coumadin'],
            'effect': 'Vitamin K in greens can neutralize blood thinners, increasing risk of blood clots.',
            'severity': 'High'
        },
        'dairy': {
            'medications': ['ciprofloxacin', 'tetracycline', 'doxycycline'],
            'effect': 'Calcium binds to antibiotics, preventing them from being absorbed properly.',
            'severity': 'Medium'
        },
        'alcohol': {
            'medications': ['acetaminophen', 'metformin', 'sertraline', 'alprazolam', 'metronidazole'],
            'effect': 'Increased risk of liver damage, lactic acidosis, or extreme drowsiness.',
            'severity': 'High'
        },
        'caffeine': {
            'medications': ['ephedrine', 'theophylline', 'albuterol'],
            'effect': 'Increased heart rate, jitters, and potential for cardiac stress.',
            'severity': 'Medium'
        },
        'bananas': {
            'medications': ['lisinopril', 'captopril', 'enalapril', 'spironolactone'],
            'effect': 'Potassium-rich foods with ACE inhibitors can lead to hyperkalemia (dangerously high potassium).',
            'severity': 'Medium'
        },
        'licorice': {
            'medications': ['digoxin', 'furosemide', 'hydrochlorothiazide'],
            'effect': 'Can lower potassium levels and interfere with blood pressure/heart medications.',
            'severity': 'Medium'
        }
    }
}


def get_medical_info(query, scraper=None):
    """
    Get medical information from web scraping and local database
    Returns formatted response
    """
    query_lower = query.lower().strip()
    results = []
    
    # First, try local database for quick response
    local_result = search_local_database(query_lower)
    if local_result:
        results.append(local_result)
    
    # Then try web scraping for additional information
    if scraper:
        try:
            web_results = scraper.search_health_topic(query)
            for web_result in web_results[:2]:  # Limit web results
                results.append({
                    'source': web_result['source'],
                    'content': web_result['content']
                })
        except Exception as e:
            print(f"Web scraping error: {e}")
    
    return format_medical_response(query, results)


def search_local_database(query):
    """Search the local medical database"""
    query_lower = query.lower()
    
    # Check symptoms
    for symptom, data in MEDICAL_DATABASE['symptoms'].items():
        if symptom in query_lower:
            return {
                'source': 'Medical Database',
                'type': 'symptom',
                'name': symptom.title(),
                'data': data
            }
    
    # Check conditions
    for condition, data in MEDICAL_DATABASE['conditions'].items():
        if condition in query_lower:
            return {
                'source': 'Medical Database',
                'type': 'condition',
                'name': condition.title(),
                'data': data
            }
    
    # Check medications
    for med_name, med_data in MEDICAL_DATABASE['medications']['common_drugs'].items():
        if med_name in query_lower:
            return {
                'source': 'Medical Database',
                'type': 'medication',
                'name': med_name.title(),
                'data': med_data
            }
    
    return None


def format_medical_response(query, results):
    """Format medical information into a readable response"""
    if not results:
        return f"""**Health Information for: {query}**

I couldn't find specific information about "{query}" in my database.

**General Health Tips:**
- Maintain a balanced diet with plenty of fruits and vegetables
- Exercise regularly (at least 150 minutes of moderate activity per week)
- Stay hydrated (8 glasses of water daily)
- Get adequate sleep (7-9 hours for adults)
- Manage stress through relaxation techniques
- Schedule regular health checkups

**When to See a Doctor:**
- Symptoms persist or worsen
- You're concerned about your health
- Any emergency symptoms (severe pain, difficulty breathing, chest pain)

⚠️ **Important**: For accurate medical advice, always consult with healthcare professionals."""

    response_parts = [f"**Health Information for: {query}**\n"]
    
    for result in results:
        if result.get('type') == 'symptom':
            data = result['data']
            response_parts.append(f"\n📋 **{result['name']}**\n")
            response_parts.append(f"\n**Possible Causes:**")
            for cause in data.get('causes', []):
                response_parts.append(f"\n• {cause}")
            response_parts.append(f"\n\n**Treatment Options:**")
            for treatment in data.get('treatment', []):
                response_parts.append(f"\n• {treatment}")
            response_parts.append(f"\n\n**⚠️ Seek Medical Help If:**")
            for warning in data.get('seek_help', []):
                response_parts.append(f"\n• {warning}")
        
        elif result.get('type') == 'condition':
            data = result['data']
            response_parts.append(f"\n📋 **{result['name']}**\n")
            response_parts.append(f"\n{data.get('description', '')}")
            if 'types' in data:
                response_parts.append(f"\n\n**Types:**")
                for t in data['types']:
                    response_parts.append(f"\n• {t}")
            if 'symptoms' in data:
                response_parts.append(f"\n\n**Symptoms:**")
                for s in data['symptoms']:
                    response_parts.append(f"\n• {s}")
            if 'management' in data:
                response_parts.append(f"\n\n**Management:**")
                for m in data['management']:
                    response_parts.append(f"\n• {m}")
            if 'crisis' in data:
                response_parts.append(f"\n\n🆘 **Crisis Info:** {data['crisis']}")
        
        elif result.get('type') == 'medication':
            data = result['data']
            response_parts.append(f"\n💊 **{result['name']}**\n")
            response_parts.append(f"\n**Uses:** {data.get('uses', 'N/A')}")
            response_parts.append(f"\n**Brand Names:** {', '.join(data.get('brands', []))}")
            response_parts.append(f"\n\n**⚠️ Warnings:** {data.get('warnings', 'N/A')}")
            if 'interactions' in data:
                response_parts.append(f"\n\n**Known Interactions:**")
                for interaction in data['interactions']:
                    response_parts.append(f"\n• {interaction}")
        
        elif result.get('source') and result.get('content'):
            response_parts.append(f"\n\n📖 **From {result['source']}:**\n")
            content = result['content'][:1000]  # Limit content length
            response_parts.append(f"\n{content}...")
    
    response_parts.append("\n\n---")
    response_parts.append("\n⚠️ **Disclaimer**: This information is for educational purposes only. Always consult with qualified healthcare professionals for medical advice, diagnosis, or treatment.")
    
    return ''.join(response_parts)


def check_drug_interactions(medications):
    """
    Check for drug interactions between multiple medications using Hybrid Approach (Local + Web)
    Returns formatted interaction report
    """
    meds_lower = [med.lower().strip() for med in medications if med.strip()]
    
    if len(meds_lower) < 2:
        return "Please provide at least 2 medications to check for interactions.", "N/A"
    
    found_interactions = []
    checked_pairs = set()
    unknown_meds = []
    
    # Check each pair of medications
    for i, med1 in enumerate(meds_lower):
        for j, med2 in enumerate(meds_lower):
            if i >= j:
                continue
            
            pair = tuple(sorted([med1, med2]))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            
            # 1. Check in local database (Verified)
            match_found = False
            for (drug1, drug2), interaction in MEDICAL_DATABASE['drug_interactions'].items():
                if (drug1 in med1 or med1 in drug1) and (drug2 in med2 or med2 in drug2):
                    found_interactions.append({
                        'drugs': f"{med1.title()} + {med2.title()}",
                        'risk': interaction['risk'],
                        'effect': interaction['effect'],
                        'source': 'Verified Database'
                    })
                    match_found = True
                    break
                elif (drug2 in med1 or med1 in drug2) and (drug1 in med2 or med2 in drug1):
                    found_interactions.append({
                        'drugs': f"{med1.title()} + {med2.title()}",
                        'risk': interaction['risk'],
                        'effect': interaction['effect'],
                        'source': 'Verified Database'
                    })
                    match_found = True
                    break
            
            # 2. If not found locally, try Dynamic Web Analysis
            if not match_found:
                # Try to detect if they are same class (e.g. both NSAIDs)
                try:
                    # Quick Wiki check for class keywords
                    def get_class_keywords(drug):
                        wiki = scraper.search_wikipedia(f"{drug} medication class")
                        if not wiki: return ""
                        text = wiki['content'].lower()
                        classes = ['nsaid', 'blood thinner', 'antibiotic', 'opioid', 'beta blocker', 'statin']
                        found = [c for c in classes if c in text[:500]]
                        return found[0] if found else ""

                    class1 = get_class_keywords(med1)
                    class2 = get_class_keywords(med2)

                    if class1 and class2 and class1 == class2:
                         found_interactions.append({
                            'drugs': f"{med1.title()} + {med2.title()}",
                            'risk': 'Medium-High',
                            'effect': f"Potential duplication - Both appear to be {class1.upper()}s. Risk of overdose/side effects.",
                            'source': 'AI Web Analysis'
                        })
                except Exception as e:
                    print(f"Web interaction check error: {e}")

    # Format response
    if found_interactions:
        risks = [i['risk'] for i in found_interactions]
        overall_risk = "High" if 'High' in risks else ("Medium" if 'Medium' in risks or 'Medium-High' in risks else "Low")
        
        response = f"""**💊 Drug Interaction Report**

**Medications Checked:** {', '.join([m.title() for m in meds_lower])}

**⚠️ Interactions Found:**
"""
        for interaction in found_interactions:
            risk_emoji = "🔴" if 'High' in interaction['risk'] else "🟡"
            response += f"\n{risk_emoji} **{interaction['drugs']}**"
            response += f"\n   Risk Level: {interaction['risk']}"
            response += f"\n   Effect: {interaction['effect']}"
            response += f"\n   Source: {interaction.get('source', 'Database')}\n"
        
        response += """
**Recommendations:**
• Consult your pharmacist or doctor about these interactions
• Do not stop taking medications without medical advice
"""
        return response, overall_risk
    
    else:
        return f"""**💊 Drug Interaction Report**

**Medications Checked:** {', '.join([m.title() for m in meds_lower])}

**✅ No Major Interactions Detected**

No verified interactions found in our database or via class-analysis. 
However, new or rare drugs may not be fully indexed.

**General Safety Tips:**
• Take medications as prescribed
• Report any unusual symptoms to your doctor immediately
""", "Low"


def analyze_symptoms(symptoms_list, duration=None, severity=None, scraper=None):
    """
    Analyze symptoms using Hybrid Approach (Local DB + Web NLP)
    """
    if not symptoms_list:
        return "Please provide symptoms to analyze.", "N/A", []
    
    symptoms_lower = [s.lower().strip() for s in symptoms_list]
    
    # 1. Check Local Emergency Database
    emergency_symptoms = {
        'chest pain', 'difficulty breathing', 'shortness of breath',
        'severe headache', 'sudden numbness', 'loss of consciousness',
        'severe abdominal pain', 'coughing blood', 'vomiting blood',
        'sudden vision loss', 'high fever with rash'
    }
    is_emergency = any(emerg in ' '.join(symptoms_lower) for emerg in emergency_symptoms)
    
    # 2. Local Database Lookup & Accumulate missing symptoms
    symptom_info = []
    recommendations = set()
    missing_symptoms = []

    for symptom in symptoms_lower:
        found = False
        for db_symptom, data in MEDICAL_DATABASE['symptoms'].items():
            if db_symptom in symptom or symptom in db_symptom:
                symptom_info.append({
                    'name': db_symptom.title(),
                    'causes': data.get('causes', []),
                    'treatment': data.get('treatment', []),
                    'source': 'Verified Database'
                })
                for rec in data.get('treatment', [])[:2]:
                    recommendations.add(rec)
                found = True
        
        if not found:
            missing_symptoms.append(symptom)

    # 3. Dynamic Web Analysis for Missing Symptoms
    if missing_symptoms and scraper:
        for ms in missing_symptoms:
            try:
                # Scrape generic "symptom" page
                wiki_data = scraper.search_wikipedia(f"{ms} symptom medical")
                if wiki_data:
                    content = wiki_data['content'][:1000].lower()
                    
                    # NLP Keyword Scanning for Risk
                    danger_words = ['emergency', 'fatal', 'urgent', 'call 911', 'severe', 'hospital', 'immediate']
                    found_dangers = [w for w in danger_words if w in content]
                    
                    if found_dangers:
                        is_emergency = True
                        recommendations.add(f"Web warning for {ms}: Mention of '{found_dangers[0]}' detected.")

                    # Simple Extraction for Recommendations/Causes
                    # (Very basic regex-like logic)
                    wiki_causes = []
                    if "caused by" in content:
                        parts = content.split("caused by")[1].split(".")
                        wiki_causes.append(parts[0].strip())
                    
                    symptom_info.append({
                        'name': ms.title(),
                        'causes': wiki_causes if wiki_causes else ["See details in description"],
                        'treatment': ["Consult doctor for diagnosis"],
                        'description': content[:300] + "...",
                        'source': 'Web Analysis (Wikipedia)'
                    })
            except Exception as e:
                print(f"Dynamic symptom check error: {e}")

    # Determine risk level
    if is_emergency:
        risk_level = "High"
    elif severity and int(severity) >= 7:
        risk_level = "Medium-High"
    elif severity and int(severity) >= 5:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    # Build response
    if is_emergency:
        response = f"""🚨 **EMERGENCY ALERT**

**Symptoms reported:** {', '.join([s.title() for s in symptoms_list])}

**⚠️ These symptoms may indicate a serious condition.**

**PLEASE:**
• Call emergency services (911) 
• Go to the nearest emergency room immediately
"""
    else:
        response = f"""**📋 Symptom Analysis**

**Reported Symptoms:** {', '.join([s.title() for s in symptoms_list])}
**Risk Assessment:** {risk_level}
"""
        if symptom_info:
            response += "\n---\n**Analysis Details:**\n"
            for info in symptom_info:
                response += f"\n**{info['name']}** ({info.get('source', 'Unknown')})"
                if info.get('description'):
                     response += f"\nNote: {info['description']}"
                if info.get('causes'):
                    response += f"\nPossible causes: {', '.join(info['causes'][:3])}"
        
        if recommendations:
            response += "\n\n**💡 Recommendations:**"
            for rec in list(recommendations)[:5]:
                response += f"\n• {rec}"

        response += "\n\n⚠️ **Disclaimer**: AI Analysis. Consult local doctor."

    return response, risk_level, list(recommendations)


# Initialize scraper instance
scraper = MedicalWebScraper()



def check_food_interactions(medications, scraper=None):
    """
    Check for food and drink interactions for a list of medications
    """
    meds_lower = [med.lower().strip() for med in medications if med.strip()]
    findings = []
    
    # 1. Local Database Match
    for food, data in MEDICAL_DATABASE['food_interactions'].items():
        matched_meds = []
        for med in meds_lower:
            if any(m in med or med in m for m in data['medications']):
                matched_meds.append(med.title())
        
        if matched_meds:
            findings.append({
                'food': food.title(),
                'medications': matched_meds,
                'effect': data['effect'],
                'severity': data['severity'],
                'source': 'Verified Medical Database'
            })
    
    # 2. Dynamic Web Search for unknown meds
    if scraper:
        # Avoid duplicate searches, only check if we don't have enough info
        for med in meds_lower[:2]: # Limit to first 2 for speed
            try:
                # Search for specific interactions
                wiki = scraper.search_wikipedia(f"{med} food and drink interactions")
                if wiki and ("avoid" in wiki['content'].lower() or "interaction" in wiki['content'].lower()):
                    content = wiki['content'].lower()
                    
                    # Look for common trigger foods in the text if not already found
                    common_triggers = ['grapefruit', 'alcohol', 'caffeine', 'dairy', 'milk', 'cheese', 'green leafy', 'kale', 'spinach', 'tyramine', 'banana']
                    for trigger in common_triggers:
                        if trigger in content and trigger not in [f['food'].lower() for f in findings]:
                            findings.append({
                                'food': trigger.title(),
                                'medications': [med.title()],
                                'effect': f"Mentioned in clinical resources: Potential interaction with {trigger}. See source.",
                                'severity': 'Review Required',
                                'source': wiki['source']
                            })
            except Exception as e:
                print(f"Web food check error: {e}")
                
    return findings
