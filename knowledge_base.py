import os
import json
import difflib

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

def load_json(filename):
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

# Load data into memory on module import
DRUGS_DB = load_json('drugs.json') or {}
SYMPTOMS_DB = load_json('symptoms.json') or []
INTERACTIONS_DB = load_json('interactions.json') or []
FOOD_INTERACTIONS_DB = load_json('food_interactions.json') or {}

def _fuzzy_match(query, choices, cutoff=0.6):
    matches = difflib.get_close_matches(query.lower(), choices, n=1, cutoff=cutoff)
    return matches[0] if matches else None

def lookup_drug_info(name):
    """Lookup drug information from medical_medicine_dataset"""
    if not DRUGS_DB:
        return None
        
    query = name.lower().strip()
    match = _fuzzy_match(query, list(DRUGS_DB.keys()))
    if match:
        return DRUGS_DB[match]
    return None

def lookup_symptom(text):
    """Lookup symptom from symptom_to_diagnosis dataset"""
    if not SYMPTOMS_DB:
        return None
        
    query = text.lower().strip()
    
    # Synonym handling
    if "loose motion" in query or "loose motions" in query:
        query = query.replace("loose motion", "diarrhea").replace("loose motions", "diarrhea")
        
    # Split query into keywords for better matching against long descriptions
    keywords = [w for w in query.split() if len(w) > 3 and w not in ['medicine', 'treatment', 'causes', 'give', 'what', 'how']]
    if not keywords:
        keywords = [query]
        
    best_match = None
    max_score = 0
    
    for s in SYMPTOMS_DB:
        # s['diagnosis'] contains the patient's long description
        # s['description'] contains the actual diagnosis name
        patient_text = s['diagnosis'].lower()
        
        score = sum(1 for kw in keywords if kw in patient_text)
        if score > max_score:
            max_score = score
            best_match = s
            
    if best_match and max_score > 0:
        return {
            'matched_symptom': best_match['description'],
            'diagnosis': best_match['diagnosis']
        }
        
    return None

def lookup_drug_interaction(drug_a, drug_b):
    """Lookup interaction between two drugs from DDI corpus + Hardcoded DB"""
    if not INTERACTIONS_DB:
        return None
        
    da = drug_a.lower().strip()
    db = drug_b.lower().strip()
    
    # Simple fuzzy match on names to ensure robustness
    all_drugs = set()
    for row in INTERACTIONS_DB:
        all_drugs.add(row['drug_a'])
        all_drugs.add(row['drug_b'])
        
    da_match = _fuzzy_match(da, all_drugs, cutoff=0.7) or da
    db_match = _fuzzy_match(db, all_drugs, cutoff=0.7) or db
    
    for interaction in INTERACTIONS_DB:
        i_a = interaction['drug_a']
        i_b = interaction['drug_b']
        
        if (da_match == i_a and db_match == i_b) or (da_match == i_b and db_match == i_a):
            return {
                'drugs': f"{interaction['drug_a'].title()} + {interaction['drug_b'].title()}",
                'interaction': interaction['interaction'],
                'source': interaction['source']
            }
            
    return None

def lookup_food_interaction(drug):
    """Lookup food interaction for a drug"""
    if not FOOD_INTERACTIONS_DB:
        return None
        
    query = drug.lower().strip()
    
    findings = []
    for food, data in FOOD_INTERACTIONS_DB.items():
        meds = data['medications']
        match = _fuzzy_match(query, meds, cutoff=0.7)
        if match or query in meds:
            findings.append({
                'food': food.title(),
                'medications': [query.title()],
                'effect': data['effect'],
                'severity': data['severity'],
                'source': 'Local Knowledge Base'
            })
            
    return findings if findings else None
