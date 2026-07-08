import os
import json
import urllib.request
import urllib.parse
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

def fetch_hf_rows(dataset, config, split, total_limit=None):
    rows = []
    offset = 0
    length = 100
    dataset_encoded = urllib.parse.quote(dataset, safe='')
    
    while True:
        if total_limit and offset >= total_limit:
            break
            
        url = f"https://datasets-server.huggingface.co/rows?dataset={dataset_encoded}&config={config}&split={split}&offset={offset}&length={length}"
        print(f"Fetching {dataset} offset {offset}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                data = json.loads(response.read().decode())
                
            batch = data.get("rows", [])
            if not batch:
                break
                
            rows.extend(batch)
            offset += length
            time.sleep(0.5)  # Rate limiting
        except Exception as e:
            print(f"Error fetching {dataset} at offset {offset}: {e}")
            break
            
    return rows

def build_drugs():
    print("Building drugs...")
    rows = fetch_hf_rows("darkknight25/medical_medicine_dataset", "default", "train", total_limit=700)
    drugs = {}
    for r in rows:
        row = r["row"]
        name = row.get("name", "").strip().lower()
        if name:
            drugs[name] = {
                "name": row.get("name"),
                "description": row.get("description", ""),
                "uses": row.get("uses", ""),
                "side_effects": row.get("side_effects", "")
            }
            
    with open(os.path.join(DATA_DIR, "drugs.json"), "w") as f:
        json.dump(drugs, f, indent=2)
    print(f"Saved {len(drugs)} drugs.")

def build_symptoms():
    print("Building symptoms...")
    rows = fetch_hf_rows("gretelai/symptom_to_diagnosis", "default", "train", total_limit=1100)
    symptoms = []
    for r in rows:
        row = r["row"]
        text = row.get("output_text", "").strip()
        diagnosis = row.get("input_text", "").strip()
        if text and diagnosis:
            symptoms.append({
                "description": text,
                "diagnosis": diagnosis
            })
            
    with open(os.path.join(DATA_DIR, "symptoms.json"), "w") as f:
        json.dump(symptoms, f, indent=2)
    print(f"Saved {len(symptoms)} symptoms.")

def build_interactions():
    print("Building interactions...")
    rows = fetch_hf_rows("bigbio/ddi_corpus", "ddi_corpus_bigbio_kb", "train", total_limit=800)
    interactions = []
    
    for r in rows:
        row = r["row"]
        entities = {e["id"]: e["text"][0].lower() for e in row.get("entities", []) if e["text"]}
        
        for rel in row.get("relations", []):
            arg1_id = rel.get("arg1_id")
            arg2_id = rel.get("arg2_id")
            rel_type = rel.get("type", "")
            
            if arg1_id in entities and arg2_id in entities:
                interactions.append({
                    "drug_a": entities[arg1_id],
                    "drug_b": entities[arg2_id],
                    "interaction": rel_type,
                    "source": "DDI Corpus"
                })
                
    # Add existing hardcoded critical interactions
    hardcoded = {
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
    }
    
    for (d1, d2), data in hardcoded.items():
        interactions.append({
            "drug_a": d1,
            "drug_b": d2,
            "interaction": f"Risk: {data['risk']}, Effect: {data['effect']}",
            "source": "Critical Database"
        })
        
    with open(os.path.join(DATA_DIR, "interactions.json"), "w") as f:
        json.dump(interactions, f, indent=2)
    print(f"Saved {len(interactions)} interactions.")

def build_food_interactions():
    print("Building food interactions...")
    hardcoded_food = {
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
    with open(os.path.join(DATA_DIR, "food_interactions.json"), "w") as f:
        json.dump(hardcoded_food, f, indent=2)
    print(f"Saved {len(hardcoded_food)} food interactions.")

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    build_drugs()
    build_symptoms()
    build_interactions()
    build_food_interactions()
    print("Knowledge base built successfully.")
