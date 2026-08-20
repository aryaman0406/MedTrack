"""
Gemini AI Service for MedTrack
Handles all AI-powered medical features using Google's Gemini API.
No fallback — all content is fetched live from Gemini.
"""

import os
import json
import re
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

_CANDIDATE_MODELS = [
    "gemini-flash-latest",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-pro-latest",
    "gemini-flash-lite-latest",
]

def _get_client():
    """Return a configured Gemini client. Raises if key is missing."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is not set or still has default placeholder. "
            "Add it to your .env file: GEMINI_API_KEY=your_key_here\n"
            "Get a free key at https://aistudio.google.com/"
        )
    return genai.Client(api_key=api_key)

_client = None

def _ensure_client():
    global _client
    if _client is None:
        _client = _get_client()
    return _client

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MEDICAL_DISCLAIMER = (
    "\n\n---\n⚠️ **Disclaimer**: This information is for educational purposes only. "
    "Always consult with qualified healthcare professionals for medical advice, "
    "diagnosis, or treatment."
)


def _ask_gemini(system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to Gemini with automatic failover across models."""
    client = _ensure_client()
    full_prompt = f"{system_prompt}\n\nUser query: {user_prompt}"
    
    last_error = None
    for model_name in _CANDIDATE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=900,
                ),
            )
            if response and response.text:
                return response.text
        except Exception as e:
            print(f"Model {model_name} error: {e}. Trying next candidate model...")
            last_error = e
            continue
            
    raise RuntimeError(f"All Gemini models unavailable. Last error: {last_error}")


def _ask_gemini_json(system_prompt: str, user_prompt: str) -> dict | list | None:
    """Send a prompt to Gemini requesting JSON output with automatic model failover."""
    client = _ensure_client()
    full_prompt = f"{system_prompt}\n\nUser query: {user_prompt}"
    
    last_error = None
    for model_name in _CANDIDATE_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                    max_output_tokens=1200,
                ),
            )
            text = response.text.strip() if response and response.text else ""
            if text:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
                    if match:
                        return json.loads(match.group(1).strip())
        except Exception as e:
            print(f"JSON Model {model_name} error: {e}. Trying next candidate model...")
            last_error = e
            continue
            
    if last_error:
        print(f"All Gemini JSON models returned errors. Last error: {last_error}")
    return None


# ===================================================================
# 1. CHATBOT
# ===================================================================

_CHAT_SYSTEM_PROMPT = """You are MedTrack's AI Medical Assistant called "Medibot".

ROLE:
- Provide helpful, accurate, and empathetic medical information.
- You can answer questions about symptoms, diseases, medications, treatments, wellness, nutrition, mental health, and general health topics.
- If the user greets you (hi, hello, hey, etc.), respond warmly and tell them what you can help with.
- If the user asks a non-medical question, politely redirect them to medical topics.

USER CONTEXT (if available):
- The user's current medications are listed below. Use this context when relevant.

FORMATTING RULES:
- Use markdown formatting with **bold** headers and bullet points.
- Structure your response with clear sections like: Overview, Causes, Symptoms, Treatment, When to See a Doctor.
- Keep responses concise but comprehensive (200-400 words).
- Always end with a medical disclaimer.
- Use medical emojis sparingly (🩺 💊 ⚠️ 🏥) for visual appeal.

SAFETY:
- Never diagnose — only provide general information.
- For emergencies, always advise calling emergency services immediately.
- Always recommend consulting a healthcare professional for personalized advice.
"""


def chat(question: str, user_medications: list[str] = None) -> str:
    """General medical chatbot response."""
    meds_context = ""
    if user_medications:
        meds_context = f"\nUser's current medications: {', '.join(user_medications)}\n"

    prompt = f"{meds_context}Question: {question}"
    result = _ask_gemini(_CHAT_SYSTEM_PROMPT, prompt)
    return result


# ===================================================================
# 2. SYMPTOM CHECKER
# ===================================================================

_SYMPTOM_SYSTEM_PROMPT = """You are a medical symptom analysis AI for MedTrack.

TASK: Analyze the user's symptoms and provide a structured medical assessment.

You MUST respond in VALID JSON format with this exact structure:
{
    "diagnosis": "A detailed markdown-formatted analysis including:\\n- Possible conditions ranked by likelihood\\n- Brief explanation of each condition\\n- How the symptoms relate to each condition",
    "risk_level": "High OR Medium OR Low",
    "recommendations": ["recommendation 1", "recommendation 2", "..."]
}

RISK LEVEL GUIDELINES:
- "High": Symptoms suggest potentially serious/life-threatening conditions (chest pain, difficulty breathing, severe bleeding, stroke symptoms, high fever with stiff neck, etc.)
- "Medium": Symptoms that need medical attention but are not immediately life-threatening
- "Low": Common, self-limiting symptoms that can often be managed at home

IMPORTANT:
- Consider duration and severity in your assessment.
- If user info (age, conditions) is provided, factor it into the analysis.
- Always include "Consult a healthcare professional" in recommendations.
- For High risk, always include "Seek immediate medical attention" or "Call emergency services".
- Provide 3-6 actionable recommendations.
"""


def analyze_symptoms(symptoms: list[str], duration: str = None,
                     severity: str = None, user_info: dict = None) -> tuple:
    """
    Analyze symptoms and return (diagnosis_text, risk_level, recommendations).
    """
    parts = [f"Symptoms: {', '.join(symptoms)}"]
    if duration:
        parts.append(f"Duration: {duration}")
    if severity:
        parts.append(f"Severity: {severity}/10")
    if user_info:
        if user_info.get('age'):
            parts.append(f"Patient age: {user_info['age']}")
        if user_info.get('conditions'):
            parts.append(f"Existing conditions: {user_info['conditions']}")

    prompt = "\n".join(parts)
    result = _ask_gemini_json(_SYMPTOM_SYSTEM_PROMPT, prompt)

    if result and isinstance(result, dict):
        diagnosis = result.get("diagnosis", "Unable to analyze symptoms. Please consult a doctor.")
        risk_level = result.get("risk_level", "Medium")
        recommendations = result.get("recommendations", ["Please consult a healthcare professional."])
        return diagnosis, risk_level, recommendations

    return ("⚠️ Unable to analyze symptoms at this time. Please consult a healthcare professional.",
            "Unknown",
            ["Please consult a healthcare professional for proper evaluation."])


# ===================================================================
# 3. DRUG INTERACTIONS
# ===================================================================

_DRUG_INTERACTION_SYSTEM_PROMPT = """You are a pharmacology AI for MedTrack that checks drug interactions.

TASK: Analyze potential interactions between the given medications.

You MUST respond in VALID JSON format with this exact structure:
{
    "report": "A detailed markdown-formatted drug interaction report including:\\n- Each interaction found between medication pairs\\n- Mechanism of interaction\\n- Clinical significance\\n- What to watch for",
    "risk_level": "High OR Medium OR Low",
    "interactions_found": true or false
}

FORMAT THE REPORT FIELD AS MARKDOWN with:
- **💊 Drug Interaction Report** as the title
- **Medications Checked:** list all medications
- For each interaction found:
  - 🔴 (High risk) or 🟡 (Medium risk) or 🟢 (Low risk) emoji prefix
  - Drug pair in bold
  - Risk Level, Mechanism, Clinical Effect, and Recommendation
- **General Safety Tips** section at the end

RISK LEVEL GUIDELINES:
- "High": Dangerous combinations that can cause serious harm (e.g., multiple blood thinners, serotonin syndrome risk, severe liver/kidney toxicity)
- "Medium": Interactions that need monitoring or dose adjustment
- "Low": Minor interactions or no significant interactions found

IMPORTANT:
- Be thorough — check ALL possible pairs.
- Include pharmacodynamic AND pharmacokinetic interactions.
- If no significant interactions are found, say so clearly but still provide general safety advice.
"""


def check_drug_interactions(medications: list[str]) -> tuple:
    """
    Check drug interactions. Returns (report_text, risk_level).
    """
    prompt = f"Check interactions between these medications: {', '.join(medications)}"
    result = _ask_gemini_json(_DRUG_INTERACTION_SYSTEM_PROMPT, prompt)

    if result and isinstance(result, dict):
        report = result.get("report", "Unable to check drug interactions.")
        risk_level = result.get("risk_level", "Unknown")
        return report, risk_level

    return ("⚠️ Unable to check drug interactions at this time. Please consult a pharmacist.", "Unknown")


# ===================================================================
# 4. FOOD-DRUG INTERACTIONS
# ===================================================================

_FOOD_INTERACTION_SYSTEM_PROMPT = """You are a pharmacology and nutrition AI for MedTrack.

TASK: Identify food and beverage interactions for the given medications.

You MUST respond in VALID JSON format — an ARRAY of interaction objects:
[
    {
        "food": "Name of food/beverage (e.g., Grapefruit)",
        "medications": ["medication1"],
        "effect": "Detailed description of the interaction and why it occurs",
        "severity": "High OR Medium OR Low",
        "source": "Gemini AI Medical Analysis"
    }
]

If no food interactions are found, return an empty array: []

GUIDELINES:
- Check for common food-drug interactions: grapefruit, alcohol, dairy, caffeine, leafy greens (vitamin K), tyramine-rich foods, high-potassium foods, high-fiber foods, etc.
- Be specific about WHY the interaction occurs (enzyme inhibition, absorption interference, etc.).
- Include practical dietary advice in the effect description.
- Only include real, clinically documented food-drug interactions.
"""


def check_food_interactions(medications: list[str]) -> list:
    """
    Check food-drug interactions. Returns list of interaction dicts.
    """
    prompt = f"Check food and beverage interactions for these medications: {', '.join(medications)}"
    result = _ask_gemini_json(_FOOD_INTERACTION_SYSTEM_PROMPT, prompt)

    if result and isinstance(result, list):
        return result

    return []


# ===================================================================
# 5. HEALTH NEWS / TIPS
# ===================================================================

_HEALTH_TIPS_SYSTEM_PROMPT = """You are a health and wellness content AI for MedTrack.

TASK: Generate fresh, evidence-based health tips and wellness articles.

You MUST respond in VALID JSON format with this structure:
{
    "news_items": [
        {
            "title": "Article title",
            "summary": "A 2-3 sentence summary of the health topic with actionable advice (150-250 chars)",
            "category": "Category name (e.g., Nutrition, Exercise, Mental Health, Sleep, Prevention, Heart Health)",
            "icon": "A single relevant emoji"
        }
    ],
    "health_tips": [
        {
            "title": "Short tip title",
            "tip": "One-sentence actionable health tip",
            "icon": "A single relevant emoji"
        }
    ]
}

GUIDELINES:
- Generate exactly 4 news_items and 6 health_tips.
- Make content diverse: cover nutrition, exercise, mental health, sleep, disease prevention, seasonal health.
- If user conditions are provided, include 1-2 personalized tips related to their conditions.
- Keep tips practical and evidence-based.
- Make titles engaging and informative.
- Vary the content — don't repeat generic advice.
"""


def generate_health_tips(user_conditions: list[str] = None) -> tuple:
    """
    Generate health tips and news. Returns (news_items, health_tips).
    """
    context = ""
    if user_conditions:
        context = f"\nUser's health conditions: {', '.join(user_conditions)}\n"

    prompt = f"{context}Generate fresh, diverse health tips and wellness content for today."
    result = _ask_gemini_json(_HEALTH_TIPS_SYSTEM_PROMPT, prompt)

    if result and isinstance(result, dict):
        news_items = result.get("news_items", [])
        health_tips = result.get("health_tips", [])
        return news_items, health_tips

    # Minimal default if Gemini fails
    return [], []


# ===================================================================
# 6. SIDE EFFECTS ANALYSIS
# ===================================================================

_SIDE_EFFECTS_SYSTEM_PROMPT = """You are a clinical pharmacology AI for MedTrack.

TASK: Analyze the user's logged medication side effects and provide clinical insights.

Provide a markdown-formatted analysis that includes:

1. **Pattern Assessment**: For each medication with reported side effects:
   - Whether the side effects are commonly expected for that medication
   - Whether the frequency/severity of occurrences is concerning

2. **Clinical Insights**:
   - Any side effects that could indicate a serious adverse reaction
   - Potential drug-related causes of the reported effects

3. **Recommendations**:
   - Which side effects to discuss with their doctor
   - Any simple management tips (e.g., "take with food" for GI side effects)
   - When to seek immediate medical attention

Keep the analysis concise, practical, and in plain language.
Use emojis sparingly for visual clarity (✅ ⚠️ 🔴 💊).
"""


def analyze_side_effects(effects_data: list[dict]) -> str:
    """
    Analyze logged side effects and return AI insights.
    effects_data: list of dicts with keys: medication_name, side_effect, severity, occurrences, avg_severity
    """
    if not effects_data:
        return ""

    lines = ["Logged side effects to analyze:"]
    for e in effects_data:
        sev_label = {1: "Mild", 2: "Moderate", 3: "Severe"}.get(int(e.get("avg_severity", e.get("severity", 1))), "Unknown")
        lines.append(
            f"- {e['medication_name']}: {e['side_effect']} "
            f"(occurrences: {e.get('occurrences', 1)}, severity: {sev_label})"
        )

    prompt = "\n".join(lines)
    return _ask_gemini(_SIDE_EFFECTS_SYSTEM_PROMPT, prompt)
