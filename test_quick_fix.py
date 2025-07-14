#!/usr/bin/env python3
"""
Quick test of the AI fixes
"""
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

def test_llm_only():
    """Test LLM without vectorstore"""
    print("=== Testing LLM Only ===")
    try:
        from app import get_llm_client
        
        llm_client, llm_source = get_llm_client()
        if llm_client is None:
            print("❌ No LLM client available")
            return False
        
        print(f"✅ LLM client created: {llm_source}")
        
        # Test a simple medical question
        question = "What are the common symptoms of diabetes?"
        medical_prompt = f"""You are a medical AI assistant. Please provide accurate medical information about: {question}

Important: This is for educational purposes only.

Question: {question}

Please provide a helpful response:"""
        
        response = llm_client.invoke(medical_prompt)
        result = response.content if hasattr(response, 'content') else str(response)
        
        print(f"✅ Response received ({len(result)} chars)")
        print(f"Preview: {result[:150]}...")
        return True
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Quick AI Fix Test")
    print("=" * 30)
    
    if test_llm_only():
        print("\n✅ AI system should work now!")
        print("Try accessing:")
        print("- http://127.0.0.1:5000/chatbot")
        print("- http://127.0.0.1:5000/symptom-checker")
        print("- http://127.0.0.1:5000/drug-interactions")
    else:
        print("\n❌ Still having issues. Check API keys.")
