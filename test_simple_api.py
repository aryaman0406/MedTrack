#!/usr/bin/env python3
"""
Simple API test without vectorstore
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_groq_simple():
    """Test GROQ API with simple query"""
    print("Testing GROQ API...")
    try:
        from langchain_groq import ChatGroq
        
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key or len(groq_key) < 50:
            print("❌ GROQ key invalid")
            return False
            
        llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.0,
            groq_api_key=groq_key,
        )
        
        response = llm.invoke("What is diabetes? Give a brief answer.")
        print(f"✅ GROQ Response: {response.content[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ GROQ failed: {e}")
        return False

def test_xai_simple():
    """Test XAI API with simple query"""
    print("\nTesting XAI API...")
    try:
        from langchain_openai import ChatOpenAI
        
        xai_key = os.getenv("XAI_API_KEY")
        if not xai_key:
            print("❌ XAI key missing")
            return False
            
        llm = ChatOpenAI(
            model="grok-beta",
            temperature=0.0,
            openai_api_key=xai_key,
            openai_api_base="https://api.x.ai/v1",
        )
        
        response = llm.invoke("What is diabetes? Give a brief answer.")
        print(f"✅ XAI Response: {response.content[:200]}...")
        return True
        
    except Exception as e:
        print(f"❌ XAI failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Simple API Test")
    print("=" * 30)
    
    groq_works = test_groq_simple()
    xai_works = test_xai_simple()
    
    print("\n" + "=" * 30)
    if groq_works or xai_works:
        print("✅ At least one API is working!")
        if groq_works:
            print("- GROQ: Working ✅")
        if xai_works:
            print("- XAI: Working ✅")
    else:
        print("❌ Both APIs failed")
