#!/usr/bin/env python3
"""
Test Flask app AI routes directly
"""
import os
import sys
from dotenv import load_dotenv

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

def test_app_imports():
    """Test if we can import the Flask app components"""
    print("=== Testing App Imports ===")
    try:
        # Test basic imports
        from flask import Flask
        print("✅ Flask import successful")
        
        from langchain_groq import ChatGroq
        print("✅ LangChain GROQ import successful")
        
        from langchain_openai import ChatOpenAI
        print("✅ LangChain OpenAI import successful")
        
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ HuggingFace embeddings import successful")
        
        from langchain_community.vectorstores import FAISS
        print("✅ FAISS import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_get_llm_client():
    """Test the get_llm_client function from app.py"""
    print("\n=== Testing get_llm_client Function ===")
    try:
        # Import the function from app.py
        from app import get_llm_client
        
        llm_client, llm_source = get_llm_client()
        
        if llm_client is None:
            print(f"❌ No LLM client available. Source: {llm_source}")
            return False
        else:
            print(f"✅ LLM client created successfully. Source: {llm_source}")
            return True, llm_client, llm_source
            
    except Exception as e:
        print(f"❌ get_llm_client failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vectorstore_function():
    """Test the get_vectorstore function from app.py"""
    print("\n=== Testing get_vectorstore Function ===")
    try:
        from app import get_vectorstore
        
        vectorstore = get_vectorstore()
        
        if vectorstore is None:
            print("❌ Vectorstore failed to load")
            return False
        else:
            print("✅ Vectorstore loaded successfully")
            return True, vectorstore
            
    except Exception as e:
        print(f"❌ get_vectorstore failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run Flask app component tests"""
    print("🔍 Flask App Component Test")
    print("=" * 40)
    
    # Test 1: Imports
    if not test_app_imports():
        print("❌ Basic imports failed. Cannot proceed.")
        return
    
    # Test 2: LLM Client
    llm_result = test_get_llm_client()
    if isinstance(llm_result, tuple):
        success, llm_client, llm_source = llm_result
        print(f"LLM Status: {llm_source}")
    else:
        print("❌ LLM client test failed")
    
    # Test 3: Vectorstore
    vs_result = test_vectorstore_function()
    if isinstance(vs_result, tuple):
        success, vectorstore = vs_result
        print("Vectorstore Status: Loaded")
    else:
        print("❌ Vectorstore test failed")
    
    print("\n" + "=" * 40)
    print("🏁 Component Test Complete")

if __name__ == "__main__":
    main()
