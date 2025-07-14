#!/usr/bin/env python3
"""
Test script to verify medibot integration
"""
import os
import sys
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_imports():
    """Test if all medibot imports work"""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import PromptTemplate
        from langchain_groq import ChatGroq
        print("✅ All medibot imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test if required environment variables are set"""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        print("✅ GROQ_API_KEY found")
        return True
    else:
        print("❌ GROQ_API_KEY not found in environment")
        return False

def test_vectorstore_path():
    """Test if vectorstore path exists"""
    db_path = "medical-chatbot/vectorstore/db_faiss"
    if os.path.exists(db_path):
        print("✅ Vectorstore path exists")
        return True
    else:
        print(f"❌ Vectorstore path not found: {db_path}")
        return False

if __name__ == "__main__":
    print("Testing medibot integration...")
    print("-" * 40)
    
    tests = [
        test_imports(),
        test_environment(),
        test_vectorstore_path()
    ]
    
    if all(tests):
        print("\n🎉 All tests passed! Medibot integration is ready.")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
