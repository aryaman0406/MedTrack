#!/usr/bin/env python3
"""
Test script to check AI functionality without running the full Flask app
"""
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_api_keys():
    """Test if API keys are properly configured"""
    print("=== API Key Test ===")
    groq_key = os.getenv("GROQ_API_KEY")
    xai_key = os.getenv("XAI_API_KEY")
    
    print(f"GROQ API Key: {'✅ Found' if groq_key else '❌ Missing'} ({len(groq_key) if groq_key else 0} chars)")
    print(f"XAI API Key: {'✅ Found' if xai_key else '❌ Missing'} ({len(xai_key) if xai_key else 0} chars)")
    
    return groq_key, xai_key

def test_vectorstore():
    """Test vectorstore loading"""
    print("\n=== Vectorstore Test ===")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        
        print("✅ Langchain imports successful")
        
        # Test embedding model
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        print("✅ Embedding model created")
        
        # Test vectorstore loading
        DB_FAISS_PATH = "medical-chatbot/vectorstore/db_faiss"
        if os.path.exists(DB_FAISS_PATH):
            print(f"✅ Vectorstore path exists: {DB_FAISS_PATH}")
            vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            print("✅ Vectorstore loaded successfully")
            
            # Test search
            docs = vectorstore.similarity_search("blood pressure", k=2)
            print(f"✅ Search test successful - found {len(docs)} documents")
            return vectorstore
        else:
            print(f"❌ Vectorstore path not found: {DB_FAISS_PATH}")
            return None
            
    except Exception as e:
        print(f"❌ Vectorstore test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_groq_connection(groq_key):
    """Test GROQ API connection"""
    print("\n=== GROQ API Test ===")
    if not groq_key or len(groq_key) < 50:
        print("❌ GROQ API key invalid or too short")
        return None
        
    try:
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0.0,
            groq_api_key=groq_key,
        )
        print("✅ GROQ client created successfully")
        
        # Test a simple query
        response = llm.invoke("What is the capital of France?")
        print(f"✅ GROQ API call successful: {response.content[:50]}...")
        return llm
        
    except Exception as e:
        print(f"❌ GROQ test failed: {e}")
        return None

def test_xai_connection(xai_key):
    """Test XAI API connection"""
    print("\n=== XAI API Test ===")
    if not xai_key:
        print("❌ XAI API key missing")
        return None
        
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="grok-beta",
            temperature=0.0,
            openai_api_key=xai_key,
            openai_api_base="https://api.x.ai/v1",
        )
        print("✅ XAI client created successfully")
        
        # Test a simple query
        response = llm.invoke("What is the capital of France?")
        print(f"✅ XAI API call successful: {response.content[:50]}...")
        return llm
        
    except Exception as e:
        print(f"❌ XAI test failed: {e}")
        return None

def test_qa_chain(vectorstore, llm, llm_name):
    """Test QA chain functionality"""
    print(f"\n=== QA Chain Test ({llm_name}) ===")
    if not vectorstore or not llm:
        print("❌ Prerequisites missing")
        return False
        
    try:
        from langchain.chains import RetrievalQA
        from langchain_core.prompts import PromptTemplate
        
        custom_prompt_template = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        
        print("✅ QA chain created successfully")
        
        # Test a medical query
        response = qa_chain.invoke({'query': 'What are the symptoms of diabetes?'})
        result = response["result"]
        source_docs = response["source_documents"]
        
        print(f"✅ QA chain test successful!")
        print(f"Answer length: {len(result)} characters")
        print(f"Source documents: {len(source_docs)} found")
        print(f"Preview: {result[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ QA chain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔬 AI System Diagnostics")
    print("=" * 50)
    
    # Test 1: API Keys
    groq_key, xai_key = test_api_keys()
    
    # Test 2: Vectorstore
    vectorstore = test_vectorstore()
    
    # Test 3: GROQ Connection
    groq_llm = test_groq_connection(groq_key)
    
    # Test 4: XAI Connection  
    xai_llm = test_xai_connection(xai_key)
    
    # Test 5: QA Chain with GROQ
    if groq_llm and vectorstore:
        test_qa_chain(vectorstore, groq_llm, "GROQ")
    
    # Test 6: QA Chain with XAI
    if xai_llm and vectorstore:
        test_qa_chain(vectorstore, xai_llm, "XAI")
    
    print("\n" + "=" * 50)
    print("🏁 Diagnostics Complete!")
    
    # Summary
    working_systems = []
    if groq_llm and vectorstore:
        working_systems.append("GROQ + Vectorstore")
    if xai_llm and vectorstore:
        working_systems.append("XAI + Vectorstore")
    
    if working_systems:
        print(f"✅ Working systems: {', '.join(working_systems)}")
        print("Your AI features should be working!")
    else:
        print("❌ No working AI systems found. Check the errors above.")

if __name__ == "__main__":
    main()
