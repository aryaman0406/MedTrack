#!/usr/bin/env python3
"""
Test script to debug medibot integration
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Configuration
DB_FAISS_PATH = "medical-chatbot/vectorstore/db_faiss"

def test_vectorstore():
    """Test if vectorstore loads correctly"""
    try:
        print("Testing vectorstore loading...")
        embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
        
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✓ Vectorstore loaded successfully")
        
        # Test retrieval
        retriever = db.as_retriever(search_kwargs={'k': 3})
        docs = retriever.invoke("What is diabetes?")
        print(f"✓ Retrieved {len(docs)} documents")
        
        return db
    except Exception as e:
        print(f"✗ Error loading vectorstore: {e}")
        return None

def test_groq_api():
    """Test if Groq API is working"""
    try:
        print("\nTesting Groq API...")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            print("✗ GROQ_API_KEY not found")
            return False
        
        print("✓ GROQ_API_KEY found")
        
        llm = ChatGroq(
            model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
            temperature=0.0,
            groq_api_key=groq_api_key,
        )
        print("✓ Groq LLM initialized")
        
        # Test a simple completion
        response = llm.invoke("Hello, how are you?")
        print(f"✓ Groq API test successful: {response.content[:50]}...")
        
        return True
    except Exception as e:
        print(f"✗ Error testing Groq API: {e}")
        return False

def test_full_pipeline():
    """Test the complete medibot pipeline"""
    try:
        print("\nTesting full medibot pipeline...")
        
        # Load vectorstore
        vectorstore = test_vectorstore()
        if not vectorstore:
            return False
        
        # Test Groq API
        if not test_groq_api():
            return False
        
        # Set up prompt template
        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """
        
        prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                groq_api_key=os.getenv("GROQ_API_KEY"),
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
        
        print("✓ QA chain created successfully")
        
        # Test query
        test_question = "What are the symptoms of diabetes?"
        response = qa_chain.invoke({'query': test_question})
        
        print(f"✓ Test query successful")
        print(f"Question: {test_question}")
        print(f"Answer: {response['result'][:100]}...")
        print(f"Sources: {len(response['source_documents'])} documents")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in full pipeline test: {e}")
        return False

if __name__ == "__main__":
    print("🔬 Testing Medibot Integration")
    print("=" * 50)
    
    success = test_full_pipeline()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! Medibot integration should work.")
    else:
        print("❌ Tests failed. Check the errors above.")
