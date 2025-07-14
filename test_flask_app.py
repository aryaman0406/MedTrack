#!/usr/bin/env python3
"""
Simple test to verify Flask app starts without errors
"""
import os
import sys
from dotenv import load_dotenv

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_app_import():
    """Test if the Flask app can be imported without errors"""
    try:
        from app import app, db
        print("✅ Flask app imported successfully")
        
        # Test app context
        with app.app_context():
            print("✅ Flask app context works")
            
        return True
    except Exception as e:
        print(f"❌ Flask app import/context error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Flask app with medibot integration...")
    print("-" * 50)
    
    if test_app_import():
        print("\n🎉 Flask app with medibot integration is ready!")
        print("You can now run: python app.py")
    else:
        print("\n⚠️ Flask app test failed. Please check the errors above.")
