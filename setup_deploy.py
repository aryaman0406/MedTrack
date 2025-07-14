#!/usr/bin/env python3
"""
Quick setup script for Medical Tracking App deployment
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and print results"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False

def main():
    print("🚀 Medical Tracking App - Deployment Setup")
    print("=" * 50)
    
    # Check if git is installed
    if not run_command("git --version", "Checking Git installation"):
        print("❌ Git is not installed. Please install Git first.")
        sys.exit(1)
    
    # Initialize git if needed
    if not os.path.exists('.git'):
        run_command("git init", "Initializing Git repository")
        run_command("git add .", "Adding files to Git")
        run_command('git commit -m "Initial commit - Medical Tracking App"', "Creating initial commit")
    else:
        print("✅ Git repository already exists")
    
    # Check for remote
    result = subprocess.run("git remote", shell=True, capture_output=True, text=True)
    if "origin" not in result.stdout:
        print("\n🔗 GitHub Setup Needed:")
        print("1. Create a new repository on GitHub")
        print("2. Copy the repository URL")
        repo_url = input("3. Enter your GitHub repository URL: ")
        if repo_url:
            run_command(f"git remote add origin {repo_url}", "Adding GitHub remote")
            run_command("git branch -M main", "Setting main branch")
    
    # Create .env template if it doesn't exist
    if not os.path.exists('.env'):
        env_content = """# API Keys for Medical Tracking App
GROQ_API_KEY=your_groq_api_key_here
XAI_API_KEY=your_xai_api_key_here
SECRET_KEY=your_random_secret_key_here
FLASK_ENV=development
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env template - Please update with your API keys")
    
    # Final push
    if input("\n📤 Push to GitHub now? (y/n): ").lower() == 'y':
        run_command("git add .", "Adding latest changes")
        run_command('git commit -m "Deploy: Ready for production deployment"', "Committing changes")
        run_command("git push -u origin main", "Pushing to GitHub")
    
    print("\n🎉 Setup Complete!")
    print("\n🌐 Deployment Options:")
    print("1. 📊 Render: https://render.com (Recommended)")
    print("2. 🚂 Railway: https://railway.app")
    print("3. ☁️ Heroku: https://heroku.com")
    print("4. 💻 GitHub Codespaces: In your repo")
    print("\n📋 See DEPLOYMENT_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    main()
