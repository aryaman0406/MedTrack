#!/bin/bash

# Medical Tracking App - Quick Deploy Script

echo "🚀 Medical Tracking App Deployment Helper"
echo "=========================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    git add .
    git commit -m "Initial commit - Medical Tracking App"
    echo "✅ Git repository created"
else
    echo "📁 Git repository found"
fi

# Check for GitHub remote
if ! git remote | grep -q "origin"; then
    echo ""
    echo "🔗 GitHub Setup Required:"
    echo "1. Create a new repository on GitHub"
    echo "2. Run: git remote add origin https://github.com/yourusername/medical-tracking-app.git"
    echo "3. Run: git branch -M main"
    echo "4. Run: git push -u origin main"
    echo ""
else
    echo "✅ GitHub remote configured"
    
    # Push latest changes
    echo "📤 Pushing latest changes..."
    git add .
    git commit -m "Deploy: Updated for production deployment"
    git push origin main
    echo "✅ Code pushed to GitHub"
fi

echo ""
echo "🌐 Deployment Options:"
echo ""
echo "📊 Option 1 - Render (Recommended):"
echo "   1. Go to https://render.com"
echo "   2. Sign up with GitHub"
echo "   3. Create 'Web Service' from your repo"
echo "   4. Add environment variables:"
echo "      - GROQ_API_KEY=your_key_here"
echo "      - XAI_API_KEY=your_key_here"
echo "      - SECRET_KEY=random_secret_key_here"
echo ""
echo "🚂 Option 2 - Railway:"
echo "   1. Go to https://railway.app"
echo "   2. Connect GitHub repo"
echo "   3. Add same environment variables"
echo ""
echo "☁️ Option 3 - Heroku:"
echo "   1. Install Heroku CLI"
echo "   2. Run: heroku create your-app-name"
echo "   3. Run: git push heroku main"
echo ""
echo "💻 Option 4 - GitHub Codespaces:"
echo "   1. Go to your GitHub repo"
echo "   2. Click 'Code' → 'Codespaces'"
echo "   3. Create new codespace"
echo "   4. Run: python app.py"
echo ""
echo "✨ All deployment files created successfully!"
echo "📋 Next: Choose a platform above and follow the steps"
