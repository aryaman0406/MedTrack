# Medical Tracking App - Quick Deploy Script for Windows

Write-Host "🚀 Medical Tracking App Deployment Helper" -ForegroundColor Green
Write-Host "===========================================" -ForegroundColor Green

# Check if git is initialized
if (-not (Test-Path ".git")) {
    Write-Host "📁 Initializing Git repository..." -ForegroundColor Yellow
    git init
    git add .
    git commit -m "Initial commit - Medical Tracking App"
    Write-Host "✅ Git repository created" -ForegroundColor Green
} else {
    Write-Host "📁 Git repository found" -ForegroundColor Green
}

# Check for GitHub remote
$remotes = git remote
if (-not ($remotes -contains "origin")) {
    Write-Host ""
    Write-Host "🔗 GitHub Setup Required:" -ForegroundColor Cyan
    Write-Host "1. Create a new repository on GitHub" -ForegroundColor White
    Write-Host "2. Run: git remote add origin https://github.com/yourusername/medical-tracking-app.git" -ForegroundColor White
    Write-Host "3. Run: git branch -M main" -ForegroundColor White
    Write-Host "4. Run: git push -u origin main" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "✅ GitHub remote configured" -ForegroundColor Green
    
    # Push latest changes
    Write-Host "📤 Pushing latest changes..." -ForegroundColor Yellow
    git add .
    git commit -m "Deploy: Updated for production deployment"
    git push origin main
    Write-Host "✅ Code pushed to GitHub" -ForegroundColor Green
}

Write-Host ""
Write-Host "🌐 Deployment Options:" -ForegroundColor Magenta
Write-Host ""
Write-Host "📊 Option 1 - Render (Recommended):" -ForegroundColor Cyan
Write-Host "   1. Go to https://render.com" -ForegroundColor White
Write-Host "   2. Sign up with GitHub" -ForegroundColor White
Write-Host "   3. Create 'Web Service' from your repo" -ForegroundColor White
Write-Host "   4. Add environment variables:" -ForegroundColor White
Write-Host "      - GROQ_API_KEY=your_key_here" -ForegroundColor Gray
Write-Host "      - XAI_API_KEY=your_key_here" -ForegroundColor Gray
Write-Host "      - SECRET_KEY=random_secret_key_here" -ForegroundColor Gray
Write-Host ""
Write-Host "🚂 Option 2 - Railway:" -ForegroundColor Cyan
Write-Host "   1. Go to https://railway.app" -ForegroundColor White
Write-Host "   2. Connect GitHub repo" -ForegroundColor White
Write-Host "   3. Add same environment variables" -ForegroundColor White
Write-Host ""
Write-Host "☁️ Option 3 - Heroku:" -ForegroundColor Cyan
Write-Host "   1. Install Heroku CLI" -ForegroundColor White
Write-Host "   2. Run: heroku create your-app-name" -ForegroundColor White
Write-Host "   3. Run: git push heroku main" -ForegroundColor White
Write-Host ""
Write-Host "💻 Option 4 - GitHub Codespaces:" -ForegroundColor Cyan
Write-Host "   1. Go to your GitHub repo" -ForegroundColor White
Write-Host "   2. Click 'Code' → 'Codespaces'" -ForegroundColor White
Write-Host "   3. Create new codespace" -ForegroundColor White
Write-Host "   4. Run: python app.py" -ForegroundColor White
Write-Host ""
Write-Host "✨ All deployment files created successfully!" -ForegroundColor Green
Write-Host "📋 Next: Choose a platform above and follow the steps" -ForegroundColor Yellow

# Keep window open
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
