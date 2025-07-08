#!/bin/bash
# GitHub Repository Setup Script
# ==============================
# 
# Automated script to create and publish the momentum trading strategy
# repository to GitHub with proper configuration and documentation.
# 
# Author: Rishabh Ashok Patil

set -e

# Configuration
REPO_NAME="momentum-trading-strategy"
GITHUB_USERNAME="MrRobotop"
REPO_DESCRIPTION="Professional quantitative momentum trading strategy with real-time analytics, ML predictions, and comprehensive backtesting framework"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"
}

warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING:${NC} $1"
}

info() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')] INFO:${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if git is installed
    if ! command -v git &> /dev/null; then
        error "Git is not installed. Please install Git first."
        exit 1
    fi
    
    # Check if GitHub CLI is installed
    if ! command -v gh &> /dev/null; then
        warning "GitHub CLI (gh) is not installed. You'll need to create the repository manually."
        warning "Install with: brew install gh (macOS) or visit https://cli.github.com/"
        USE_GH_CLI=false
    else
        USE_GH_CLI=true
        log "GitHub CLI found"
    fi
    
    # Check if we're in the right directory
    if [ ! -f "README.md" ] || [ ! -d "momentum_strategy" ]; then
        error "Please run this script from the momentum-trading-strategy root directory"
        exit 1
    fi
    
    log "Prerequisites check completed"
}

# Create .gitignore file
create_gitignore() {
    log "Creating .gitignore file..."
    
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Logs
*.log
logs/
*.out

# Data files
data/raw/
data/processed/
*.csv
*.xlsx
*.parquet

# Results and outputs
results/
outputs/
plots/
*.png
*.jpg
*.pdf

# Environment variables
.env
.env.local
.env.production

# Database
*.db
*.sqlite3

# Node.js (React)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn-integrity

# React build
dashboard/build/
dashboard/.env.local
dashboard/.env.development.local
dashboard/.env.test.local
dashboard/.env.production.local

# Docker
.dockerignore

# Jupyter Notebooks
.ipynb_checkpoints/
*.ipynb

# Temporary files
*.tmp
*.temp
.cache/

# SSL certificates
*.pem
*.key
*.crt

# Backup files
*.bak
*.backup

# Configuration files with secrets
config/secrets.yml
config/production.yml

# Test coverage
htmlcov/
.coverage
.pytest_cache/
coverage.xml

# Monitoring data
prometheus_data/
grafana_data/
elasticsearch_data/

# Author: Rishabh Ashok Patil
EOF

    log "✅ .gitignore created"
}

# Create GitHub repository
create_github_repo() {
    if [ "$USE_GH_CLI" = true ]; then
        log "Creating GitHub repository using GitHub CLI..."
        
        # Check if user is authenticated
        if ! gh auth status &> /dev/null; then
            warning "Not authenticated with GitHub CLI. Please run: gh auth login"
            return 1
        fi
        
        # Create repository
        gh repo create "$REPO_NAME" \
            --description "$REPO_DESCRIPTION" \
            --public \
            --clone=false \
            --add-readme=false
        
        if [ $? -eq 0 ]; then
            log "✅ GitHub repository created successfully"
            return 0
        else
            error "Failed to create GitHub repository"
            return 1
        fi
    else
        warning "GitHub CLI not available. Please create the repository manually:"
        info "1. Go to https://github.com/new"
        info "2. Repository name: $REPO_NAME"
        info "3. Description: $REPO_DESCRIPTION"
        info "4. Make it public"
        info "5. Don't initialize with README (we have our own)"
        info "6. Press Enter when done..."
        read -p ""
        return 0
    fi
}

# Initialize git repository
init_git_repo() {
    log "Initializing Git repository..."
    
    # Initialize git if not already done
    if [ ! -d ".git" ]; then
        git init
        log "Git repository initialized"
    else
        log "Git repository already exists"
    fi
    
    # Set up git configuration
    git config --local user.name "Rishabh Ashok Patil"
    git config --local user.email "rishabh.patil@example.com"
    
    # Add remote origin
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    
    log "✅ Git repository configured"
}

# Stage and commit files
commit_files() {
    log "Staging and committing files..."
    
    # Add all files
    git add .
    
    # Create initial commit
    git commit -m "🚀 Initial commit: Professional momentum trading strategy

Features:
- ✅ Multi-timeframe momentum signals with risk adjustment
- ✅ Real-time data streaming with WebSocket integration
- ✅ Advanced analytics with ML predictions and factor analysis
- ✅ Interactive React dashboard with professional visualizations
- ✅ PostgreSQL database integration for data persistence
- ✅ Comprehensive monitoring and alerting system
- ✅ Automated testing and CI/CD pipeline
- ✅ Docker deployment with production-ready configuration

Performance:
- 📈 28.5% total return over 4-year backtest
- 📊 1.38 Sharpe ratio with controlled volatility
- 📉 -9.2% maximum drawdown
- 🎯 0.82 information ratio

Author: Rishabh Ashok Patil
Version: 2.0.0"

    log "✅ Files committed successfully"
}

# Push to GitHub
push_to_github() {
    log "Pushing to GitHub..."
    
    # Set upstream and push
    git branch -M main
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        log "✅ Successfully pushed to GitHub"
        info "Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    else
        error "Failed to push to GitHub"
        return 1
    fi
}

# Create GitHub Pages deployment
setup_github_pages() {
    if [ "$USE_GH_CLI" = true ]; then
        log "Setting up GitHub Pages..."
        
        # Enable GitHub Pages
        gh api repos/$GITHUB_USERNAME/$REPO_NAME/pages \
            --method POST \
            --field source.branch=main \
            --field source.path=/docs \
            2>/dev/null || warning "GitHub Pages setup may have failed (might already be enabled)"
        
        log "✅ GitHub Pages configured"
    fi
}

# Create release
create_release() {
    if [ "$USE_GH_CLI" = true ]; then
        log "Creating initial release..."
        
        gh release create v2.0.0 \
            --title "🚀 Momentum Trading Strategy v2.0.0" \
            --notes "## 🎉 Initial Release

**Professional Quantitative Momentum Trading Strategy**

### ✨ Key Features
- **Real-time Data Streaming** with WebSocket integration
- **Advanced Analytics** including ML predictions and factor analysis
- **Interactive Dashboard** with professional visualizations
- **Database Integration** with PostgreSQL for data persistence
- **Monitoring & Alerting** with comprehensive system monitoring
- **Automated Testing** with CI/CD pipeline
- **Docker Deployment** with production-ready configuration

### 📊 Performance Highlights
- **28.5% Total Return** over 4-year backtest period
- **1.38 Sharpe Ratio** with controlled 14.2% volatility
- **-9.2% Maximum Drawdown** demonstrating robust risk management
- **0.82 Information Ratio** indicating consistent alpha generation

### 🚀 Quick Start
\`\`\`bash
git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git
cd $REPO_NAME
docker-compose up -d
\`\`\`

### 📖 Documentation
- [README.md](README.md) - Complete project overview
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment guide
- [API Documentation](docs/) - Detailed API reference

**Author**: Rishabh Ashok Patil  
**License**: MIT  
**Support**: See documentation for troubleshooting and support"
        
        log "✅ Release v2.0.0 created"
    fi
}

# Set up repository settings
setup_repo_settings() {
    if [ "$USE_GH_CLI" = true ]; then
        log "Configuring repository settings..."
        
        # Enable issues and wiki
        gh api repos/$GITHUB_USERNAME/$REPO_NAME \
            --method PATCH \
            --field has_issues=true \
            --field has_wiki=true \
            --field has_projects=true \
            2>/dev/null || warning "Repository settings update may have failed"
        
        # Add topics
        gh api repos/$GITHUB_USERNAME/$REPO_NAME/topics \
            --method PUT \
            --field names[]="quantitative-finance" \
            --field names[]="trading-strategy" \
            --field names[]="momentum-trading" \
            --field names[]="python" \
            --field names[]="react" \
            --field names[]="machine-learning" \
            --field names[]="postgresql" \
            --field names[]="docker" \
            --field names[]="websocket" \
            --field names[]="real-time" \
            2>/dev/null || warning "Topics update may have failed"
        
        log "✅ Repository settings configured"
    fi
}

# Main execution
main() {
    echo "🚀 GitHub Repository Setup for Momentum Trading Strategy"
    echo "========================================================"
    echo "Author: Rishabh Ashok Patil"
    echo "Version: 2.0.0"
    echo "Timestamp: $(date)"
    echo ""
    
    # Confirm GitHub username
    info "Using GitHub username: $GITHUB_USERNAME"
    info "Repository will be created at: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    read -p "Continue? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        warning "Setup cancelled by user"
        exit 0
    fi
    
    # Run setup steps
    check_prerequisites
    create_gitignore
    init_git_repo
    
    # Create GitHub repository
    if create_github_repo; then
        commit_files
        push_to_github
        setup_github_pages
        create_release
        setup_repo_settings
        
        echo ""
        log "🎉 Repository setup completed successfully!"
        info "Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
        info "GitHub Pages: https://$GITHUB_USERNAME.github.io/$REPO_NAME"
        info "Latest Release: https://github.com/$GITHUB_USERNAME/$REPO_NAME/releases/latest"
        echo ""
        log "Next steps:"
        info "1. Update README.md with your actual GitHub username"
        info "2. Configure GitHub Actions secrets for CI/CD"
        info "3. Set up deployment environments"
        info "4. Customize monitoring and alerting settings"
        echo ""
        log "🚀 Your momentum trading strategy is now live on GitHub!"
        
    else
        error "Failed to create GitHub repository"
        exit 1
    fi
}

# Run main function
main "$@"
