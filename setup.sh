#!/bin/bash

# Backlog Builder Local Testing Setup Script
# This script sets up the development environment for local testing

set -e

echo "🚀 Setting up Backlog Builder for local testing..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if running in the correct directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the Backlog_builder root directory"
    exit 1
fi

print_header "1. Checking system requirements..."

# Check for Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check for Node.js
if ! command -v node &> /dev/null; then
    print_warning "Node.js is not installed. Some features may not work."
fi

print_status "System requirements check completed"

print_header "2. Setting up environment files..."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Backlog Builder Environment Configuration

# General Settings
NODE_ENV=development
PORT=3000
PREFER_LOCAL_AI=true

# AI Service URLs
LOCAL_STT_URL=http://localhost:8001
LOCAL_LLM_URL=http://localhost:8002
LLM_PORT=8002
STT_PORT=8001

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Database Configuration (PostgreSQL)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=backlog_builder
POSTGRES_USER=backlog_user
POSTGRES_PASSWORD=backlog_password

# Model Configuration
MODEL_CACHE_DIR=./models
TORCH_HOME=./models
HF_HOME=./models
TRANSFORMERS_CACHE=./models

# Security
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
BCRYPT_ROUNDS=12

# File Upload Settings
MAX_FILE_SIZE=100MB
UPLOAD_DIR=./uploads
ALLOWED_EXTENSIONS=.mp3,.wav,.m4a,.mp4,.avi,.mov

# GitHub Integration (Optional)
GITHUB_TOKEN=
GITHUB_REPO_OWNER=
GITHUB_REPO_NAME=

# Cloud AI Services (Optional - for fallback)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
EOF
    print_status "Created .env file with default settings"
else
    print_status ".env file already exists"
fi

print_header "3. Creating necessary directories..."

# Create directories if they don't exist
mkdir -p uploads
mkdir -p models
mkdir -p logs
mkdir -p backend/storage/uploads
mkdir -p backend/storage/transcripts
mkdir -p backend/storage/processed

print_status "Directories created"

print_header "4. Setting up Python virtual environment for AI services..."

# Create virtual environment for Python AI services
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Python virtual environment created"
fi

# Activate virtual environment and install dependencies
source venv/bin/activate

print_status "Installing Python dependencies..."
pip install --upgrade pip

# Install core dependencies first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
cd ai-services/llm-processing
pip install -r requirements.txt
cd ../..

# Install spaCy English model
python -m spacy download en_core_web_sm

print_status "Python dependencies installed"

print_header "5. Installing Node.js dependencies..."

# Install backend dependencies
if [ -d "backend" ]; then
    cd backend
    if [ -f "package.json" ]; then
        npm install
        print_status "Backend dependencies installed"
    fi
    cd ..
fi

# Install frontend dependencies
if [ -d "frontend" ]; then
    cd frontend
    if [ -f "package.json" ]; then
        npm install
        print_status "Frontend dependencies installed"
    fi
    cd ..
fi

print_header "6. Starting Redis server..."

# Check if Redis is running
if ! pgrep -x "redis-server" > /dev/null; then
    # Try to start Redis with Docker
    docker run -d --name backlog-redis -p 6379:6379 redis:7-alpine
    print_status "Started Redis server in Docker"
else
    print_status "Redis server is already running"
fi

print_header "7. Creating test scripts..."

# Create a test script for the LLM service
cat > test_llm_service.py << 'EOF'
#!/usr/bin/env python3
"""
Test script for the Local LLM Service
"""

import asyncio
import json
import sys
import time
import requests

# Test data
TEST_TRANSCRIPT = """
John: Welcome everyone to the sprint planning meeting. We need to discuss the user authentication feature.
Sarah: I think we should implement OAuth2 for better security. I can work on this next week.
Mike: Agreed. We also need to fix the login bug that users reported. Sarah, can you handle that too?
Sarah: Sure, I'll take both tasks. The OAuth2 implementation should be done by Friday.
John: Great. Mike, can you work on the database optimization we discussed?
Mike: Yes, I'll start that tomorrow. Should be completed by next Wednesday.
John: Perfect. Let's also plan the API documentation update.
Sarah: I can help with that after the authentication work is done.
John: Excellent. Meeting adjourned.
"""

def test_health_endpoint():
    """Test the health check endpoint"""
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data.get('status')}")
            print(f"   GPU Available: {data.get('gpu_available')}")
            print(f"   Device: {data.get('device')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_meeting_analysis():
    """Test the meeting analysis endpoint"""
    try:
        payload = {
            "transcript": TEST_TRANSCRIPT,
            "meeting_type": "sprint_planning",
            "extract_action_items": True,
            "extract_decisions": True,
            "generate_summary": True,
            "detect_sentiment": True
        }
        
        print("🧪 Testing meeting analysis...")
        response = requests.post("http://localhost:8002/analyze", json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Meeting analysis successful")
            print(f"   Processing time: {data.get('processing_time', 0):.2f} seconds")
            print(f"   Action items found: {len(data.get('action_items', []))}")
            print(f"   Decisions found: {len(data.get('decisions', []))}")
            if data.get('summary'):
                print(f"   Summary generated: ✅")
            return True
        else:
            print(f"❌ Meeting analysis failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Meeting analysis failed: {e}")
        return False

def test_ticket_generation():
    """Test the ticket generation endpoint"""
    try:
        payload = {
            "action_item": "Implement OAuth2 authentication for better security",
            "context": "Sprint planning meeting - user authentication feature discussion",
            "ticket_type": "feature"
        }
        
        print("🎫 Testing ticket generation...")
        response = requests.post("http://localhost:8002/generate-ticket", json=payload, timeout=20)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Ticket generation successful")
            print(f"   Title: {data.get('title', 'N/A')}")
            print(f"   Priority: {data.get('priority', 'N/A')}")
            print(f"   Labels: {', '.join(data.get('labels', []))}")
            return True
        else:
            print(f"❌ Ticket generation failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ticket generation failed: {e}")
        return False

def main():
    print("🚀 Testing Backlog Builder LLM Service")
    print("=" * 50)
    
    # Wait a bit for service to be ready
    print("⏳ Waiting for service to be ready...")
    time.sleep(2)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_health_endpoint():
        tests_passed += 1
    
    if test_meeting_analysis():
        tests_passed += 1
        
    if test_ticket_generation():
        tests_passed += 1
    
    print("=" * 50)
    print(f"🏁 Tests completed: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! The LLM service is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the service logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

chmod +x test_llm_service.py

# Create a startup script for local development
cat > start_local_dev.sh << 'EOF'
#!/bin/bash

# Start Backlog Builder in local development mode

set -e

echo "🚀 Starting Backlog Builder Local Development Environment"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if .env exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found. Run ./setup.sh first."
    exit 1
fi

# Load environment variables
export $(cat .env | grep -v '#' | awk '/=/ {print $1}')

print_status "Starting services..."

# Start Redis if not running
if ! pgrep -x "redis-server" > /dev/null; then
    print_status "Starting Redis..."
    docker run -d --name backlog-redis -p 6379:6379 redis:7-alpine || true
fi

# Activate Python virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    print_status "Activated Python virtual environment"
fi

# Start the LLM service in background
print_status "Starting LLM Processing Service on port ${LLM_PORT:-8002}..."
cd ai-services/llm-processing
python huggingface-local.py &
LLM_PID=$!
cd ../..

# Wait a bit for LLM service to start
sleep 5

# Start the backend API service
if [ -d "backend" ]; then
    print_status "Starting Backend API Service on port ${PORT:-3000}..."
    cd backend
    npm run dev &
    BACKEND_PID=$!
    cd ..
fi

# Start the frontend development server
if [ -d "frontend" ]; then
    print_status "Starting Frontend Development Server..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
fi

print_status "All services started!"
print_status "LLM Service: http://localhost:${LLM_PORT:-8002}"
print_status "Backend API: http://localhost:${PORT:-3000}"
print_status "Frontend: http://localhost:5173 (default Vite port)"
print_status "Redis: localhost:6379"

# Function to cleanup on exit
cleanup() {
    echo ""
    print_status "Shutting down services..."
    
    if [ ! -z "$LLM_PID" ]; then
        kill $LLM_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    
    # Stop Redis container
    docker stop backlog-redis 2>/dev/null || true
    docker rm backlog-redis 2>/dev/null || true
    
    print_status "Cleanup completed"
}

# Trap signals to cleanup
trap cleanup EXIT INT TERM

print_status "Press Ctrl+C to stop all services"

# Wait for user to stop
wait
EOF

chmod +x start_local_dev.sh

print_status "Test scripts created"

print_header "8. Final setup steps..."

echo ""
print_status "✅ Setup completed successfully!"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Start the development environment:"
echo "   ${GREEN}./start_local_dev.sh${NC}"
echo ""
echo "2. Test the LLM service:"
echo "   ${GREEN}python test_llm_service.py${NC}"
echo ""
echo "3. Access the services:"
echo "   - LLM Service: http://localhost:8002/docs"
echo "   - Backend API: http://localhost:3000"
echo "   - Frontend: http://localhost:5173"
echo ""
echo -e "${YELLOW}Note:${NC} The first run may take several minutes as AI models are downloaded."
echo ""
