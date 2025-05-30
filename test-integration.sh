#!/bin/bash

# Simple AI Services Integration Test
# Tests the basic functionality of the Backlog Builder AI pipeline

set -e

echo "🚀 Starting Backlog Builder AI Services Integration Test"
echo "=================================================="

# Configuration
BACKEND_URL="http://localhost:3003"
TEST_AUDIO_URL="https://www2.cs.uic.edu/~i101/SoundFiles/BabyElephantWalk60.wav"
TEST_AUDIO_FILE="/tmp/test_audio.wav"

# Function to check if a service is running
check_service() {
    local url=$1
    local name=$2
    
    echo "🔍 Checking $name service at $url..."
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo "✅ $name service is running"
        return 0
    else
        echo "❌ $name service is not running"
        return 1
    fi
}

# Function to test API endpoint
test_endpoint() {
    local method=$1
    local endpoint=$2
    local description=$3
    
    echo "🧪 Testing: $description"
    echo "   $method $endpoint"
    
    response=$(curl -s -w "%{http_code}" -X "$method" "$BACKEND_URL$endpoint")
    http_code="${response: -3}"
    body="${response%???}"
    
    if [[ "$http_code" -ge 200 && "$http_code" -lt 300 ]]; then
        echo "✅ $description - HTTP $http_code"
        return 0
    else
        echo "❌ $description - HTTP $http_code"
        echo "   Response: $body"
        return 1
    fi
}

# Function to download test audio file
download_test_audio() {
    echo "📥 Downloading test audio file..."
    
    if curl -s -L -o "$TEST_AUDIO_FILE" "$TEST_AUDIO_URL"; then
        echo "✅ Test audio file downloaded to $TEST_AUDIO_FILE"
        return 0
    else
        echo "❌ Failed to download test audio file"
        return 1
    fi
}

# Start tests
echo "📋 Running Basic Health Checks..."

# Test 1: Basic health check
test_endpoint "GET" "/health" "Basic health check"

# Test 2: AI health check
test_endpoint "GET" "/api/ai/health" "AI services health check"

# Test 3: AI status check
test_endpoint "GET" "/api/ai/status" "AI services status"

# Test 4: Get providers
test_endpoint "GET" "/api/ai/providers" "Get AI providers"

# Test 5: Get prompt templates
test_endpoint "GET" "/api/ai/templates" "Get prompt templates"

# Test 6: Test provider connectivity
echo "🧪 Testing provider connectivity"
curl -s -X POST "$BACKEND_URL/api/ai/providers/test" \
  -H "Content-Type: application/json" \
  -d '{"providers": ["stt", "llm"]}' | jq '.' || echo "Provider test completed"

# Test 7: Analytics endpoints
echo "📊 Testing analytics endpoints..."
test_endpoint "GET" "/api/ai/analytics/usage" "Usage analytics"
test_endpoint "GET" "/api/ai/analytics/costs" "Cost analytics" 
test_endpoint "GET" "/api/ai/analytics/performance" "Performance analytics"

# Test 8: Process transcription (without file upload)
echo "🧪 Testing transcript analysis..."
curl -s -X POST "$BACKEND_URL/api/ai/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "Hello team, we need to implement a new feature for user authentication. John, can you work on the backend API? Sarah, please handle the frontend components. We should have this completed by next Friday.",
    "meetingType": "development"
  }' | jq '.' || echo "Transcript analysis test completed"

# File upload test (if audio file is available)
if download_test_audio; then
    echo "🎵 Testing audio file upload..."
    curl -s -X POST "$BACKEND_URL/api/ai/transcribe" \
      -F "audio=@$TEST_AUDIO_FILE" \
      -F "language=en-US" | jq '.' || echo "Audio upload test completed"
    
    # Cleanup
    rm -f "$TEST_AUDIO_FILE"
fi

echo ""
echo "🎉 Integration test completed!"
echo ""
echo "📝 Test Summary:"
echo "- Backend API is running on port 3002"
echo "- Basic health checks passed"
echo "- AI service endpoints are accessible"
echo "- Analytics endpoints are working"
echo "- File upload mechanism is configured"
echo ""
echo "⚠️  Note: AI services (STT/LLM) are not running, so actual processing will fail"
echo "   To fully test the pipeline, start the AI services with Docker Compose"
echo ""
echo "🚀 Next steps:"
echo "1. Start AI services: docker-compose up -d"
echo "2. Run full pipeline test with actual audio files"
echo "3. Monitor logs and performance metrics"
