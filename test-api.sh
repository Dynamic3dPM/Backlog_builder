#!/bin/bash

# Backlog Builder API Testing Script
# This script provides various ways to test the Backlog Builder API endpoints

BASE_URL="http://localhost:3000"

echo "=== Backlog Builder API Testing ==="
echo "Base URL: $BASE_URL"
echo ""

# Test 1: Health Check
echo "1. Testing Health Check..."
curl -X GET "$BASE_URL/health" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

# Test 2: AI Health Check
echo "2. Testing AI Health Check..."
curl -X GET "$BASE_URL/api/ai/health" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

# Test 3: AI Service Status
echo "3. Testing AI Service Status..."
curl -X GET "$BASE_URL/api/ai/status" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

# Test 4: Get AI Providers
echo "4. Testing Get AI Providers..."
curl -X GET "$BASE_URL/api/ai/providers" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

# Test 5: Get Available Templates
echo "5. Testing Get Available Templates..."
curl -X GET "$BASE_URL/api/ai/templates" \
  -H "Content-Type: application/json" \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

# Test 6: Test AI Providers Connectivity
echo "6. Testing AI Providers Connectivity..."
curl -X POST "$BASE_URL/api/ai/providers/test" \
  -H "Content-Type: application/json" \
  -d '{"providers": ["local"]}' \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

# Test 7: Analyze Sample Text (if you want to test without audio)
echo "7. Testing Meeting Analysis with Sample Text..."
curl -X POST "$BASE_URL/api/ai/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "In todays meeting, we discussed the new user authentication system. John will implement OAuth2 by Friday. Sarah will create the user interface mockups by Wednesday. We decided to use PostgreSQL for the database. The deployment should be ready by next Monday.",
    "options": {
      "extractActionItems": true,
      "generateSummary": true,
      "identifyDecisions": true
    }
  }' \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

# Test 8: Generate Action Items
echo "8. Testing Action Items Generation..."
curl -X POST "$BASE_URL/api/ai/action-items" \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "John needs to finish the authentication module by Friday. Sarah should review the API documentation. We need to schedule a follow-up meeting next week."
  }' \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

# Test 9: Generate Summary
echo "9. Testing Summary Generation..."
curl -X POST "$BASE_URL/api/ai/summary" \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "We had a productive meeting discussing the new project timeline. The team agreed on the deliverables and assigned responsibilities to each member."
  }' \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

# Test 10: Extract Decisions
echo "10. Testing Decision Extraction..."
curl -X POST "$BASE_URL/api/ai/decisions" \
  -H "Content-Type: application/json" \
  -d '{
    "transcription": "After much discussion, we decided to use React for the frontend. We also agreed to deploy on AWS instead of Azure. The budget was approved for the additional security features."
  }' \
  -w "\nHTTP Status: %{http_code}\n" \
  2>/dev/null
echo ""

echo "=== Testing Complete ==="
echo ""
echo "To test file upload endpoints, use:"
echo "curl -X POST $BASE_URL/api/ai/transcribe -F 'audio=@/path/to/your/audio/file.mp3'"
echo ""
echo "For more detailed testing, check the generated Postman collection or use the interactive testing script."
