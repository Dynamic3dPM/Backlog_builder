#!/bin/bash

# Interactive API Testing Script for Backlog Builder
# This script provides an interactive menu to test different API endpoints

BASE_URL="http://localhost:3000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to make API calls and display formatted output
make_api_call() {
    local method=$1
    local endpoint=$2
    local data=$3
    local content_type=$4
    
    echo -e "${BLUE}Making $method request to: $endpoint${NC}"
    
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" "$BASE_URL$endpoint" -H "Content-Type: ${content_type:-application/json}")
    else
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X "$method" "$BASE_URL$endpoint" -H "Content-Type: ${content_type:-application/json}" -d "$data")
    fi
    
    # Split response and status
    body=$(echo "$response" | sed '$d')
    status=$(echo "$response" | tail -n1 | cut -d: -f2)
    
    echo -e "${YELLOW}Response Status: $status${NC}"
    
    if [ "$status" -ge 200 ] && [ "$status" -lt 300 ]; then
        echo -e "${GREEN}Response Body:${NC}"
        echo "$body" | jq . 2>/dev/null || echo "$body"
    else
        echo -e "${RED}Error Response:${NC}"
        echo "$body" | jq . 2>/dev/null || echo "$body"
    fi
    
    echo ""
    read -p "Press Enter to continue..."
    echo ""
}

# Function to test file upload
test_file_upload() {
    echo -e "${BLUE}Testing File Upload Endpoint${NC}"
    echo "Note: This requires an actual audio file to test properly."
    echo "You can create a test file with: 'echo \"test\" > test.mp3'"
    echo ""
    
    read -p "Enter path to audio file (or press Enter to skip): " file_path
    
    if [ -n "$file_path" ] && [ -f "$file_path" ]; then
        echo -e "${BLUE}Uploading file: $file_path${NC}"
        response=$(curl -s -w "\nHTTP_STATUS:%{http_code}" -X POST "$BASE_URL/api/ai/transcribe" -F "audio=@$file_path")
        
        body=$(echo "$response" | sed '$d')
        status=$(echo "$response" | tail -n1 | cut -d: -f2)
        
        echo -e "${YELLOW}Response Status: $status${NC}"
        
        if [ "$status" -ge 200 ] && [ "$status" -lt 300 ]; then
            echo -e "${GREEN}Response Body:${NC}"
            echo "$body" | jq . 2>/dev/null || echo "$body"
        else
            echo -e "${RED}Error Response:${NC}"
            echo "$body" | jq . 2>/dev/null || echo "$body"
        fi
    else
        echo "File not found or no file specified. Skipping upload test."
    fi
    
    echo ""
    read -p "Press Enter to continue..."
    echo ""
}

# Main menu
while true; do
    clear
    echo -e "${GREEN}=== Backlog Builder API Interactive Tester ===${NC}"
    echo -e "${BLUE}Base URL: $BASE_URL${NC}"
    echo ""
    echo "1. Test Backend Health Check"
    echo "2. Test AI Service Health Check"
    echo "3. Test AI Service Status"
    echo "4. Test AI Providers"
    echo "5. Test AI Provider Connectivity"
    echo "6. Test Get Prompt Templates"
    echo "7. Test Meeting Analysis (with sample data)"
    echo "8. Test Action Items Generation"
    echo "9. Test Summary Generation"
    echo "10. Test Decision Extraction"
    echo "11. Test Generate Tickets"
    echo "12. Test File Upload (Transcribe)"
    echo "13. Test Usage Analytics"
    echo "14. Test All Endpoints (Quick Run)"
    echo "15. Exit"
    echo ""
    read -p "Select an option (1-15): " choice
    
    case $choice in
        1)
            make_api_call "GET" "/health"
            ;;
        2)
            make_api_call "GET" "/api/ai/health"
            ;;
        3)
            make_api_call "GET" "/api/ai/status"
            ;;
        4)
            make_api_call "GET" "/api/ai/providers"
            ;;
        5)
            make_api_call "POST" "/api/ai/providers/test" '{"providers": ["local"]}'
            ;;
        6)
            make_api_call "GET" "/api/ai/templates"
            ;;
        7)
            data='{
  "transcription": "In todays meeting, we discussed the new user authentication system. John will implement OAuth2 by Friday. Sarah will create the user interface mockups by Wednesday. We decided to use PostgreSQL for the database. The deployment should be ready by next Monday.",
  "options": {
    "extractActionItems": true,
    "generateSummary": true,
    "identifyDecisions": true
  }
}'
            make_api_call "POST" "/api/ai/analyze" "$data"
            ;;
        8)
            data='{
  "transcription": "John needs to finish the authentication module by Friday. Sarah should review the API documentation. We need to schedule a follow-up meeting next week."
}'
            make_api_call "POST" "/api/ai/action-items" "$data"
            ;;
        9)
            data='{
  "transcription": "We had a productive meeting discussing the new project timeline. The team agreed on the deliverables and assigned responsibilities to each member."
}'
            make_api_call "POST" "/api/ai/summary" "$data"
            ;;
        10)
            data='{
  "transcription": "After much discussion, we decided to use React for the frontend. We also agreed to deploy on AWS instead of Azure. The budget was approved for the additional security features."
}'
            make_api_call "POST" "/api/ai/decisions" "$data"
            ;;
        11)
            data='{
  "analysis": {
    "actionItems": [
      "Implement OAuth2 authentication",
      "Create UI mockups",
      "Set up PostgreSQL database"
    ],
    "decisions": [
      "Use PostgreSQL for database",
      "Deploy by next Monday"
    ],
    "summary": "Meeting about new authentication system"
  },
  "project": "authentication-project",
  "template": "default"
}'
            make_api_call "POST" "/api/ai/generate-tickets" "$data"
            ;;
        12)
            test_file_upload
            ;;
        13)
            make_api_call "GET" "/api/ai/analytics/usage"
            ;;
        14)
            echo -e "${YELLOW}Running all tests...${NC}"
            ./test-api.sh
            read -p "Press Enter to continue..."
            ;;
        15)
            echo -e "${GREEN}Goodbye!${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option. Please try again.${NC}"
            sleep 2
            ;;
    esac
done
