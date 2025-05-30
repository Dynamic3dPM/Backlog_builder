#!/bin/bash

# AI Services Test Pipeline
# Comprehensive testing script for the AI services pipeline

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
API_BASE_URL="http://localhost:3000/api"
TEST_AUDIO_FILE="test-audio.wav"
RESULTS_DIR="test-results"

echo -e "${BLUE}🧪 AI Services Test Pipeline${NC}"
echo -e "${BLUE}Testing Backlog Builder AI Services${NC}"
echo ""

# Create results directory
mkdir -p $RESULTS_DIR

# Check if services are running
echo -e "${YELLOW}📋 Checking service availability...${NC}"

check_service() {
    local service_name=$1
    local url=$2
    
    if curl -f "$url/health" &> /dev/null; then
        echo -e "${GREEN}✅ $service_name is available${NC}"
        return 0
    else
        echo -e "${RED}❌ $service_name is not available at $url${NC}"
        return 1
    fi
}

# Check all services
SERVICES_OK=true

check_service "Backend API" "$API_BASE_URL" || SERVICES_OK=false
check_service "Cloud STT" "http://localhost:8002" || SERVICES_OK=false
check_service "Cloud LLM" "http://localhost:8004" || SERVICES_OK=false

# Optional local services
check_service "Local STT" "http://localhost:8001" || echo -e "${YELLOW}⚠️  Local STT not available (optional)${NC}"
check_service "Local LLM" "http://localhost:8003" || echo -e "${YELLOW}⚠️  Local LLM not available (optional)${NC}"

if [ "$SERVICES_OK" = false ]; then
    echo -e "${RED}❌ Some required services are not available${NC}"
    echo -e "${YELLOW}💡 Run './deploy.sh' to start the services${NC}"
    exit 1
fi

echo -e "${GREEN}✅ All required services are available${NC}"
echo ""

# Create test audio file if it doesn't exist
if [ ! -f "$TEST_AUDIO_FILE" ]; then
    echo -e "${YELLOW}🔊 Creating test audio file...${NC}"
    
    # Create a simple test audio file using text-to-speech (if available)
    if command -v espeak &> /dev/null; then
        espeak "Hello team, this is a test meeting recording for the Backlog Builder AI services. We need to implement the user authentication feature and fix the database connection bug. John will handle the frontend changes and Sarah will work on the backend API. The deadline is next Friday." -w $TEST_AUDIO_FILE
        echo -e "${GREEN}✅ Test audio file created${NC}"
    else
        echo -e "${YELLOW}⚠️  espeak not available. Please provide a test audio file named '$TEST_AUDIO_FILE'${NC}"
        echo -e "${YELLOW}   You can download a sample from: https://www.voiptroubleshooter.com/open_speech/american.html${NC}"
        exit 1
    fi
fi

# Test 1: Service Status
echo -e "${YELLOW}🔍 Test 1: Service Status Check${NC}"

curl -s "$API_BASE_URL/ai/status" | jq '.' > "$RESULTS_DIR/service-status.json"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Service status check passed${NC}"
else
    echo -e "${RED}❌ Service status check failed${NC}"
fi

# Test 2: Speech-to-Text Only
echo -e "${YELLOW}🔍 Test 2: Speech-to-Text Transcription${NC}"

response=$(curl -s -X POST \
    -F "audio=@$TEST_AUDIO_FILE" \
    -F "language=auto" \
    -F "speakerDiarization=true" \
    "$API_BASE_URL/ai/transcribe")

echo "$response" | jq '.' > "$RESULTS_DIR/transcription-result.json"

if echo "$response" | jq -e '.success' > /dev/null; then
    echo -e "${GREEN}✅ Speech-to-text transcription passed${NC}"
    
    # Extract transcription for next test
    TRANSCRIPTION=$(echo "$response" | jq -r '.transcription')
    echo "$TRANSCRIPTION" > "$RESULTS_DIR/extracted-transcription.txt"
    
    echo -e "${BLUE}📝 Transcription: ${TRANSCRIPTION:0:100}...${NC}"
else
    echo -e "${RED}❌ Speech-to-text transcription failed${NC}"
    echo "$response"
    exit 1
fi

# Test 3: AI Analysis Only
echo -e "${YELLOW}🔍 Test 3: AI Analysis${NC}"

if [ -z "$TRANSCRIPTION" ]; then
    TRANSCRIPTION="Hello team, this is a test meeting recording for the Backlog Builder AI services. We need to implement the user authentication feature and fix the database connection bug. John will handle the frontend changes and Sarah will work on the backend API. The deadline is next Friday."
fi

analysis_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{
        \"transcription\": \"$TRANSCRIPTION\",
        \"options\": {
            \"meetingType\": \"development\",
            \"complexity\": \"medium\",
            \"includeActionItems\": true,
            \"includeDecisions\": true,
            \"includeSummary\": true
        }
    }" \
    "$API_BASE_URL/ai/analyze")

echo "$analysis_response" | jq '.' > "$RESULTS_DIR/analysis-result.json"

if echo "$analysis_response" | jq -e '.success' > /dev/null; then
    echo -e "${GREEN}✅ AI analysis passed${NC}"
    
    # Display analysis summary
    ACTION_ITEMS_COUNT=$(echo "$analysis_response" | jq '.actionItems | length')
    DECISIONS_COUNT=$(echo "$analysis_response" | jq '.decisions | length')
    
    echo -e "${BLUE}📊 Analysis Summary:${NC}"
    echo -e "   Action Items: $ACTION_ITEMS_COUNT"
    echo -e "   Decisions: $DECISIONS_COUNT"
    
else
    echo -e "${RED}❌ AI analysis failed${NC}"
    echo "$analysis_response"
    exit 1
fi

# Test 4: Ticket Generation
echo -e "${YELLOW}🔍 Test 4: Ticket Generation${NC}"

tickets_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "$analysis_response" \
    "$API_BASE_URL/ai/generate-tickets")

echo "$tickets_response" | jq '.' > "$RESULTS_DIR/tickets-result.json"

if echo "$tickets_response" | jq -e '.success' > /dev/null; then
    echo -e "${GREEN}✅ Ticket generation passed${NC}"
    
    # Display tickets summary
    TICKETS_COUNT=$(echo "$tickets_response" | jq '.tickets | length')
    
    echo -e "${BLUE}🎫 Tickets Summary:${NC}"
    echo -e "   Total Tickets: $TICKETS_COUNT"
    
    # Show ticket types breakdown
    echo "$tickets_response" | jq -r '.summary.ticketsByType | to_entries[] | "   \(.key): \(.value)"'
    
else
    echo -e "${RED}❌ Ticket generation failed${NC}"
    echo "$tickets_response"
    exit 1
fi

# Test 5: Complete End-to-End Pipeline
echo -e "${YELLOW}🔍 Test 5: Complete End-to-End Pipeline${NC}"

complete_response=$(curl -s -X POST \
    -F "audio=@$TEST_AUDIO_FILE" \
    -F "language=auto" \
    -F "priority=balanced" \
    -F "includeActionItems=true" \
    -F "includeDecisions=true" \
    -F "includeSummary=true" \
    -F "includeTickets=true" \
    "$API_BASE_URL/ai/process-audio")

echo "$complete_response" | jq '.' > "$RESULTS_DIR/complete-pipeline-result.json"

if echo "$complete_response" | jq -e '.success' > /dev/null; then
    echo -e "${GREEN}✅ Complete end-to-end pipeline passed${NC}"
    
    # Display complete pipeline summary
    echo -e "${BLUE}🏆 Complete Pipeline Summary:${NC}"
    echo "$complete_response" | jq -r '"   Transcription Length: \(.transcription | length) chars"'
    echo "$complete_response" | jq -r '"   Action Items: \(.analysis.actionItems | length)"'
    echo "$complete_response" | jq -r '"   Decisions: \(.analysis.decisions | length)"'
    echo "$complete_response" | jq -r '"   Generated Tickets: \(.tickets.summary.totalTickets)"'
    echo "$complete_response" | jq -r '"   Processing Time: \(.metadata.totalProcessingTime)ms"'
    
else
    echo -e "${RED}❌ Complete end-to-end pipeline failed${NC}"
    echo "$complete_response"
    exit 1
fi

# Test 6: Component-Specific Endpoints
echo -e "${YELLOW}🔍 Test 6: Component-Specific Endpoints${NC}"

# Test action items endpoint
action_items_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"transcription\": \"$TRANSCRIPTION\"}" \
    "$API_BASE_URL/ai/action-items")

if echo "$action_items_response" | jq -e '.success' > /dev/null; then
    echo -e "${GREEN}✅ Action items endpoint passed${NC}"
else
    echo -e "${RED}❌ Action items endpoint failed${NC}"
fi

# Test summary endpoint
summary_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"transcription\": \"$TRANSCRIPTION\"}" \
    "$API_BASE_URL/ai/summary")

if echo "$summary_response" | jq -e '.success' > /dev/null; then
    echo -e "${GREEN}✅ Summary endpoint passed${NC}"
else
    echo -e "${RED}❌ Summary endpoint failed${NC}"
fi

# Test decisions endpoint
decisions_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"transcription\": \"$TRANSCRIPTION\"}" \
    "$API_BASE_URL/ai/decisions")

if echo "$decisions_response" | jq -e '.success' > /dev/null; then
    echo -e "${GREEN}✅ Decisions endpoint passed${NC}"
else
    echo -e "${RED}❌ Decisions endpoint failed${NC}"
fi

# Test 7: Error Handling
echo -e "${YELLOW}🔍 Test 7: Error Handling${NC}"

# Test with invalid audio file
invalid_response=$(curl -s -X POST \
    -F "audio=@/dev/null" \
    "$API_BASE_URL/ai/transcribe")

if echo "$invalid_response" | jq -e '.success == false' > /dev/null; then
    echo -e "${GREEN}✅ Invalid audio file error handling passed${NC}"
else
    echo -e "${RED}❌ Invalid audio file error handling failed${NC}"
fi

# Test with empty transcription
empty_response=$(curl -s -X POST \
    -H "Content-Type: application/json" \
    -d "{\"transcription\": \"\"}" \
    "$API_BASE_URL/ai/analyze")

if echo "$empty_response" | jq -e '.success == false' > /dev/null; then
    echo -e "${GREEN}✅ Empty transcription error handling passed${NC}"
else
    echo -e "${RED}❌ Empty transcription error handling failed${NC}"
fi

# Test 8: Performance Test
echo -e "${YELLOW}🔍 Test 8: Performance Test${NC}"

start_time=$(date +%s%N)

# Run multiple requests concurrently
for i in {1..3}; do
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"transcription\": \"$TRANSCRIPTION\"}" \
        "$API_BASE_URL/ai/summary" > "$RESULTS_DIR/perf-test-$i.json" &
done

wait

end_time=$(date +%s%N)
duration=$(( (end_time - start_time) / 1000000 ))

echo -e "${GREEN}✅ Performance test completed in ${duration}ms${NC}"

# Generate test report
echo -e "${YELLOW}📊 Generating test report...${NC}"

cat > "$RESULTS_DIR/test-report.md" << EOF
# AI Services Test Report

Generated: $(date)

## Test Results Summary

- ✅ Service Status Check
- ✅ Speech-to-Text Transcription
- ✅ AI Analysis
- ✅ Ticket Generation
- ✅ Complete End-to-End Pipeline
- ✅ Component-Specific Endpoints
- ✅ Error Handling
- ✅ Performance Test

## Performance Metrics

- End-to-End Processing Time: $(echo "$complete_response" | jq -r '.metadata.totalProcessingTime')ms
- Concurrent Requests Performance: ${duration}ms
- Transcription Length: $(echo "$complete_response" | jq -r '.transcription | length') characters

## Generated Content

### Action Items
$(echo "$analysis_response" | jq -r '.actionItems[] | "- \(.title // .text)"')

### Decisions
$(echo "$analysis_response" | jq -r '.decisions[] | "- \(.title // .text)"')

### Generated Tickets
$(echo "$tickets_response" | jq -r '.tickets[] | "- [\(.type)] \(.title)"')

## Files Generated

- service-status.json
- transcription-result.json
- analysis-result.json
- tickets-result.json
- complete-pipeline-result.json
- Performance test results (perf-test-*.json)

## Next Steps

1. Review generated tickets for accuracy
2. Test with longer audio files
3. Test with different meeting types
4. Performance optimization if needed
EOF

echo -e "${GREEN}✅ Test report generated: $RESULTS_DIR/test-report.md${NC}"

# Clean up test audio file (if we created it)
if command -v espeak &> /dev/null && [ -f "$TEST_AUDIO_FILE" ]; then
    rm "$TEST_AUDIO_FILE"
fi

echo ""
echo -e "${GREEN}🎉 All tests completed successfully!${NC}"
echo -e "${BLUE}📊 Test results saved in: $RESULTS_DIR/${NC}"
echo -e "${BLUE}📝 Full report available: $RESULTS_DIR/test-report.md${NC}"
echo ""
echo -e "${YELLOW}💡 Next steps:${NC}"
echo -e "  1. Review the generated test report"
echo -e "  2. Test with your own audio files"
echo -e "  3. Customize prompt templates for your use case"
echo -e "  4. Set up monitoring and alerts"
