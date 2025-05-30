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
