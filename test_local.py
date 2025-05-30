#!/usr/bin/env python3
"""
Simplified Local Testing for Backlog Builder LLM Service
This script allows you to test the LLM service directly without Docker
"""

import os
import sys
import subprocess
import time
import threading
import requests
import json

# Add the project directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, 'ai-services', 'llm-processing'))

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'transformers', 
        'sentence-transformers', 'pydantic', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print("✅ Dependencies installed")
    else:
        print("✅ All dependencies satisfied")

def start_llm_service():
    """Start the LLM service in a separate process"""
    print("🚀 Starting LLM service...")
    
    # Set environment variables
    os.environ['LLM_PORT'] = '8002'
    os.environ['REDIS_URL'] = 'redis://localhost:6379'
    
    # Change to the LLM service directory
    llm_dir = os.path.join(project_root, 'ai-services', 'llm-processing')
    
    # Start the service
    cmd = [sys.executable, 'huggingface-local.py']
    process = subprocess.Popen(
        cmd, 
        cwd=llm_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def wait_for_service(url, timeout=60):
    """Wait for the service to be ready"""
    print(f"⏳ Waiting for service at {url} to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Service is ready!")
                return True
        except:
            pass
        time.sleep(2)
        print("   Still waiting...")
    
    print("❌ Service did not start within timeout")
    return False

def test_llm_service():
    """Test the LLM service endpoints"""
    base_url = "http://localhost:8002"
    
    print("\n🧪 Testing LLM Service...")
    
    # Test data
    test_transcript = """
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
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Health Check
    try:
        print("\n1️⃣ Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Health check passed")
            print(f"      Status: {data.get('status')}")
            print(f"      GPU Available: {data.get('gpu_available')}")
            print(f"      Device: {data.get('device')}")
            tests_passed += 1
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
    
    # Test 2: Meeting Analysis
    try:
        print("\n2️⃣ Testing meeting analysis...")
        payload = {
            "transcript": test_transcript,
            "meeting_type": "sprint_planning",
            "extract_action_items": True,
            "extract_decisions": True,
            "generate_summary": True,
            "detect_sentiment": True
        }
        
        response = requests.post(f"{base_url}/analyze", json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Meeting analysis successful")
            print(f"      Processing time: {data.get('processing_time', 0):.2f} seconds")
            print(f"      Action items found: {len(data.get('action_items', []))}")
            print(f"      Decisions found: {len(data.get('decisions', []))}")
            
            # Print some results
            if data.get('action_items'):
                print("      Sample action item:")
                action_item = data['action_items'][0]
                print(f"        Task: {action_item.get('task', 'N/A')}")
                print(f"        Assignee: {action_item.get('assignee', 'N/A')}")
                print(f"        Priority: {action_item.get('priority', 'N/A')}")
            
            tests_passed += 1
        else:
            print(f"   ❌ Meeting analysis failed: {response.status_code}")
            print(f"      Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Meeting analysis failed: {e}")
    
    # Test 3: Ticket Generation
    try:
        print("\n3️⃣ Testing ticket generation...")
        payload = {
            "action_item": "Implement OAuth2 authentication for better security",
            "context": "Sprint planning meeting - user authentication feature discussion",
            "ticket_type": "feature"
        }
        
        response = requests.post(f"{base_url}/generate-ticket", json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Ticket generation successful")
            print(f"      Title: {data.get('title', 'N/A')}")
            print(f"      Priority: {data.get('priority', 'N/A')}")
            print(f"      Labels: {', '.join(data.get('labels', []))}")
            print(f"      Acceptance Criteria: {len(data.get('acceptance_criteria', []))} items")
            tests_passed += 1
        else:
            print(f"   ❌ Ticket generation failed: {response.status_code}")
            print(f"      Response: {response.text}")
    except Exception as e:
        print(f"   ❌ Ticket generation failed: {e}")
    
    print(f"\n🏁 Test Results: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests

def main():
    """Main testing function"""
    print("🎯 Backlog Builder - Simplified Local Testing")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Start the service
    service_process = start_llm_service()
    
    try:
        # Wait for service to be ready
        if wait_for_service("http://localhost:8002"):
            # Run tests
            success = test_llm_service()
            
            if success:
                print("\n🎉 All tests passed! The LLM service is working correctly.")
                
                print("\n📖 API Documentation:")
                print("   Swagger UI: http://localhost:8002/docs")
                print("   ReDoc: http://localhost:8002/redoc")
                
                print("\n🔧 You can now:")
                print("   1. Test the API endpoints using the Swagger UI")
                print("   2. Integrate with your frontend application")
                print("   3. Start the backend service to test the full pipeline")
                
                # Keep service running
                input("\n⏸️  Press Enter to stop the service...")
            else:
                print("\n⚠️  Some tests failed. Check the error messages above.")
        else:
            print("\n❌ Service failed to start properly.")
    
    finally:
        # Cleanup
        print("\n🛑 Stopping service...")
        service_process.terminate()
        service_process.wait()
        print("✅ Cleanup completed")

if __name__ == "__main__":
    main()
