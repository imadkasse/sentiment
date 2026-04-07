"""
Test script for Sentiment Analysis API
Run this AFTER starting the Flask server with: python app.py
"""

import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    print("=" * 50)
    print("TEST 1: Health Check")
    print("=" * 50)
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_home():
    print("\n" + "=" * 50)
    print("TEST 2: Home Endpoint")
    print("=" * 50)
    try:
        response = requests.get(BASE_URL)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_demo(text, expected_sentiment):
    print("\n" + "-" * 50)
    print(f"Testing: '{text}'")
    print(f"Expected: {expected_sentiment}")
    print("-" * 50)
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"text": text, "model": "demo"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        return result.get('sentiment') == expected_sentiment
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_error_handling():
    print("\n" + "=" * 50)
    print("TEST: Error Handling")
    print("=" * 50)
    
    # Test 1: Empty text
    try:
        response = requests.post(f"{BASE_URL}/predict", json={"text": ""})
        print(f"Empty text - Status: {response.status_code}, Error: {response.json().get('error')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Missing text field
    try:
        response = requests.post(f"{BASE_URL}/predict", json={})
        print(f"Missing text - Status: {response.status_code}, Error: {response.json().get('error')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Invalid model
    try:
        response = requests.post(f"{BASE_URL}/predict", json={"text": "good movie", "model": "invalid"})
        print(f"Invalid model - Status: {response.status_code}, Error: {response.json().get('error')}")
    except Exception as e:
        print(f"Error: {e}")

def run_all_tests():
    print("\n" + "=" * 60)
    print("?? SENTIMENT ANALYSIS API TEST SUITE")
    print("=" * 60)
    print("\nMake sure the server is running: python app.py")
    print("-" * 60)
    
    # Basic endpoint tests
    health_ok = test_health()
    home_ok = test_home()
    
    # Sentiment prediction tests
    print("\n" + "=" * 50)
    print("TEST 3-6: Sentiment Predictions (Demo Mode)")
    print("=" * 50)
    
    test_cases = [
        ("This movie was absolutely amazing and wonderful!", "positive"),
        ("I love this film, it's the best thing ever!", "positive"),
        ("This was the worst movie I have ever seen. Terrible and boring!", "negative"),
        ("What a waste of time. Awful and disappointing.", "negative"),
        ("The movie was okay, nothing special.", "neutral"),
    ]
    
    results = []
    for text, expected in test_cases:
        result = test_predict_demo(text, expected)
        results.append(result)
    
    # Error handling
    test_error_handling()
    
    # Summary
    print("\n" + "=" * 60)
    print("?? TEST SUMMARY")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total} sentiment tests")
    print(f"Health endpoint: {'?' if health_ok else '?'}")
    print(f"Home endpoint: {'?' if home_ok else '?'}")
    
    if passed == total and health_ok and home_ok:
        print("\n?? ALL TESTS PASSED!")
    else:
        print("\n?? Some tests failed. Check server logs.")
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
