"""
Comprehensive API Testing Suite
Unit tests, integration tests, and load tests for AI APIs
"""
import unittest
import asyncio
import json
import time
from typing import Dict, Any
from datetime import datetime

import pytest
import requests
from fastapi.testclient import TestClient
import httpx

# Import the FastAPI app
# For testing, we'll create a simplified version
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Create test app
app = FastAPI()

class TestMessage(BaseModel):
    message: str

@app.post("/test/echo")
async def echo_endpoint(message: TestMessage):
    """Simple echo endpoint for testing"""
    return {"echo": message.message, "timestamp": datetime.now().isoformat()}

@app.get("/test/error/{error_code}")
async def error_test(error_code: int):
    """Generate errors for testing"""
    if error_code == 400:
        raise HTTPException(status_code=400, detail="Bad Request")
    elif error_code == 404:
        raise HTTPException(status_code=404, detail="Not Found")
    elif error_code == 500:
        raise HTTPException(status_code=500, detail="Internal Server Error")
    return {"status": "ok"}

# Unit Tests
class TestAPIEndpoints(unittest.TestCase):
    """Unit tests for API endpoints"""
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_echo_endpoint(self):
        """Test echo endpoint with valid input"""
        response = self.client.post("/test/echo", json={"message": "Hello, World!"})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("echo", data)
        self.assertEqual(data["echo"], "Hello, World!")
        self.assertIn("timestamp", data)
    
    def test_echo_endpoint_empty_message(self):
        """Test echo endpoint with empty message"""
        response = self.client.post("/test/echo", json={"message": ""})
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["echo"], "")
    
    def test_error_endpoints(self):
        """Test error generation endpoints"""
        # Test 400 error
        response = self.client.get("/test/error/400")
        self.assertEqual(response.status_code, 400)
        
        # Test 404 error
        response = self.client.get("/test/error/404")
        self.assertEqual(response.status_code, 404)
        
        # Test 500 error
        response = self.client.get("/test/error/500")
        self.assertEqual(response.status_code, 500)
        
        # Test valid response
        response = self.client.get("/test/error/200")
        self.assertEqual(response.status_code, 200)

# Integration Tests
class IntegrationTests:
    """Integration tests for external API calls"""
    
    BASE_URL = "http://localhost:8000"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def test_api_health(self):
        """Test API health endpoint"""
        try:
            response = await self.client.get(f"{self.BASE_URL}/health")
            return response.status_code == 200
        except:
            return False
    
    async def test_chat_endpoint(self):
        """Test chat endpoint with sample message"""
        payload = {
            "messages": [
                {"role": "user", "content": "Hello, AI!"}
            ],
            "model": "gpt-3.5-turbo"
        }
        
        try:
            response = await self.client.post(
                f"{self.BASE_URL}/chat",
                json=payload,
                headers={"Authorization": "Bearer demo-token"}
            )
            return response.status_code in [200, 201]
        except:
            return False
    
    async def test_streaming_endpoint(self):
        """Test streaming endpoint"""
        try:
            async with self.client.stream(
                "POST",
                f"{self.BASE_URL}/chat/stream",
                json={
                    "messages": [{"role": "user", "content": "Test stream"}],
                    "stream": True
                },
                headers={"Authorization": "Bearer demo-token"}
            ) as response:
                return response.status_code == 200
        except:
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("🧪 Running Integration Tests...")
        
        tests = [
            ("API Health Check", self.test_api_health),
            ("Chat Endpoint", self.test_chat_endpoint),
            ("Streaming Endpoint", self.test_streaming_endpoint)
        ]
        
        results = {}
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results[test_name] = "✅ PASS" if result else "❌ FAIL"
                print(f"  {test_name}: {results[test_name]}")
            except Exception as e:
                results[test_name] = f"❌ ERROR: {str(e)}"
                print(f"  {test_name}: {results[test_name]}")
        
        return results

# Load Testing
class LoadTester:
    """Simple load testing utility"""
    
    def __init__(self, base_url: str, num_requests: int = 100):
        self.base_url = base_url
        self.num_requests = num_requests
        self.results = {
            "success": 0,
            "failures": 0,
            "response_times": [],
            "errors": []
        }
    
    def test_endpoint(self, endpoint: str, method: str = "GET", payload: Dict = None):
        """Test a single endpoint under load"""
        import concurrent.futures
        import threading
        
        print(f"⚡ Load Testing {endpoint} with {self.num_requests} requests...")
        
        start_time = time.time()
        
        def make_request(i):
            try:
                request_start = time.time()
                
                if method == "GET":
                    response = requests.get(f"{self.base_url}{endpoint}")
                elif method == "POST":
                    response = requests.post(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                        headers={"Authorization": "Bearer demo-token"}
                    )
                
                request_time = time.time() - request_start
                
                if 200 <= response.status_code < 300:
                    self.results["success"] += 1
                    self.results["response_times"].append(request_time)
                else:
                    self.results["failures"] += 1
                    self.results["errors"].append(f"Request {i}: {response.status_code}")
                    
            except Exception as e:
                self.results["failures"] += 1
                self.results["errors"].append(f"Request {i}: {str(e)}")
        
        # Use thread pool for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            executor.map(make_request, range(self.num_requests))
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        if self.results["response_times"]:
            avg_time = sum(self.results["response_times"]) / len(self.results["response_times"])
            min_time = min(self.results["response_times"])
            max_time = max(self.results["response_times"])
        else:
            avg_time = min_time = max_time = 0
        
        requests_per_second = self.num_requests / total_time if total_time > 0 else 0
        
        return {
            "endpoint": endpoint,
            "total_requests": self.num_requests,
            "success": self.results["success"],
            "failures": self.results["failures"],
            "success_rate": (self.results["success"] / self.num_requests) * 100,
            "total_time": round(total_time, 2),
            "requests_per_second": round(requests_per_second, 2),
            "avg_response_time": round(avg_time * 1000, 2),  # ms
            "min_response_time": round(min_time * 1000, 2),
            "max_response_time": round(max_time * 1000, 2)
        }

# Security Tests
class SecurityTests:
    """Security testing utilities"""
    
    @staticmethod
    def test_sql_injection(endpoint: str, base_url: str = "http://localhost:8000"):
        """Test for SQL injection vulnerabilities"""
        print(f"🛡️  Testing SQL Injection on {endpoint}")
        
        payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "' UNION SELECT NULL, NULL --",
            "1' AND '1'='1"
        ]
        
        results = []
        for payload in payloads:
            try:
                response = requests.get(f"{base_url}{endpoint}?input={payload}")
                results.append({
                    "payload": payload,
                    "status_code": response.status_code,
                    "body_length": len(response.text)
                })
            except Exception as e:
                results.append({
                    "payload": payload,
                    "error": str(e)
                })
        
        return results
    
    @staticmethod
    def test_xss(endpoint: str, base_url: str = "http://localhost:8000"):
        """Test for XSS vulnerabilities"""
        print(f"🛡️  Testing XSS on {endpoint}")
        
        payloads = [
            "<script>alert('XSS')</script>",
            "<img src='x' onerror='alert(1)'>",
            "javascript:alert('XSS')"
        ]
        
        results = []
        for payload in payloads:
            try:
                response = requests.post(
                    f"{base_url}{endpoint}",
                    json={"message": payload}
                )
                results.append({
                    "payload": payload,
                    "status_code": response.status_code
                })
            except Exception as e:
                results.append({
                    "payload": payload,
                    "error": str(e)
                })
        
        return results

# Performance Monitoring
class PerformanceMonitor:
    """Monitor API performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "error_rates": [],
            "throughput": []
        }
    
    def record_response_time(self, endpoint: str, response_time: float):
        """Record response time for an endpoint"""
        self.metrics["response_times"].append({
            "endpoint": endpoint,
            "time": response_time,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 1000 records
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"] = self.metrics["response_times"][-1000:]
    
    def record_error(self, endpoint: str, error: str):
        """Record error for an endpoint"""
        self.metrics["error_rates"].append({
            "endpoint": endpoint,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_summary(self):
        """Get performance summary"""
        if not self.metrics["response_times"]:
            return {"message": "No data collected yet"}
        
        # Calculate statistics
        times = [r["time"] for r in self.metrics["response_times"]]
        
        return {
            "total_requests": len(self.metrics["response_times"]),
            "total_errors": len(self.metrics["error_rates"]),
            "error_rate": len(self.metrics["error_rates"]) / len(self.metrics["response_times"]) * 100 if self.metrics["response_times"] else 0,
            "avg_response_time": sum(times) / len(times),
            "p95_response_time": sorted(times)[int(len(times) * 0.95)] if times else 0,
            "p99_response_time": sorted(times)[int(len(times) * 0.99)] if times else 0,
            "monitoring_period": {
                "start": self.metrics["response_times"][0]["timestamp"] if self.metrics["response_times"] else None,
                "end": self.metrics["response_times"][-1]["timestamp"] if self.metrics["response_times"] else None
            }
        }

# Test Runner
def run_tests():
    """Run all tests"""
    print("=" * 60)
    print("🧪 COMPREHENSIVE API TEST SUITE")
    print("=" * 60)
    
    # Run unit tests
    print("\n1. Running Unit Tests...")
    unittest_loader = unittest.TestLoader()
    suite = unittest_loader.loadTestsFromTestCase(TestAPIEndpoints)
    runner = unittest.TextTestRunner(verbosity=2)
    unit_results = runner.run(suite)
    
    # Run integration tests
    print("\n2. Running Integration Tests...")
    async def run_integration():
        tester = IntegrationTests()
        return await tester.run_all_tests()
    
    integration_results = asyncio.run(run_integration())
    
    # Run load tests
    print("\n3. Running Load Tests...")
    load_tester = LoadTester("http://localhost:8000", num_requests=50)
    load_results = load_tester.test_endpoint("/health")
    
    print("\n📊 Load Test Results:")
    for key, value in load_results.items():
        print(f"  {key}: {value}")
    
    # Run security tests
    print("\n4. Running Security Tests...")
    security_tester = SecurityTests()
    sql_results = security_tester.test_sql_injection("/test/echo")
    xss_results = security_tester.test_xss("/test/echo")
    
    print(f"  SQL Injection tests: {len(sql_results)} payloads tested")
    print(f"  XSS tests: {len(xss_results)} payloads tested")
    
    # Performance monitoring
    print("\n5. Performance Monitoring Setup...")
    monitor = PerformanceMonitor()
    
    # Simulate some requests
    for i in range(10):
        start = time.time()
        # Simulate request
        time.sleep(0.1)
        response_time = time.time() - start
        monitor.record_response_time("/test/echo", response_time)
    
    perf_summary = monitor.get_summary()
    print("  Performance Summary:")
    for key, value in perf_summary.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ Test Suite Complete!")
    print("=" * 60)
    
    return {
        "unit_tests": unit_results.wasSuccessful() if hasattr(unit_results, 'wasSuccessful') else "N/A",
        "integration_tests": integration_results,
        "load_test": load_results,
        "security_tests": {
            "sql_injection": len(sql_results),
            "xss": len(xss_results)
        },
        "performance_summary": perf_summary
    }

# API Mock Server for Testing
class MockAPIServer:
    """Mock API server for testing without external dependencies"""
    
    def __init__(self):
        self.app = FastAPI()
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.get("/mock/health")
        async def health():
            return {"status": "healthy", "service": "mock-api"}
        
        @self.app.post("/mock/chat")
        async def chat():
            return {
                "response": "This is a mock AI response",
                "model": "mock-gpt",
                "usage": {"total_tokens": 42}
            }
        
        @self.app.get("/mock/stream")
        async def stream():
            async def generate():
                for i in range(5):
                    yield f"data: Mock chunk {i}\n\n"
                    await asyncio.sleep(0.1)
            
            from fastapi.responses import StreamingResponse
            return StreamingResponse(generate(), media_type="text/event-stream")
    
    def run(self, port: int = 8888):
        """Run the mock server"""
        import uvicorn
        print(f"🚀 Mock API server running on http://localhost:{port}")
        uvicorn.run(self.app, host="0.0.0.0", port=port)

# Test Configuration
class TestConfig:
    """Test configuration and utilities"""
    
    @staticmethod
    def generate_test_data(num_samples: int = 100):
        """Generate test data for load testing"""
        import random
        import string
        
        test_data = []
        for i in range(num_samples):
            message = ''.join(random.choices(string.ascii_letters + ' ', k=random.randint(10, 100)))
            test_data.append({
                "messages": [
                    {"role": "user", "content": f"Test message {i}: {message}"}
                ],
                "model": random.choice(["gpt-3.5-turbo", "gpt-4"]),
                "temperature": random.uniform(0.1, 1.0)
            })
        
        return test_data
    
    @staticmethod
    def save_test_results(results: dict, filename: str = "test_results.json"):
        """Save test results to file"""
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Test results saved to {filename}")

# Main execution
if __name__ == "__main__":
    print("API Testing Suite")
    print("=" * 50)
    
    # Check if we should run tests or start mock server
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "mock":
        # Start mock server
        server = MockAPIServer()
        server.run()
    elif len(sys.argv) > 1 and sys.argv[1] == "load":
        # Run load test only
        print("Running load test...")
        load_tester = LoadTester("http://localhost:8000", num_requests=100)
        results = load_tester.test_endpoint("/health")
        print(json.dumps(results, indent=2))
    else:
        # Run comprehensive test suite
        results = run_tests()
        
        # Save results
        TestConfig.save_test_results(results)
        
        # Generate report
        print("\n📋 TEST REPORT SUMMARY")
        print("-" * 30)
        print(f"Unit Tests: {'✅ PASS' if results['unit_tests'] else '❌ FAIL'}")
        print("Integration Tests:")
        for test_name, result in results["integration_tests"].items():
            print(f"  {test_name}: {result}")
        
        load_test = results["load_test"]
        print(f"\nLoad Test ({load_test['total_requests']} requests):")
        print(f"  Success Rate: {load_test['success_rate']:.1f}%")
        print(f"  Avg Response Time: {load_test['avg_response_time']}ms")
        print(f"  Requests/sec: {load_test['requests_per_second']}")