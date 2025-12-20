"""
API Error Handler - Days 9-10
Advanced error handling for OpenAI API and external APIs
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from datetime import datetime, timedelta
from enum import Enum
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_errors.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class APIErrorMetrics:
    """Track API error metrics"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.last_error_time: Dict[str, datetime] = {}
        self.total_requests = 0
        self.failed_requests = 0
        
    def record_error(self, error_type: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Record an error occurrence"""
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_error_time[error_type] = datetime.now()
        self.failed_requests += 1
        
        logger.warning(f"API Error: {error_type} (Severity: {severity.value})")
        
        # Log critical errors immediately
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error_type}")
    
    def record_success(self):
        """Record a successful request"""
        self.total_requests += 1
    
    def get_error_rate(self) -> float:
        """Calculate current error rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    def get_summary(self) -> Dict[str, Any]:
        """Get error metrics summary"""
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate": 100 - self.get_error_rate(),
            "error_rate": self.get_error_rate(),
            "error_counts": self.error_counts,
            "last_errors": {k: v.isoformat() for k, v in self.last_error_time.items()}
        }

class CircuitBreaker:
    """Circuit breaker pattern for API calls"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        """
        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds to wait before attempting to close circuit
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
                if time_since_failure > self.reset_timeout:
                    # Move to HALF_OPEN state
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                    return True
            return False
        
        if self.state == "HALF_OPEN":
            return True
        
        return False
    
    def record_success(self):
        """Record a successful call"""
        if self.state == "HALF_OPEN":
            # Close the circuit
            self.state = "CLOSED"
            self.failure_count = 0
            self.last_failure_time = None
            logger.info("Circuit breaker CLOSED - service recovered")
        else:
            # Reset failure count on success
            self.failure_count = 0
    
    def record_failure(self):
        """Record a failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != "OPEN":
                self.state = "OPEN"
                logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
        
        logger.debug(f"Failure recorded. Count: {self.failure_count}, State: {self.state}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "can_execute": self.can_execute()
        }

class APIErrorHandler:
    """Comprehensive API error handler"""
    
    def __init__(self, openai_client: Optional[OpenAI] = None):
        self.openai_client = openai_client or OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.metrics = APIErrorMetrics()
        self.circuit_breaker = CircuitBreaker()
        
        # Rate limiting configuration
        self.rate_limit_config = {
            "requests_per_minute": 60,
            "tokens_per_minute": 90000,
            "last_request_time": None,
            "request_count": 0,
            "token_count": 0
        }
    
    def check_rate_limit(self, estimated_tokens: int = 0) -> bool:
        """Check if rate limit would be exceeded"""
        now = datetime.now()
        
        # Reset counters if more than a minute has passed
        if (self.rate_limit_config["last_request_time"] and 
            (now - self.rate_limit_config["last_request_time"]).total_seconds() > 60):
            self.rate_limit_config["request_count"] = 0
            self.rate_limit_config["token_count"] = 0
        
        # Check limits
        if (self.rate_limit_config["request_count"] >= self.rate_limit_config["requests_per_minute"] or
            self.rate_limit_config["token_count"] + estimated_tokens > self.rate_limit_config["tokens_per_minute"]):
            return False
        
        return True
    
    def update_rate_limit(self, used_tokens: int = 0):
        """Update rate limit counters"""
        self.rate_limit_config["last_request_time"] = datetime.now()
        self.rate_limit_config["request_count"] += 1
        self.rate_limit_config["token_count"] += used_tokens
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def call_openai_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Call OpenAI API with retry logic"""
        
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is OPEN - service unavailable")
        
        # Check rate limit
        estimated_tokens = kwargs.get('max_tokens', 100)
        if not self.check_rate_limit(estimated_tokens):
            time_to_wait = 60 - (datetime.now() - self.rate_limit_config["last_request_time"]).total_seconds()
            raise RateLimitError(f"Rate limit exceeded. Wait {time_to_wait:.1f} seconds.")
        
        try:
            # Make the API call
            result = func(*args, **kwargs)
            
            # Record success
            self.metrics.record_success()
            self.circuit_breaker.record_success()
            self.update_rate_limit(estimated_tokens)
            
            return result
            
        except RateLimitError as e:
            self.metrics.record_error("rate_limit", ErrorSeverity.MEDIUM)
            self.circuit_breaker.record_failure()
            
            # Extract retry time from error if available
            retry_after = getattr(e, 'retry_after', 60)
            logger.warning(f"Rate limit hit. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            
            raise  # Let tenacity handle the retry
            
        except APITimeoutError as e:
            self.metrics.record_error("timeout", ErrorSeverity.MEDIUM)
            self.circuit_breaker.record_failure()
            logger.warning(f"API timeout: {e}")
            raise
            
        except APIConnectionError as e:
            self.metrics.record_error("connection", ErrorSeverity.HIGH)
            self.circuit_breaker.record_failure()
            logger.error(f"API connection error: {e}")
            raise
            
        except APIError as e:
            error_code = getattr(e, 'code', 'unknown')
            
            # Classify error severity
            if error_code == 'invalid_api_key':
                severity = ErrorSeverity.CRITICAL
            elif error_code in ['context_length_exceeded', 'invalid_request_error']:
                severity = ErrorSeverity.HIGH
            else:
                severity = ErrorSeverity.MEDIUM
            
            self.metrics.record_error(f"api_error_{error_code}", severity)
            self.circuit_breaker.record_failure()
            logger.error(f"API error ({error_code}): {e}")
            
            # Don't retry on certain errors
            if error_code in ['invalid_api_key', 'invalid_request_error']:
                raise
            
            # For other errors, retry
            raise
            
        except Exception as e:
            self.metrics.record_error("unknown", ErrorSeverity.HIGH)
            self.circuit_breaker.record_failure()
            logger.error(f"Unknown error: {e}")
            raise
    
    def call_external_api(self, url: str, method: str = "GET", 
                         headers: Dict[str, str] = None, 
                         data: Dict[str, Any] = None,
                         timeout: int = 10,
                         retries: int = 3) -> Dict[str, Any]:
        """Call external API with error handling"""
        
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is OPEN - service unavailable")
        
        for attempt in range(retries):
            try:
                logger.info(f"Calling external API: {url} (Attempt {attempt + 1}/{retries})")
                
                response = httpx.request(
                    method=method,
                    url=url,
                    headers=headers or {},
                    json=data,
                    timeout=timeout
                )
                
                response.raise_for_status()
                
                # Success
                self.metrics.record_success()
                self.circuit_breaker.record_success()
                
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "data": response.json() if response.content else None,
                    "headers": dict(response.headers)
                }
                
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                
                # Classify error
                if status_code == 429:  # Too Many Requests
                    self.metrics.record_error("external_rate_limit", ErrorSeverity.MEDIUM)
                    retry_after = int(e.response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                    time.sleep(retry_after)
                    
                elif status_code >= 500:  # Server errors
                    self.metrics.record_error(f"external_server_error_{status_code}", ErrorSeverity.MEDIUM)
                    self.circuit_breaker.record_failure()
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Server error {status_code}. Retrying in {wait_time}s.")
                    time.sleep(wait_time)
                    
                elif status_code >= 400:  # Client errors
                    self.metrics.record_error(f"external_client_error_{status_code}", ErrorSeverity.HIGH)
                    # Don't retry client errors (except 429)
                    return {
                        "success": False,
                        "error": f"Client error: {status_code}",
                        "details": str(e)
                    }
                    
            except httpx.TimeoutException:
                self.metrics.record_error("external_timeout", ErrorSeverity.MEDIUM)
                wait_time = 2 ** attempt
                logger.warning(f"Timeout. Retrying in {wait_time}s.")
                time.sleep(wait_time)
                
            except httpx.RequestError as e:
                self.metrics.record_error("external_connection", ErrorSeverity.HIGH)
                self.circuit_breaker.record_failure()
                logger.error(f"Connection error: {e}")
                return {
                    "success": False,
                    "error": "Connection failed",
                    "details": str(e)
                }
        
        # All retries failed
        self.metrics.record_error("external_all_retries_failed", ErrorSeverity.HIGH)
        self.circuit_breaker.record_failure()
        
        return {
            "success": False,
            "error": f"All {retries} attempts failed"
        }
    
    def safe_chat_completion(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Safe wrapper for chat completions with full error handling"""
        try:
            result = self.call_openai_with_retry(
                self.openai_client.chat.completions.create,
                messages=messages,
                **kwargs
            )
            
            return {
                "success": True,
                "data": result,
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens if result.usage else 0,
                    "completion_tokens": result.usage.completion_tokens if result.usage else 0,
                    "total_tokens": result.usage.total_tokens if result.usage else 0
                }
            }
            
        except Exception as e:
            error_type = type(e).__name__
            
            return {
                "success": False,
                "error": str(e),
                "error_type": error_type,
                "suggestion": self._get_error_suggestion(error_type, str(e))
            }
    
    def _get_error_suggestion(self, error_type: str, error_message: str) -> str:
        """Get user-friendly suggestions for common errors"""
        suggestions = {
            "RateLimitError": "You've hit the rate limit. Wait a minute or reduce request frequency.",
            "APITimeoutError": "The request timed out. Check your network connection or reduce request size.",
            "APIConnectionError": "Connection failed. Check your internet connection and API endpoint.",
            "AuthenticationError": "Invalid API key. Check your .env file and OpenAI account.",
            "InvalidRequestError": "Invalid request parameters. Check your input data.",
            "ContextLengthExceeded": "Input too long. Reduce the text length or use summarization.",
            "APIError": "OpenAI API error. Check the error message for details."
        }
        
        # Try to match error type
        for key in suggestions:
            if key.lower() in error_type.lower():
                return suggestions[key]
        
        # Default suggestion
        return "An unexpected error occurred. Check the logs for details."
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of API handler"""
        metrics_summary = self.metrics.get_summary()
        circuit_state = self.circuit_breaker.get_state()
        
        # Determine overall status
        if circuit_state["state"] == "OPEN":
            status = "UNHEALTHY"
        elif metrics_summary["error_rate"] > 10:  # More than 10% error rate
            status = "DEGRADED"
        else:
            status = "HEALTHY"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics_summary,
            "circuit_breaker": circuit_state,
            "rate_limit": {
                "requests_this_minute": self.rate_limit_config["request_count"],
                "tokens_this_minute": self.rate_limit_config["token_count"],
                "limits": {
                    "requests_per_minute": self.rate_limit_config["requests_per_minute"],
                    "tokens_per_minute": self.rate_limit_config["tokens_per_minute"]
                }
            }
        }
    
    def reset(self):
        """Reset all error tracking"""
        self.metrics = APIErrorMetrics()
        self.circuit_breaker = CircuitBreaker()
        self.rate_limit_config.update({
            "last_request_time": None,
            "request_count": 0,
            "token_count": 0
        })
        logger.info("Error handler reset")

def interactive_error_handler_demo():
    """Interactive demo of error handling features"""
    print(Fore.CYAN + "\n" + "="*70)
    print(Fore.CYAN + "🛡️  API Error Handler Demo - Days 9-10")
    print(Fore.CYAN + "="*70)
    
    try:
        handler = APIErrorHandler()
        
        while True:
            print(Fore.YELLOW + "\n" + "━" * 50)
            print(Fore.YELLOW + "🛠️  Error Handler Options:")
            print(Fore.YELLOW + "1. Test Safe Chat Completion")
            print(Fore.YELLOW + "2. Test External API Call")
            print(Fore.YELLOW + "3. View Health Status")
            print(Fore.YELLOW + "4. View Error Metrics")
            print(Fore.YELLOW + "5. Simulate Errors")
            print(Fore.YELLOW + "6. Reset Handler")
            print(Fore.YELLOW + "7. Exit")
            print(Fore.YELLOW + "━" * 50)
            
            choice = input(Fore.WHITE + "\nSelect option (1-7): ").strip()
            
            if choice == "7":
                print(Fore.YELLOW + "👋 Goodbye!")
                break
            
            if choice == "1":
                print(Fore.CYAN + "\n🤖 Testing safe chat completion...")
                
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello and tell me a fun fact!"}
                ]
                
                result = handler.safe_chat_completion(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    max_tokens=100
                )
                
                if result["success"]:
                    print(Fore.GREEN + "✅ Success!")
                    print(Fore.WHITE + f"Response: {result['data'].choices[0].message.content}")
                    if "usage" in result:
                        print(Fore.CYAN + f"Tokens used: {result['usage']['total_tokens']}")
                else:
                    print(Fore.RED + f"❌ Failed: {result['error']}")
                    print(Fore.YELLOW + f"Suggestion: {result['suggestion']}")
            
            elif choice == "2":
                print(Fore.CYAN + "\n🌐 Testing external API call...")
                
                # Test with a public API
                url = "https://jsonplaceholder.typicode.com/posts/1"
                
                result = handler.call_external_api(url)
                
                if result["success"]:
                    print(Fore.GREEN + "✅ API call successful!")
                    print(Fore.CYAN + f"Status: {result['status_code']}")
                    print(Fore.WHITE + f"Data: {json.dumps(result['data'], indent=2)[:200]}...")
                else:
                    print(Fore.RED + f"❌ API call failed: {result['error']}")
            
            elif choice == "3":
                print(Fore.CYAN + "\n🏥 Health Status Check...")
                
                health = handler.get_health_status()
                
                print(Fore.YELLOW + "\n📊 Overall Status:")
                status_color = Fore.GREEN if health["status"] == "HEALTHY" else Fore.RED
                print(status_color + f"  Status: {health['status']}")
                print(Fore.CYAN + f"  Timestamp: {health['timestamp']}")
                
                print(Fore.YELLOW + "\n⚡ Rate Limit Status:")
                rate_config = health["rate_limit"]
                print(Fore.CYAN + f"  Requests this minute: {rate_config['requests_this_minute']}/{rate_config['limits']['requests_per_minute']}")
                print(Fore.CYAN + f"  Tokens this minute: {rate_config['tokens_this_minute']}/{rate_config['limits']['tokens_per_minute']}")
                
                print(Fore.YELLOW + "\n🔌 Circuit Breaker:")
                cb = health["circuit_breaker"]
                state_color = Fore.GREEN if cb["state"] == "CLOSED" else Fore.RED
                print(state_color + f"  State: {cb['state']}")
                print(Fore.CYAN + f"  Can execute: {cb['can_execute']}")
                print(Fore.CYAN + f"  Failure count: {cb['failure_count']}")
            
            elif choice == "4":
                print(Fore.CYAN + "\n📈 Error Metrics...")
                
                metrics = handler.metrics.get_summary()
                
                print(Fore.YELLOW + "\n📊 Request Statistics:")
                print(Fore.CYAN + f"  Total requests: {metrics['total_requests']}")
                print(Fore.CYAN + f"  Failed requests: {metrics['failed_requests']}")
                print(Fore.CYAN + f"  Success rate: {metrics['success_rate']:.1f}%")
                print(Fore.CYAN + f"  Error rate: {metrics['error_rate']:.1f}%")
                
                if metrics['error_counts']:
                    print(Fore.YELLOW + "\n🚨 Error Breakdown:")
                    for error_type, count in metrics['error_counts'].items():
                        print(Fore.RED + f"  {error_type}: {count}")
                
                if metrics['last_errors']:
                    print(Fore.YELLOW + "\n⏰ Last Error Times:")
                    for error_type, timestamp in metrics['last_errors'].items():
                        print(Fore.CYAN + f"  {error_type}: {timestamp}")
            
            elif choice == "5":
                print(Fore.CYAN + "\n🧪 Simulating errors...")
                print(Fore.YELLOW + "1. Record random error")
                print(Fore.YELLOW + "2. Trigger circuit breaker")
                print(Fore.YELLOW + "3. Back to main menu")
                
                sim_choice = input(Fore.WHITE + "\nSelect simulation (1-3): ").strip()
                
                if sim_choice == "1":
                    handler.metrics.record_error("simulated_error", ErrorSeverity.LOW)
                    print(Fore.GREEN + "✅ Simulated error recorded")
                
                elif sim_choice == "2":
                    for i in range(6):  # Exceed failure threshold
                        handler.circuit_breaker.record_failure()
                    print(Fore.YELLOW + "⚠️ Circuit breaker should now be OPEN")
                    
                    # Check state
                    state = handler.circuit_breaker.get_state()
                    print(Fore.CYAN + f"State: {state['state']}")
                    print(Fore.CYAN + f"Can execute: {state['can_execute']}")
            
            elif choice == "6":
                print(Fore.CYAN + "\n🔄 Resetting error handler...")
                handler.reset()
                print(Fore.GREEN + "✅ Handler reset complete")
            
            else:
                print(Fore.RED + "❌ Invalid choice")
            
            input(Fore.YELLOW + "\nPress Enter to continue...")
    
    except Exception as e:
        print(Fore.RED + f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    interactive_error_handler_demo()