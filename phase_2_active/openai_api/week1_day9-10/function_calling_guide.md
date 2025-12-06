# 📚 Function Calling Guide - Best Practices

## 🎯 What is Function Calling?

Function calling allows OpenAI models to intelligently choose to output a JSON object containing arguments to call functions/tools you've defined. This enables:

1. **Structured data extraction** from natural language
2. **Tool/API integration** - models can decide when to use external tools
3. **Multi-step reasoning** - chain function calls for complex tasks

---

## 🔧 Basic Structure

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and country"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

---

## 📝 Best Practices

### 1. Clear Function Names & Descriptions

```python
# GOOD: Clear and specific
{
    "name": "calculate_shipping_cost",
    "description": "Calculate shipping cost based on weight, destination, and service type"
}

# BAD: Vague
{
    "name": "calculate",
    "description": "Do calculation"
}
```

### 2. Detailed Parameter Descriptions

```python
# GOOD: Specific with examples
{
    "type": "string",
    "description": "Email address format: user@example.com. Must be valid."
}

# BAD: Generic
{
    "type": "string",
    "description": "Email"
}
```

### 3. Use Enums for Limited Options

```python
# GOOD: Clear valid options
{
    "type": "string",
    "enum": ["pending", "processing", "shipped", "delivered"],
    "description": "Order status"
}

# BAD: Free text for constrained values
{
    "type": "string",
    "description": "Order status"
}
```

### 4. Required vs Optional Parameters

```python
# Mark only truly required fields as required
"required": ["location", "date"]

# Optional parameters should have defaults or be nullable
{
    "type": "string",
    "description": "Optional notes",
    "nullable": True
}
```

---

## 🚀 Advanced Patterns

### Pattern 1: Sequential Function Calls

```python
# Model can call multiple functions in sequence
messages = [
    {"role": "user", "content": "What's the weather in Tokyo and compare it to London?"}
]

# Model might call:
# 1. get_weather(location="Tokyo")
# 2. get_weather(location="London")
# 3. compare_weather(locations=["Tokyo", "London"])
```

### Pattern 2: Conditional Execution

```python
# Based on conversation, model decides which function to call
tools = [
    {
        "name": "search_products",
        "description": "Search for products by keyword"
    },
    {
        "name": "get_product_details",
        "description": "Get detailed information about a specific product"
    },
    {
        "name": "check_availability",
        "description": "Check if a product is in stock"
    }
]

# User: "Find laptops under $1000"
# → search_products(query="laptops", max_price=1000)

# User: "Is the MacBook Pro available?"
# → check_availability(product_id="macbook-pro")
```

### Pattern 3: Validation & Transformation

```python
# Use functions to validate and transform data
def validate_email(email: str) -> Dict:
    """Validate email format and domain"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        return {"valid": False, "error": "Invalid email format"}
    
    # Check domain (example)
    if email.endswith("@example.com"):
        return {"valid": False, "error": "Disallowed domain"}
    
    return {"valid": True, "normalized": email.lower()}
```

---

## 🛡️ Error Handling

### 1. Graceful Failure

```python
def handle_function_call(function_name: str, arguments: Dict) -> Dict:
    try:
        result = call_function(function_name, arguments)
        return {
            "success": True,
            "data": result,
            "function": function_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "function": function_name,
            "suggestion": "Try with different parameters"
        }
```

### 2. Retry Logic

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def call_external_api(url: str, params: Dict):
    # Implementation with retries
    pass
```

---

## 📊 JSON Mode Best Practices

### 1. Schema Validation

```python
from pydantic import BaseModel, Field

class UserProfile(BaseModel):
    name: str = Field(description="User's full name")
    email: str = Field(description="Valid email address")
    age: Optional[int] = Field(None, description="Age in years", ge=0, le=120)

# Use with OpenAI JSON mode
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    response_format={"type": "json_object"}
)
```

### 2. Structured Extraction Prompts

```python
prompt = """
Extract the following information from the text and return as JSON:

Required fields:
- name (string): Person's full name
- email (string): Email address
- phone (string): Phone number, if mentioned
- company (string): Company name, if mentioned

Text: {text}

Return ONLY valid JSON.
"""
```

---

## 🔄 Workflow Examples

### Example 1: Customer Support Bot

```python
tools = [
    {
        "name": "search_knowledge_base",
        "description": "Search knowledge base articles"
    },
    {
        "name": "create_support_ticket",
        "description": "Create a new support ticket"
    },
    {
        "name": "check_order_status",
        "description": "Check status of an order"
    },
    {
        "name": "schedule_callback",
        "description": "Schedule a callback from support"
    }
]

# User: "My order #12345 hasn't arrived"
# → check_order_status(order_id="12345")
# → create_support_ticket(issue="delayed_order", order_id="12345")
```

### Example 2: Travel Assistant

```python
tools = [
    {
        "name": "search_flights",
        "description": "Search for available flights"
    },
    {
        "name": "search_hotels",
        "description": "Search for hotels"
    },
    {
        "name": "get_weather_forecast",
        "description": "Get weather forecast"
    },
    {
        "name": "convert_currency",
        "description": "Convert between currencies"
    }
]

# User: "Plan a trip to Paris next week"
# → search_flights(destination="Paris", date="2024-12-20")
# → search_hotels(location="Paris", check_in="2024-12-20", nights=7)
# → get_weather_forecast(location="Paris", days=7)
```

---

## 🚨 Common Pitfalls & Solutions

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Vague Descriptions** | Model doesn't understand when to use the function | Be specific about use cases |
| **Too Many Required Parameters** | Model can't gather all info from user message | Make parameters optional when possible, ask for clarification |
| **No Error Handling** | Function fails and breaks conversation | Always wrap in try-except, return helpful error messages |
| **Ignoring Context** | Functions don't consider conversation history | Pass relevant context to functions |

---

## 📈 Performance Tips

- **Cache Results**: Cache frequent function calls
- **Batch Processing**: Group similar requests
- **Async Operations**: Use async/await for I/O bound operations
- **Limit Tool Choice**: Use `tool_choice` parameter to guide model
- **Stream Responses**: For long-running functions, stream partial results

---

## 🔍 Debugging Tips

### 1. Log Function Calls

```python
def log_function_call(function_name: str, arguments: Dict, result: Any):
    logger.info(f"Function: {function_name}")
    logger.info(f"Arguments: {arguments}")
    logger.info(f"Result: {result}")
```

### 2. Test with Different Models

```python
# Test with different models
models_to_test = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]

for model in models_to_test:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools
    )
    # Compare results
```

### 3. Use Validation Examples

```python
# Provide examples of valid and invalid calls
validation_examples = [
    {
        "input": "What's the weather in Tokyo?",
        "expected_function": "get_weather",
        "expected_args": {"location": "Tokyo"}
    },
    {
        "input": "Compare Tokyo and London weather",
        "expected_functions": [
            {"name": "get_weather", "args": {"location": "Tokyo"}},
            {"name": "get_weather", "args": {"location": "London"}},
            {"name": "compare_weather", "args": {"locations": ["Tokyo", "London"]}}
        ]
    }
]
```

---

## 🎯 When to Use Function Calling

### ✅ Use Function Calling When:

- You need structured data from unstructured text
- You want to integrate with external APIs/tools
- You need multi-step reasoning with tool use
- You want to enforce specific output formats

### ❌ Consider Alternatives When:

- Simple Q&A without external tools needed
- Single-step tasks without structured output
- Very simple data extraction (regex might suffice)
- Performance-critical applications (direct API calls faster)

---

## 📚 Resources

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI Cookbook - Function Calling Examples](https://cookbook.openai.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/) for schema validation
- [Tenacity](https://tenacity.readthedocs.io/) for retry logic