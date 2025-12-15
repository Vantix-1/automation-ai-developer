# Function Calling & Structured Outputs

---

## 🎯 Learning Objectives

- Master OpenAI function calling for tool/API integration
- Implement structured outputs with JSON mode
- Build production-ready error handling systems
- Create intelligent assistants that interact with real-world APIs
- Learn advanced prompt engineering for data extraction

---

## 📋 Prerequisites

- ✅ Completed Week 1: Days 6-8 (Advanced Tools)
- ✅ OpenAI API key in `.env` file
- ✅ Python 3.11+ installed
- ✅ Understanding of basic API concepts

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to this folder
cd phase_2_active/openai_api/week1_day9-10

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Create/update `.env` file:

```env
OPENAI_API_KEY=sk-your-key-here
# Optional for real weather data:
# WEATHER_API_KEY=your-weather-api-key
```

### 3. Run the Demos

```bash
# Weather Assistant with function calling
python weather_assistant.py

# Structured data extraction
python structured_outputs.py

# Advanced error handling
python api_error_handler.py
```

---

## 📁 File Structure

```
week1_day9-10/
├── weather_assistant.py      # Function calling demo with weather API
├── structured_outputs.py     # JSON mode & data extraction
├── api_error_handler.py      # Advanced error handling
├── function_calling_guide.md # Best practices guide
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

---

## 🛠️ Features Implemented

### Weather Assistant (`weather_assistant.py`)

- **Function Calling Implementation** - Define and handle multiple tools
- **Real/Mock Weather Data** - Switch between real API and mock data
- **Multiple Operations** - Current weather, forecasts, comparisons, alerts
- **Interactive Chat** - Natural language interface to weather functions
- **Temperature Unit Conversion** - Celsius/Fahrenheit support
- **Error Handling** - Graceful degradation when APIs fail

### Structured Outputs (`structured_outputs.py`)

- **JSON Mode Usage** - Force structured JSON responses from OpenAI
- **Pydantic Schema Integration** - Type-safe data extraction
- **Multiple Use Cases** - Contact info, products, meetings, news articles
- **Data Transformation** - Convert between different schemas
- **Validation & Fixing** - Auto-correct invalid data with AI
- **Template-based Generation** - Create structured data from templates

### API Error Handler (`api_error_handler.py`)

- **Circuit Breaker Pattern** - Prevent cascading failures
- **Retry Logic** - Exponential backoff with tenacity
- **Rate Limiting** - Token and request rate management
- **Comprehensive Metrics** - Track success rates and error types
- **Health Monitoring** - System status and recovery tracking
- **User-friendly Errors** - Clear suggestions for common issues

---

## 🎮 How to Use

### 1. Weather Assistant Demo

```bash
python weather_assistant.py
```

**Available Commands:**

- `What's the weather in Tokyo?`
- `Compare New York and London weather`
- `3-day forecast for Paris`
- `Are there weather alerts in Miami?`
- `help` - Show available commands
- `clear` - Clear conversation history
- `quit` - Exit

### 2. Structured Outputs Demo

```bash
python structured_outputs.py
```

**Interactive Menu:**

1. Extract contact information from text
2. Extract product details
3. Summarize meeting notes
4. Parse news articles
5. Batch extract from multiple texts
6. Validate and fix invalid data
7. Generate from templates
8. Transform between formats

### 3. API Error Handler Demo

```bash
python api_error_handler.py
```

**Features to Test:**

- Safe API calls with automatic retries
- External API error handling
- Health status monitoring
- Circuit breaker simulation
- Error metrics tracking

---

## 💡 Key Learnings

### 1. Function Calling Concepts

- **Tool Definitions** - How to define functions for OpenAI
- **Parameter Schemas** - JSON Schema for function arguments
- **Tool Choice** - Letting model decide when to use tools
- **Response Handling** - Processing function call results

### 2. Structured Data Patterns

- **JSON Mode** - Forcing structured output format
- **Schema Validation** - Using Pydantic for type safety
- **Data Extraction** - Pulling structured data from unstructured text
- **Template Filling** - Generating data from templates

### 3. Production Error Handling

- **Circuit Breakers** - Preventing system overload
- **Retry Strategies** - Exponential backoff and jitter
- **Rate Limiting** - Managing API quotas
- **Graceful Degradation** - Fallback mechanisms
- **Comprehensive Logging** - Tracking for debugging

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Function Calling Not Working** | Check tool definitions and descriptions |
| **JSON Mode Errors** | Ensure `response_format` is set correctly |
| **Rate Limiting** | Implement proper rate limiting in error handler |
| **Schema Validation Failures** | Check Pydantic model definitions |
| **API Connection Issues** | Verify network and API keys |

### Debug Commands

```bash
# Test function calling
python -c "from openai import OpenAI; client = OpenAI(); print('OpenAI connection OK')"

# Test Pydantic
python -c "from pydantic import BaseModel; print('Pydantic version:', BaseModel.__version__)"

# Test tenacity (retry library)
python -c "import tenacity; print('Tenacity version:', tenacity.__version__)"
```

---

## 🎯 Next Steps

### Enhance Weather Assistant

- **Add Real Weather APIs** - OpenWeatherMap, WeatherAPI, etc.
- **Historical Data** - Add weather history lookup
- **Location Autocomplete** - Improve location parsing
- **Weather Maps** - Integrate visual weather maps
- **Notifications** - Weather alert notifications

### Extend Structured Outputs

- **More Schemas** - Add industry-specific schemas
- **Database Integration** - Store extracted data
- **Stream Processing** - Handle streaming data
- **Custom Validation** - Add business logic validation
- **Schema Evolution** - Handle schema changes over time

### Improve Error Handling

- **Distributed Tracing** - Add request tracing
- **Alerting System** - Notify on critical errors
- **Performance Metrics** - Track latency and throughput
- **A/B Testing** - Compare different error strategies
- **Automated Recovery** - Self-healing systems

---

## 📊 Success Metrics

- ✅ Successfully make 5+ different function calls
- ✅ Extract structured data with 90%+ accuracy
- ✅ Handle API errors gracefully with automatic recovery
- ✅ Implement circuit breaker that triggers correctly
- ✅ Create at least 3 custom Pydantic schemas
- ✅ Achieve < 1% error rate in production-like conditions

---

## 📚 Resources

- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling)
- [OpenAI JSON Mode Guide](https://platform.openai.com/docs/guides/structured-outputs)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Tenacity Retry Library](https://tenacity.readthedocs.io/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)

---

<div align="center">

**Happy Coding! 🚀**


</div>