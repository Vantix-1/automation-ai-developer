# OpenAI API Fundamentals

## 🎯 Learning Objectives

- Set up OpenAI API environment
- Make first successful API call
- Understand different GPT models
- Implement basic chat interface
- Learn about tokens and costs

## 🎯 Enhanced Features Added

1. **Cost Calculator** - Real-time cost tracking and estimation
2. **JSON Export** - Full conversation export with metadata
3. **Streamlit Web Interface** - Modern web UI for chatting

## 📁 Updated File Structure

```
phase_2_active/openai_api/week1_day1-2/
├── enhanced_openai.py       # Main enhanced script
├── streamlit_app.py          # Web interface
├── requirements.txt          # Updated dependencies
├── .env.example              # Environment template
└── exports/                  # Auto-created export folder
```

## 🚀 Quick Start

### 1. Command Line Interface

```bash
# Activate virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install enhanced dependencies
pip install -r requirements.txt

# Run enhanced script
python enhanced_openai.py
```

### 2. Web Interface

```bash
# Run Streamlit app
streamlit run streamlit_app.py
```

## ✨ New Features

### 💰 Cost Calculator

- Real-time cost tracking per message
- Model-specific pricing
- Session statistics
- Cost breakdown by model

### 📤 JSON Export

- Full conversation history
- Metadata (tokens, costs, timestamps)
- Text version for readability
- Automatic export folder creation

### 🌐 Web Interface

- Modern, responsive UI
- Real-time streaming responses
- Interactive configuration panel
- Download conversation as JSON
- Cost calculator widget

## 🔧 Configuration

### API Key Setup

1. Copy `.env.example` to `.env`
2. Add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-key-here
```

### Running Both Interfaces

```bash
# Terminal chat with cost tracking
python enhanced_openai.py

# Web interface (opens in browser)
streamlit run streamlit_app.py
```

## 📊 Cost Management Features

### Real-time Tracking

- Token usage estimation
- Cost calculation per message
- Session total tracking
- Model comparison

### Export Options

```python
# Automatic export
connector.export_to_json()  # Creates timestamped JSON file

# Manual export
connector.export_to_json("my_conversation.json")
```

## 🎮 Enhanced Chat Commands

- `stats` - Show detailed session statistics
- `export` - Export conversation to JSON
- `model <name>` - Switch models on the fly
- `system <prompt>` - Update system prompt

## 💡 Example Usage

### Cost-Aware Chatting

```bash
$ python enhanced_openai.py
🎯 Start interactive chat with cost tracking? (y/n): y

👤 You: Explain quantum computing
🤖 AI: [Streaming response...]
📊 Estimated cost: $0.000156

👤 You: stats
📊 Session Statistics:
Model: gpt-3.5-turbo
Total Tokens: 2,450
Total Cost: $0.003675
Exchanges: 5
```

### Web Interface Features

- **Sidebar Configuration**: Model, temperature, system prompt
- **Real-time Stats**: Token count and cost tracking
- **Export Options**: Download conversation as JSON
- **Cost Calculator**: Estimate costs before chatting

## 🐛 Troubleshooting

### Common Issues

- **Streamlit not opening**: Check if port 8501 is available
- **API key errors**: Verify `.env` file format
- **Import errors**: Reinstall dependencies with `pip install -r requirements.txt`

### Debug Commands

```python
# Check Streamlit installation
import streamlit as st
print(st.__version__)

# Test OpenAI connection
import openai
openai.api_key = os.getenv('OPENAI_API_KEY')
models = openai.Model.list()
print(f"Available models: {len(models.data)}")
```