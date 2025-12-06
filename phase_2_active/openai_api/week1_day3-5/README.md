# Week 1: Days 3-5 - Prompt Engineering & Enhanced Chatbot

## 🎯 Learning Objectives

- Master prompt engineering principles from DeepLearning.AI course
- Build an enhanced AI chatbot with context management
- Implement different prompt templates for various use cases
- Learn token estimation and cost optimization
- Experiment with advanced prompt patterns

## 📋 Prerequisites

- ✅ Completed Week 1: Days 1-2 (OpenAI API basics)
- ✅ OpenAI API key in `.env` file
- ✅ Python 3.11+ installed

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to this folder
cd phase_2_active/openai_api/week1_day3-5

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy example environment file
copy .env.example .env  # Windows
# or
cp .env.example .env    # Mac/Linux

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Enhanced Chatbot

```bash
python ai_chat_bot.py
```

## 📁 File Structure

```
week1_day3-5/
├── ai_chat_bot.py           # Main enhanced chatbot
├── prompt_lab.py            # Prompt engineering experiments
├── summarizer.py            # Content summarization tool
├── prompt_templates.md      # Collection of prompt templates
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

## 🛠️ Features Implemented

### Enhanced Chatbot (ai_chat_bot.py)

- **Multiple Prompt Templates**: Coding assistant, creative writer, learning coach, productivity expert
- **Context Management**: Automatic conversation history optimization
- **Token Estimation**: Approximate token counting without external libraries
- **Cost Tracking**: Real-time cost estimation
- **Template Switching**: Change AI behavior on the fly
- **Conversation Export**: Save chats as JSON or text

### Prompt Lab (prompt_lab.py)

- **Prompt Variations**: Test different versions of prompts
- **Temperature Experiments**: See how temperature affects responses
- **Comparative Analysis**: Compare results side by side

### Content Summarizer (summarizer.py)

- **Multiple Styles**: Concise, detailed, bullet points, TL;DR
- **Long Document Support**: Chunking for large texts
- **Sentiment Analysis**: Additional text insights

## 🎮 How to Use

### Interactive Chat Demo

```bash
python ai_chat_bot.py
```

You'll see:
- Available prompt templates
- Template selection menu
- Interactive chat interface

### Chat Commands

- `template <name>` - Switch to a different template
- `stats` - Show conversation statistics
- `export` - Export conversation to file
- `quit` - Exit the chat

### Experiment with Prompts

```bash
python prompt_lab.py
```

### Summarize Content

```bash
python summarizer.py
```

## 📚 Prompt Engineering Templates

### Available Templates

- **Coding Assistant** - Help with programming (temperature: 0.3)
- **Creative Writer** - Storytelling and content creation (temperature: 0.8)
- **Learning Coach** - Educational explanations (temperature: 0.5)
- **Productivity Expert** - Time management advice (temperature: 0.4)

### Prompt Patterns

- **Zero-shot**: Simple direct prompts
- **Few-shot**: Examples provided in prompt
- **Chain-of-thought**: Step-by-step reasoning
- **Role-playing**: Assume specific personas

## 💡 Key Learnings

### 1. Prompt Engineering Principles

- Be specific and provide context
- Use examples to guide the AI
- Specify desired format and length
- Iterate and refine prompts

### 2. Cost Optimization

- Estimate tokens using simple calculation (~4 chars per token)
- Manage context window to avoid excessive tokens
- Use appropriate models for tasks (GPT-3.5-turbo for testing)

### 3. Conversation Management

- Maintain conversation history
- Implement context window limits
- Handle system prompts effectively

## 🐛 Troubleshooting

### Common Issues

- **ModuleNotFoundError**: Install missing packages with `pip install -r requirements.txt`
- **API Key Error**: Ensure `.env` file exists with correct API key
- **Rate Limiting**: Wait 60 seconds and try again
- **Context Too Long**: Chatbot automatically manages context, but very long conversations may need manual reset

### Debug Commands

```python
# Check environment
python -c "import os; print('API Key exists:', bool(os.getenv('OPENAI_API_KEY')))"

# Test OpenAI connection
python -c "from openai import OpenAI; import os; client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')); print('Connection test:', client.models.list().data[0].id if client.models.list().data else 'No models')"
```