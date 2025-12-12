# Agents & Tools

**Objective:** Master LangChain agents and build custom tools for specialized tasks

## 🚀 Quick Start

```bash
# Navigate to Days 17-19
cd week2_day17-19

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Set API key
echo "OPENAI_API_KEY=your_key_here" > .env

# Optional: Install web search capabilities
pip install duckduckgo-search wikipedia
```

## 📊 Day-by-Day Learning

### Agent Fundamentals
**File:** `agents_intro.py`

- ✅ Understand LangChain agents architecture
- ✅ Implement basic agents with tools
- ✅ Learn ReAct (Reasoning + Acting) pattern
- ✅ Build agent with calculator, text processing, and weather tools

### Custom Tools
**File:** `custom_tools.py`

- ✅ Create custom tools from scratch
- ✅ Build database query tools
- ✅ Implement file system operations
- ✅ Create API integration tools
- ✅ Learn tool decorator patterns

### Research Agent
**File:** `research_agent.py`

- ✅ Build web search capabilities
- ✅ Integrate Wikipedia searches
- ✅ Create citation management system
- ✅ Implement content analysis tools
- ✅ Build complete research workflow

## 📁 Files Overview

### agents_intro.py
- Basic agent implementation
- ReAct pattern demonstration
- Multiple tool integration
- Step-by-step reasoning visualization

### custom_tools.py
- Custom tool creation patterns
- Database, filesystem, and API tools
- Tool building workshop
- Real-world tool examples

### research_agent.py
- Web search integration (DuckDuckGo)
- Wikipedia API integration
- Citation management system
- Content quality analysis
- Interactive research session

## 🛠️ Required Tools Setup

### Core Installation
```bash
# Basic installation
pip install langchain langchain-openai openai python-dotenv

# For web search (Day 19)
pip install duckduckgo-search wikipedia

# For better tool functionality
pip install requests pandas numpy
```

### Optional Enhancements
```bash
# Real database connections
pip install sqlalchemy psycopg2-binary

# Advanced APIs
pip install google-search-results

# File processing
pip install python-magic
```

---
## 🔧 Key Concepts

### 1. Agents vs Chains
- **Chains:** Deterministic workflows
- **Agents:** Dynamic tool selection based on input
- Use chains for predictable tasks
- Use agents for complex, variable tasks

### 2. Tool Design Principles
- Single responsibility per tool
- Clear input/output specifications
- Error handling and validation
- Proper documentation

### 3. Research Agent Architecture
- Web search for current information
- Wikipedia for foundational knowledge
- Citation tracking for credibility
- Content analysis for quality

## 📚 Example Usage

### Run Agent Introduction
```bash
python agents_intro.py
```

### Build Custom Tools
```bash
python custom_tools.py
```

### Conduct Research
```bash
python research_agent.py

# Interactive mode
python research_agent.py --interactive
```

## 🆘 Troubleshooting

### Common Issues

**Tool Import Errors**
```bash
# Missing duckduckgo-search
pip install duckduckgo-search

# Missing wikipedia
pip install wikipedia
```

**Agent Execution Errors**
```python
# Increase max_iterations
agent_executor = AgentExecutor(
    max_iterations=10,  # Increase from default 5
    handle_parsing_errors=True
)
```

**API Rate Limits**
```python
# Add delays between calls
import time
time.sleep(1)  # 1 second delay
```

### Quick Fixes
```bash
# Reinstall all packages
pip install --upgrade --force-reinstall -r requirements.txt

# Clear cache
pip cache purge
```

## 🎓 Advanced Topics

### After:
- **Multi-Agent Systems:** Agents that collaborate
- **Tool Orchestration:** Complex tool workflows
- **Memory in Agents:** Stateful agent conversations
- **Custom Prompts:** Specialized agent instructions
- **Evaluation:** Testing agent performance

### Project Ideas
- Customer support agent with knowledge base
- Data analysis agent with visualization tools
- Code review agent with linting tools
- Personal research assistant with bookmarks
- E-commerce agent with product search

## 📖 Additional Resources

### Documentation
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [LangChain Tools](https://python.langchain.com/docs/modules/agents/tools/)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)

### Tutorials
- LangChain Agents Crash Course
- Building Custom Tools Workshop
- Research Agent Implementation Guide

### Community
- LangChain Discord #agents channel
- GitHub LangChain examples
- Stack Overflow #langchain-agents