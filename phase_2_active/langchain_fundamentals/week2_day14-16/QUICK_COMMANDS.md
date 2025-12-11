# ⚡ Quick Command Reference

## 🚀 Run Examples

```powershell
# Day 14: Sequential Chains
python sequential_chains.py

# Day 15: Memory Systems  
python memory_systems.py

# Day 15-16: Interactive Assistant
python multi_step_assistant.py              # Chat mode
python multi_step_assistant.py example      # Example workflow
python multi_step_assistant.py test         # Quick test
python multi_step_assistant.py --debug      # Debug mode

# Day 16: Chain Routing
python chain_routing.py
```

---

## 🔧 Useful Commands

```powershell
# Check installed packages
pip list | Select-String "langchain"

# Test imports
python -c "from langchain_core.prompts import ChatPromptTemplate; print('✅ Works!')"

# Check API key
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('✅' if os.getenv('OPENAI_API_KEY') else '❌')"

# Quick LangChain test
python -c "
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model='gpt-3.5-turbo')
prompt = ChatPromptTemplate.from_template('Say hello in {language}')
chain = prompt | llm | StrOutputParser()
print(chain.invoke({'language': 'Spanish'}))
"
```

---

## 📝 Common Code Patterns

### Simple Chain
```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("Explain {topic}")
chain = prompt | llm | StrOutputParser()

result = chain.invoke({"topic": "AI"})
```

### With Memory
```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

history = ChatMessageHistory()
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()
response = chain.invoke({"history": history.messages, "input": "Hello"})

history.add_user_message("Hello")
history.add_ai_message(response)
```

### Sequential Steps
```python
# Step 1: Generate
step1_prompt = ChatPromptTemplate.from_template("Generate ideas for {topic}")
step1 = step1_prompt | llm | StrOutputParser()

# Step 2: Refine
step2_prompt = ChatPromptTemplate.from_template("Refine these ideas: {ideas}")
step2 = step2_prompt | llm | StrOutputParser()

# Chain them
chain = (
    step1 
    | (lambda ideas: {"ideas": ideas})
    | step2
)

result = chain.invoke({"topic": "AI projects"})
```

---

## 🐛 Quick Fixes

### Fix: Module not found
```powershell
pip install langchain langchain-openai langchain-core langchain-community
```

### Fix: API key not working
```powershell
# Windows
$env:OPENAI_API_KEY="your-key-here"

# Or in .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

### Fix: Import errors
Replace:
- `from langchain.chains import LLMChain` → Use modern LCEL
- `from langchain.prompts import PromptTemplate` → `from langchain_core.prompts import ChatPromptTemplate`
- `chain.run()` → `chain.invoke()`

---

## ✅ Verify Setup

```powershell
# All-in-one verification
python -c "
import sys
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.chat_message_histories import ChatMessageHistory
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    print('✅ All imports successful!')
    print('✅ API Key:', '✓' if os.getenv('OPENAI_API_KEY') else '✗ MISSING')
    print('✅ Python:', sys.version.split()[0])
    
    import langchain
    print('✅ LangChain:', langchain.__version__)
except Exception as e:
    print(f'❌ Error: {e}')
"
```

---

## 📊 File Overview

| File | Purpose | Runtime |
|------|---------|---------|
| `sequential_chains.py` | Multi-step workflows | ~30 sec |
| `memory_systems.py` | Memory demos | ~45 sec |
| `multi_step_assistant.py` | Interactive chat | Interactive |
| `chain_routing.py` | Smart routing | ~60 sec |

---

## 🎯 Learning Checklist

```
Day 14: Sequential Chains
├─ [ ] Run sequential_chains.py
├─ [ ] Understand | operator
├─ [ ] Create custom workflow
└─ [ ] Practice error handling

Day 15: Memory Systems
├─ [ ] Run memory_systems.py
├─ [ ] Try all memory types
├─ [ ] Chat with assistant
└─ [ ] Save/load history

Day 16: Chain Routing
├─ [ ] Run chain_routing.py
├─ [ ] Test query routing
├─ [ ] Try orchestration
└─ [ ] Build custom router
```

---

## 💾 Save This!

Keep this file open as a reference while working through Days 14-16. It contains all the commands you'll need!