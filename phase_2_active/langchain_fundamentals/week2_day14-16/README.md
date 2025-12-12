# Complete Setup Guide

---

## 🚀 Quick Start

### 1. Set Up Your API Key

```powershell
# Create .env file in week2_day14-16 directory
echo "OPENAI_API_KEY=your_key_here" > .env
```

### 2. Replace Your Python Files

Copy the updated code from the artifacts:
- ✅ `sequential_chains.py` - Modern LCEL version
- ✅ `memory_systems.py` - Modern LCEL version
- ✅ `multi_step_assistant.py` - Modern LCEL version
- ✅ `chain_routing.py` - Modern LCEL version

### 3. Run the Examples

```powershell
# Sequential Chains
python sequential_chains.py

# Memory Systems
python memory_systems.py

# Multi-Step Assistant (interactive)
python multi_step_assistant.py

# Multi-Step Assistant (example workflow)
python multi_step_assistant.py example

# Multi-Step Assistant (quick test)
python multi_step_assistant.py test

# Chain Routing
python chain_routing.py
```

---


## 📚 File Descriptions

### `sequential_chains.py`
**Purpose:** Multi-step AI workflows

**Features:**
- Basic 2-step chain (name → slogan)
- Content creation pipeline (outline → write → optimize)
- E-commerce workflow (description → audience → strategy)

**Run:**
```powershell
python sequential_chains.py
```

---

### `memory_systems.py`
**Purpose:** Conversation memory demonstrations

**Features:**
- Buffer Memory (complete history)
- Window Memory (last N messages)
- Summary Memory (compressed history)
- File Persistence (save/load conversations)

**Run:**
```powershell
python memory_systems.py
```

---

### `multi_step_assistant.py`
**Purpose:** Production-ready assistant with 4-step reasoning

**Features:**
- Understanding → Research → Synthesis → Validation
- Conversation memory across sessions
- Interactive chat interface
- Example workflows

**Run:**
```powershell
# Interactive mode
python multi_step_assistant.py

# Example workflow
python multi_step_assistant.py example

# Quick test
python multi_step_assistant.py test

# Debug mode
python multi_step_assistant.py --debug
```

---

### `chain_routing.py`
**Purpose:** Smart query routing and orchestration

**Features:**
- 4 specialized chains (QA, Analysis, Creative, Code)
- Automatic query classification
- Complex workflow orchestration
- Conditional branching

**Run:**
```powershell
python chain_routing.py
```

---

## 🎯 Learning Path

### Sequential Chains
1. Run `sequential_chains.py`
2. Understand the `|` operator for chaining
3. Experiment with different prompts
4. Create your own 3-step workflow

**Key Concepts:**
- Chaining prompts with `|`
- Passing data between steps
- Error handling
- Building complex pipelines

---

### Memory Systems
1. Run `memory_systems.py`
2. Try different memory types
3. Understand when to use each type
4. Practice with `multi_step_assistant.py`

**Key Concepts:**
- Buffer Memory (everything)
- Window Memory (recent only)
- Summary Memory (compressed)
- Persistence (save/load)

---

### Chain Routing
1. Run `chain_routing.py`
2. Test with different query types
3. Understand routing logic
4. Build custom routing rules

**Key Concepts:**
- Query classification
- Specialized chains
- Workflow orchestration
- Fallback mechanisms

---

## 💡 Common Patterns

### Pattern 1: Simple Chain
```python
# Prompt → LLM → Parse
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"input": "your text"})
```

### Pattern 2: Sequential Chain
```python
# Step 1 → Step 2 → Step 3
chain = (
    prompt1 | llm | parser
    | (lambda x: {"result": x})
    | prompt2 | llm | parser
)
```

### Pattern 3: With Memory
```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("Hello")
history.add_ai_message("Hi there!")

# Use in prompt
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
```

### Pattern 4: Routing
```python
from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (condition1, chain1),
    (condition2, chain2),
    default_chain  # Fallback
)
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'langchain.chains'"
**Solution:** The old `langchain.chains` module doesn't exist in 1.x. Use the modern code provided.

### Issue: "AttributeError: 'ChatOpenAI' object has no attribute 'run'"
**Solution:** Replace `.run()` with `.invoke()`. The API changed in 1.x.

### Issue: Memory not persisting
**Solution:** Make sure you're calling `.add_user_message()` and `.add_ai_message()` after each exchange.

### Issue: Chains not connecting properly
**Solution:** Check that you're passing dictionaries with the correct keys between chain steps.

---
