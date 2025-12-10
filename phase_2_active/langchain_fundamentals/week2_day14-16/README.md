# 🎯 Week 2: Days 14-16 - Sequential Chains & Memory

**Objective:** Master multi-step AI workflows and conversation memory systems

---

## 🚀 One-Command Setup

```bash
# 1. Navigate to week2_day14-16
cd week2_day14-16

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install ALL dependencies with one command
pip install -r requirements.txt

# 5. Set your OpenAI API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

---

## 📊 Day-by-Day Learning Path

### **Day 14: Sequential Chains**
**Files:** `sequential_chains.py`

**Concepts:**
- SimpleSequentialChain for linear workflows
- SequentialChain for complex branching
- Variable passing between chains
- Error handling in production

---

### **Day 15: Memory Systems**
**Files:** `memory_systems.py`, `multi_step_assistant.py`

**Concepts:**
- ConversationBufferMemory (complete history)
- ConversationBufferWindowMemory (sliding window)
- ConversationSummaryMemory (LLM summaries)
- File persistence for conversations

---

### **Day 16: Chain Routing & Orchestration**
**Files:** `chain_routing.py`

**Concepts:**
- Smart query routing to appropriate chains
- Workflow orchestration
- Four specialized chains (QA, Analysis, Creative, Code)
- Fallback mechanisms

---

## 📦 About requirements.txt

### What's Included

```bash
# Check what's installed:
pip list | grep -E "(langchain|openai|chroma|rich)"
```

### Core Packages (Must Have)
- `langchain` - Main framework
- `langchain-openai` - OpenAI integration
- `openai` - OpenAI API client
- `python-dotenv` - Environment management
- `chromadb` - Vector database for memory

### Memory & Storage
- `chromadb` - Stores conversation embeddings
- `redis` - Fast in-memory caching
- `pypdf` + `unstructured` - Document processing

### UI & Development
- `rich` - Beautiful terminal output
- `colorama` - Cross-platform colors
- `pytest` + `black` + `flake8` - Testing & code quality

---

## 🛠️ Installation Verification

```bash
# Test 1: Check packages
python -c "
import langchain, openai, chromadb, rich
print('✅ All core packages installed')
print(f'  LangChain: {langchain.__version__}')
print(f'  OpenAI: {openai.__version__}')
print(f'  ChromaDB: {chromadb.__version__}')
"

# Test 2: Basic LangChain functionality
python -c "
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
llm = ChatOpenAI(temperature=0.7)
prompt = PromptTemplate.from_template('Tell me about {topic}')
print('✅ LangChain setup successful!')
"
```

---

## 🎯 Run the Examples

### Day 14: Sequential Chains
```bash
python sequential_chains.py
```
Creates multi-step workflows for content generation and product marketing

### Day 15: Memory Systems
```bash
python memory_systems.py
```
Shows different memory types and conversation persistence

### Day 15-16: Interactive Assistant
```bash
python multi_step_assistant.py
```
Runs assistant with 4-step reasoning and memory

### Day 16: Chain Routing
```bash
python chain_routing.py
```
Demonstrates smart workflow routing

---

## 🔧 Troubleshooting

### Common Issues & Solutions

**1. Installation Timeout**
```bash
# Install without dependencies first
pip install --no-deps langchain-openai openai

# Then install the rest
pip install -r requirements.txt --no-deps
```

**2. ChromaDB Issues**
```bash
# If ChromaDB fails, use in-memory only
python -c "import chromadb; print('ChromaDB working')"
```

**3. Memory Errors**
```bash
# Reduce memory usage
export TOKENIZERS_PARALLELISM=false
```

**4. OpenAI API Key**
```bash
# Verify .env file
cat .env
# Should show: OPENAI_API_KEY=sk-...
```

### Quick Fix Commands
```bash
# Recreate virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📚 Package Details

### Essential for Days 14-16

| Package | Purpose | Required |
|---------|---------|----------|
| langchain | Core framework | ✅ Yes |
| langchain-openai | OpenAI integration | ✅ Yes |
| openai | API client | ✅ Yes |
| chromadb | Vector storage for memory | ✅ Yes |
| rich | Terminal UI | ✅ Yes |
| python-dotenv | Environment variables | ✅ Yes |

### For Advanced Features

| Package | Purpose | Use Case |
|---------|---------|----------|
| redis | Fast caching | Persistent session storage |
| pypdf | PDF processing | Document memory |
| sentence-transformers | Local embeddings | Offline memory similarity |
| tiktoken | Token counting | Memory optimization |

---

## 🎯 Success Checklist

### After Installation:
- ✅ `pip install -r requirements.txt` completes without errors
- ✅ Can import langchain and openai
- ✅ .env file exists with API key
- ✅ All 4 Python files run successfully

### After Day 14:
- ✅ Understand SimpleSequentialChain vs SequentialChain
- ✅ Can create 3-step content workflow
- ✅ Know how to pass variables between chains

### After Day 15:
- ✅ Implemented all 3 memory types
- ✅ Can save/load conversation history
- ✅ Built assistant with persistent memory

### After Day 16:
- ✅ Created custom chain router
- ✅ Understand workflow orchestration
- ✅ Built fallback mechanisms

---

## 📖 Additional Resources

### Documentation
- [LangChain Sequential Chains](https://python.langchain.com/docs/modules/chains/)
- [LangChain Memory](https://python.langchain.com/docs/modules/memory/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Learning Path
1. Start with `sequential_chains.py` (Day 14)
2. Move to `memory_systems.py` (Day 15)
3. Build assistant with `multi_step_assistant.py` (Day 15-16)
4. Finish with `chain_routing.py` (Day 16)

---

## 🚀 One-Click Setup Script

### **setup_week2.sh**
```bash
#!/bin/bash

# Create Week 2 directory structure
mkdir -p week2_day14-16

# Create single requirements.txt
cat > week2_day14-16/requirements.txt << 'EOF'
# 🎯 Week 2: Days 14-16 - Sequential Chains & Memory
langchain==0.1.0
langchain-openai==0.0.2
langchain-community==0.0.10
openai==1.3.0
tiktoken==0.5.2
sentence-transformers==2.2.2
chromadb==0.4.18
pypdf==3.17.4
unstructured==0.12.2
python-dotenv==1.0.0
pydantic==2.5.0
orjson==3.9.10
redis==5.0.1
rich==13.7.0
colorama==0.4.6
tqdm==4.66.1
pytest==7.4.3
black==23.11.0
flake8==6.1.0
mypy==1.7.0
EOF

echo "✅ Created week2_day14-16/requirements.txt"
echo ""
echo "📦 To install:"
echo "   cd week2_day14-16"
echo "   python -m venv venv"
echo "   source venv/bin/activate  # or venv\Scripts\activate on Windows"
echo "   pip install -r requirements.txt"
echo ""
echo "🚀 Then run:"
echo "   python sequential_chains.py     # Day 14"
echo "   python memory_systems.py       # Day 15"
echo "   python multi_step_assistant.py # Day 15-16"
echo "   python chain_routing.py        # Day 16"
```

### Minimal Setup Script (If you want a lighter version)
```bash
#!/bin/bash

# Minimal requirements for learning
cat > week2_day14-16/requirements_minimal.txt << 'EOF'
# 🎯 Minimal setup for Days 14-16
langchain==0.1.0
langchain-openai==0.0.2
openai==1.3.0
python-dotenv==1.0.0
rich==13.7.0
EOF

echo "✅ Created minimal requirements file"
echo "📦 Install with: pip install -r requirements_minimal.txt"
```

---

## 📝 How to Use This Setup

### 1. Create the Directory
```bash
mkdir -p week2_day14-16
cd week2_day14-16
```

### 2. Create requirements.txt
```bash
# Copy the requirements.txt content above into a file
touch requirements.txt
# Paste the content
```

### 3. Install Everything
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
# Run this test
python -c "
try:
    import langchain, openai, chromadb, rich, pydantic
    print('✅ All packages installed successfully!')
    print(f'LangChain: {langchain.__version__}')
except ImportError as e:
    print(f'❌ Missing package: {e}')
"
```

---

## 🎯 What This Gives You

### Complete Development Environment:
- ✅ LangChain with all memory systems
- ✅ Vector database for persistent memory
- ✅ Beautiful terminal interfaces
- ✅ Full testing framework
- ✅ Code quality tools

### Single File Management:
- One requirements.txt to rule them all
- No complex dependency management
- Easy version control
- Simple reproduction

### Production Ready:
- Error handling dependencies
- Performance tools
- Monitoring capabilities
- Security packages

---

## 🎓 Next Steps

After completing Days 14-16:
1. Review and refactor your code
2. Add error handling to all chains
3. Create custom memory implementations
4. Prepare for Week 3: Agents & Tools

**Tip:** Run `pip list` to see all installed packages. Use `pip freeze > requirements.txt` to save your exact versions.

---

**Ready to Master AI Workflows?** Complete all three days to become proficient with sequential chains and memory! 🚀