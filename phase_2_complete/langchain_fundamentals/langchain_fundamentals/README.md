# LangChain Fundamentals

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to this folder
cd phase_2_active/langchain_fundamentals/week2_day11-13

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
Ensure `.env` file exists with:

```env
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Demos

```bash
# LangChain setup and fundamentals
python langchain_setup.py

# Simple chain building examples
python simple_chains.py

# Advanced prompt templates
python prompt_templates.py
```

## 📁 File Structure

```
week2_day11-13/
├── langchain_setup.py      # LangChain introduction & setup
├── simple_chains.py        # Chain building examples
├── prompt_templates.py     # Advanced prompt templates
├── requirements.txt        # Python dependencies
├── .env.example           # Environment template
└── README.md              # This file
```

## 🛠️ Features Implemented

### LangChain Setup (`langchain_setup.py`)
- **LangChain Initialization**: Proper setup with OpenAI integration
- **Basic Chat Examples**: Simple LLM interactions using LangChain
- **Prompt Templates**: Creating and using reusable prompt templates
- **LLM Chains**: Building and executing chains for complex tasks
- **Sequential Chains**: Multi-step chain execution
- **Conversation Memory**: Implementing memory for chat applications
- **Text Embeddings**: Working with OpenAI embeddings
- **Cost Tracking**: Monitoring token usage and costs
- **Interactive Demo**: Hands-on exploration of all features

### Simple Chains (`simple_chains.py`)
- **Basic LLMChain**: Foundation chain building
- **Transformation Chains**: Data preprocessing and transformation
- **Conditional Chains**: Chains with branching logic
- **Chat Prompt Chains**: Conversation-style chains
- **Parallel Chains**: Running multiple chains simultaneously
- **Chain Comparison**: Testing different chain configurations
- **Interactive Builder**: Build and test chains interactively

### Advanced Prompt Templates (`prompt_templates.py`)
- **Basic Templates**: String interpolation and formatting
- **Chat Templates**: Conversation-style prompt templates
- **Few-Shot Templates**: Learning from examples
- **Dynamic Example Selection**: Smart example choosing
- **Template Partials**: Pre-filled template configurations
- **Custom Output Parsing**: Structured response parsing
- **Template Validation**: Debugging and validating templates
- **Interactive Lab**: Experiment with different template types

## 🎮 How to Use

### 1. LangChain Setup Demo

```bash
python langchain_setup.py
```

**Interactive Menu:**
- **Basic Chat** - Simple LLM interaction
- **Prompt Templates** - Template creation and usage
- **LLM Chains** - Chain building and execution
- **Sequential Chains** - Multi-step processing
- **Conversation Memory** - Memory implementation
- **Text Embeddings** - Working with embeddings
- **Cost Tracking** - Monitoring usage and costs

### 2. Simple Chains Demo

```bash
python simple_chains.py
```

**Chain Types to Explore:**
- **Basic LLMChain** - Foundation chains
- **Transformation Chain** - Data processing chains
- **Conditional Chain** - Logic-based chains
- **Chat Prompt Chain** - Conversation chains
- **Parallel Chains** - Simultaneous execution
- **Chain Comparison** - Performance comparison

### 3. Prompt Templates Demo

```bash
python prompt_templates.py
```

**Template Features:**
- **Basic Templates** - String formatting
- **Chat Templates** - Conversation templates
- **Few-Shot Templates** - Example-based learning
- **Dynamic Selection** - Smart example choosing
- **Template Partials** - Pre-configured templates
- **Output Parsing** - Structured response handling
- **Template Validation** - Debugging tools

## 💡 Key Learnings

### 1. LangChain Architecture
- **Models**: LLM wrappers and embeddings
- **Prompts**: Template management and formatting
- **Chains**: Sequential and conditional workflows
- **Memory**: Conversation state management
- **Agents**: Tool-using LLMs (coming in Days 14-16)
- **Callbacks**: Monitoring and logging

### 2. Chain Construction
- **Modular Design**: Building reusable components
- **Error Handling**: Graceful failure recovery
- **Performance**: Optimizing token usage
- **Cost Management**: Tracking and controlling expenses
- **Testing**: Validating chain behavior

### 3. Prompt Engineering with LangChain
- **Template Variables**: Dynamic content insertion
- **Few-Shot Learning**: Teaching through examples
- **Output Formatting**: Structured response generation
- **Context Management**: Maintaining conversation flow
- **Validation**: Ensuring prompt correctness

## 🐛 Troubleshooting

### Common Issues
- **Import Errors**: Ensure all packages are installed: `pip install -r requirements.txt`
- **API Key Issues**: Check `.env` file and OpenAI account
- **Template Errors**: Validate template variables match input variables
- **Memory Issues**: Check conversation buffer limits
- **Cost Concerns**: Monitor with `get_openai_callback()`

### Debug Commands

```bash
# Check LangChain installation
python -c "import langchain; print('LangChain version:', langchain.__version__)"

# Test OpenAI connection
python -c "from langchain_openai import ChatOpenAI; import os; llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY')); print('Connection OK')"

# Test embeddings
python -c "from langchain_openai import OpenAIEmbeddings; import os; emb = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY')); print('Embeddings OK')"
```

## 🎯 Next Steps

### Extend LangChain Setup
- **Add More Models**: Integrate other LLM providers (Anthropic, Cohere, etc.)
- **Database Integration**: Connect to vector databases (Chroma, Pinecone)
- **Custom Callbacks**: Implement advanced logging and monitoring
- **Async Support**: Add asynchronous chain execution
- **Batch Processing**: Handle multiple inputs efficiently

### Enhance Chain Building
- **Custom Chains**: Build domain-specific chain types
- **Error Recovery**: Implement robust error handling
- **Caching**: Add response caching for efficiency
- **Rate Limiting**: Implement request throttling
- **Validation**: Add input/output validation

### Advanced Template Features
- **Template Inheritance**: Create template hierarchies
- **Internationalization**: Multi-language template support
- **A/B Testing**: Compare template performance
- **Template Versioning**: Manage template evolution
- **Template Sharing**: Create reusable template libraries

## 📊 Success Metrics
- Successfully initialize LangChain with OpenAI
- Build and test at least 5 different chain types
- Create reusable prompt templates for common tasks
- Implement conversation memory in a chat application
- Achieve cost savings through chain optimization
- Extract structured data using custom output parsers

## 📚 Resources
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [LangChain Cookbook](https://github.com/langchain-ai/langchain/tree/master/cookbook)
- [OpenAI API Documentation](https://platform.openai.com/docs/api-reference)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Vector Databases Guide](https://www.pinecone.io/learn/vector-database/)