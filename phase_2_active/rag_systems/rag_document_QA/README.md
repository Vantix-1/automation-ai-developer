# Week 5-6, Days 28-30: RAG Document Q&A Systems

## 📚 Overview

This week focuses on building production-ready **Retrieval Augmented Generation (RAG)** systems for document question answering. You'll learn how to create intelligent systems that can answer questions based on documents you provide.

## 🎯 Learning Objectives

By the end of this week, you will be able to:

1. **Build complete RAG pipelines** from document ingestion to answer generation
2. **Implement source citation** to show where answers come from
3. **Handle multiple document formats** (PDF, TXT, CSV, DOCX, Markdown)
4. **Create conversational RAG systems** with memory
5. **Deploy production-ready RAG applications**

## 📁 Files in This Directory

### 1. `rag_document_qa.py`
A complete, production-ready RAG system with advanced features:

**Features:**
- Multiple document format support (PDF, TXT, DOCX, MD, CSV)
- Smart text chunking with configurable overlap
- Choice of embedding models (OpenAI or HuggingFace)
- Vector database integration (ChromaDB or FAISS)
- Conversational memory for multi-turn conversations
- Source citation and similarity scoring
- Streaming responses support
- Interactive command-line interface
- Export capabilities (JSON output)

**Key Components:**
- `RAGDocumentQASystem` class with comprehensive methods
- Document loading and preprocessing
- Vector store creation and management
- QA chain initialization
- Semantic search functionality
- Statistics and reporting

### 2. `rag_with_sources.py`
An advanced RAG system with detailed source tracking and citation:

**Features:**
- Detailed document metadata tracking
- Source citation with confidence scores
- Cost tracking for API calls
- Multiple citation formats (inline, footnotes)
- Document deduplication
- Comprehensive statistics and reporting
- Export to various formats
- Batch processing capabilities

**Key Components:**
- `RAGSystemWithSources` class with source tracking
- `DocumentMetadata` and `ChunkMetadata` data classes
- `SourceCitation` for detailed attribution
- `AnswerWithSources` for structured responses
- Confidence scoring and relevance calculation
- Report generation

## 🚀 Getting Started

### Prerequisites

```bash
# Install required packages
pip install -r ../requirements.txt

# Set up environment variables
echo "OPENAI_API_KEY=your-api-key-here" > .env
Basic Usage
python
from rag_document_qa import RAGDocumentQASystem

# Initialize the system
rag = RAGDocumentQASystem(
    persist_directory="./data/chroma_db",
    embedding_model="openai",
    model_name="gpt-4"
)

# Load documents
documents = rag.load_documents(["document1.pdf", "document2.txt"])

# Create vector store
rag.create_vector_store()

# Initialize QA chain
rag.initialize_qa_chain()

# Ask questions
answer = rag.ask_question("What is artificial intelligence?")
print(answer["answer"])
Advanced Usage with Sources
python
from rag_with_sources import RAGSystemWithSources

# Initialize with source tracking
rag = RAGSystemWithSources(
    persist_directory="./data/chroma_sources",
    llm_model="gpt-4"
)

# Load documents with metadata
metadata = rag.load_documents(["research_paper.pdf"])

# Create vector store
rag.create_vector_store()

# Ask question with detailed citations
answer = rag.ask_question_with_sources("What were the key findings?")
answer.print_formatted()
📊 Key Concepts
1. Document Processing Pipeline
text
Documents → Load → Split → Embed → Store → Retrieve → Generate → Answer
2. Text Chunking Strategies
RecursiveCharacterTextSplitter: Smart splitting at semantic boundaries

CharacterTextSplitter: Fixed-size character splitting

Overlap: Prevents context loss between chunks

3. Vector Databases
ChromaDB: Persistent, lightweight vector database

FAISS: Facebook's efficient similarity search

Embeddings: Convert text to numerical vectors

4. Retrieval Strategies
Semantic Search: Find similar content using embeddings

Hybrid Search: Combine semantic and keyword search

Re-ranking: Improve result relevance

5. Generation with Sources
Citation Generation: Automatically cite sources

Confidence Scoring: Measure answer reliability

Context Window Management: Handle long contexts

🧪 Examples
Example 1: Basic RAG System
bash
# Run basic example
python rag_document_qa.py --mode basic

# Run with custom documents
python rag_document_qa.py --folder ./documents --question "What is AI?"
Example 2: Interactive Mode
bash
# Start interactive session
python rag_document_qa.py --mode interactive --folder ./documents

# Commands in interactive mode:
#   Type your question to get answers
#   'stats' - Show document statistics
#   'search <query>' - Perform semantic search
#   'clear' - Clear conversation memory
#   'quit' - Exit
Example 3: Batch Processing
bash
# Create questions file
echo '["What is machine learning?", "How does deep learning work?"]' > questions.json

# Process batch
python rag_with_sources.py --mode batch --folder ./documents --questions questions.json --output answers.json
Example 4: Web Scraping RAG
python
# Load web pages
documents = rag.load_web_page("https://en.wikipedia.org/wiki/Artificial_intelligence")
rag.create_vector_store()

# Ask questions about web content
answer = rag.ask_question("What is the history of AI?")
🔧 Configuration Options
System Configuration
python
RAGDocumentQASystem(
    persist_directory="./data/chroma_db",  # Storage location
    embedding_model="openai",              # "openai" or "huggingface"
    model_name="gpt-4",                    # OpenAI model
    chunk_size=1000,                       # Characters per chunk
    chunk_overlap=200,                     # Overlap between chunks
)
QA Chain Configuration
python
rag.initialize_qa_chain(
    chain_type="stuff",           # "stuff", "map_reduce", "refine", "map_rerank"
    return_source_documents=True, # Include source documents
    temperature=0.1,              # LLM temperature (0.0-2.0)
    streaming=False,              # Enable streaming responses
)
📈 Performance Optimization
1. Chunk Size Optimization
Small chunks (500-1000 chars): Better for precise retrieval

Large chunks (2000-4000 chars): Better for context preservation

Overlap (10-20%): Prevents context loss

2. Retrieval Parameters
python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 4,                    # Number of chunks to retrieve
        "score_threshold": 0.7,    # Minimum similarity score
        "filter": {"source": "important.pdf"}  # Metadata filter
    }
)
3. Cost Optimization
Use smaller models (gpt-3.5-turbo) for development

Implement caching for frequent queries

Monitor token usage with callbacks

🧪 Testing Your RAG System
Test Questions
python
test_questions = [
    "What is the main topic of this document?",
    "Can you summarize the key points?",
    "What evidence supports the conclusions?",
    "Are there any limitations mentioned?",
    "What recommendations are provided?"
]
Evaluation Metrics
Answer Relevance: Does the answer address the question?

Source Accuracy: Are citations correct?

Response Time: How long does it take?

Token Efficiency: Cost per answer

🐛 Troubleshooting
Common Issues
"No documents loaded"

Check file paths exist

Verify file formats are supported

Check encoding for text files

Poor answer quality

Adjust chunk size and overlap

Increase retrieval count (k parameter)

Try different embedding models

High latency

Reduce chunk size

Use smaller LLM model

Implement caching

API Key errors

Verify OPENAI_API_KEY is set

Check API key permissions

Ensure sufficient credits

Debug Mode
python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Test individual components
chunks = rag.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Check vector store
stats = rag.get_document_stats()
print(stats)
📚 Additional Resources
Documentation
LangChain Documentation

ChromaDB Documentation

OpenAI Embeddings

Research Papers
Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks

Dense Passage Retrieval for Open-Domain Question Answering

Tools & Libraries
LlamaIndex: Alternative to LangChain

Weaviate: Cloud-native vector database

Pinecone: Managed vector database service

🎯 Next Steps
Week 5-6, Days 31-33: Advanced RAG Techniques
Hybrid Search: Combine semantic and keyword search

Query Expansion: Improve search with query rewriting

Multi-document RAG: Handle multiple document collections

Re-ranking: Improve result quality with second-stage ranking

Production Deployment
API Development: Create REST API for RAG system

Docker Containerization: Package for deployment

Monitoring: Track performance and usage

Scaling: Handle multiple users and documents

📊 Project Ideas
Beginner Projects
Document Summarizer: Summarize long documents

Research Assistant: Answer questions about research papers

FAQ Generator: Create FAQs from documentation

Intermediate Projects
Legal Document Analyzer: Answer questions about legal documents

Medical Literature Assistant: Help with medical research

Academic Paper Reviewer: Analyze and critique papers

Advanced Projects
Enterprise Knowledge Base: Company-wide RAG system

Multi-modal RAG: Combine text with images/tables

Real-time RAG: Process streaming documents

🤝 Contributing
Found a bug or have a feature request? Please open an issue or submit a pull request.

📄 License
This project is part of the AI Developer Roadmap. Use it for learning and building your own AI applications!

Happy Building! 🚀

*Next: Days 31-33 - Advanced RAG Techniques (Hybrid Search, Re-ranking, Multi-doc)*