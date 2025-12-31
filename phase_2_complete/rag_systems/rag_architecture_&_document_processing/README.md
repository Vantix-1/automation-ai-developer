# 📚 RAG Architecture & Document Processing

**Objective:** Master document processing pipelines and vector database setup for RAG systems

## 🎯 Learning Objectives

### **Day 25: Document Loading**
- ✅ Load documents from multiple formats (PDF, DOCX, TXT, CSV, PPTX)
- ✅ Implement parallel document processing
- ✅ Handle web content and directory structures
- ✅ Create a document loader factory system

### **Day 26: Text Splitting**
- ✅ Implement 6+ chunking strategies
- ✅ Handle structured and unstructured text
- ✅ Optimize for semantic coherence
- ✅ Create token-aware splitting for LLMs

### **Day 27: Vector Store Setup**
- ✅ Configure multiple vector databases (ChromaDB, FAISS, Simple)
- ✅ Implement document indexing with metadata
- ✅ Create backup and restore functionality
- ✅ Build a vector store manager for multiple collections

## 📁 Files

### **document_loader.py**
Production-ready document loading system:
- Support for 7+ file formats
- Parallel processing with ThreadPoolExecutor
- Web content extraction
- Directory scanning with filters
- Factory pattern for loader creation

### **text_splitting.py**
Advanced text chunking strategies:
- Fixed-size, Sentence, Paragraph, Recursive splitting
- Semantic and Sliding Window chunking
- Token-aware splitting for LLM context windows
- Performance benchmarking and comparison

### **vector_store_setup.py**
Multi-backend vector database system:
- ChromaDB, FAISS, and Simple store implementations
- Document indexing with rich metadata
- Backup and restore functionality
- Collection management and statistics

## 🚀 Quick Start

```bash
cd week5-6_day25-27

# Install required packages
pip install pypdf python-docx python-pptx beautifulsoup4 requests
pip install chromadb faiss-cpu sentence-transformers

# Test document loading
python document_loader.py

# Test text splitting
python text_splitting.py

# Test vector stores
python vector_store_setup.py
```

## 📚 Key Concepts

### 1. Document Processing Pipeline

```
Documents → Load → Clean → Split → Embed → Store → Index
    ↑           ↓
 Formats     Chunks
    ↑           ↓
 Sources    Metadata
```

### 2. Chunking Strategies

- **Fixed Size:** Simple character-based chunks
- **Sentence:** Maintains sentence boundaries
- **Paragraph:** Preserves document structure
- **Recursive:** Hierarchical splitting with multiple separators
- **Semantic:** Uses embeddings to find natural breakpoints
- **Sliding Window:** Overlapping chunks for context preservation

### 3. Vector Store Options

- **ChromaDB:** Production-ready, persistent storage
- **FAISS:** High-performance similarity search
- **Simple Store:** Lightweight, in-memory for testing
- **Hybrid:** Combine multiple stores for different use cases

## 🛠️ Installation

### Core Packages

```bash
# Document processing
pip install pypdf>=3.17.4,<4.0.0
pip install python-docx>=0.8.11,<1.0.0
pip install python-pptx>=0.6.21,<1.0.0
pip install beautifulsoup4>=4.12.0,<5.0.0
pip install requests>=2.31.0,<3.0.0

# Vector databases
pip install chromadb>=0.4.18,<0.5.0
pip install faiss-cpu>=1.7.4,<2.0.0
pip install sentence-transformers>=2.2.2,<3.0.0

# Text processing
pip install nltk>=3.8.1,<4.0.0
pip install tiktoken>=0.5.2,<0.6.0
```

### Optional Packages

```bash
# For advanced document parsing
pip install unstructured>=0.12.2,<0.13.0

# For GPU acceleration (if available)
pip install faiss-gpu>=1.7.4,<2.0.0

# For visualization
pip install matplotlib>=3.7.0,<4.0.0
```

## 📊 Example Usage

### Document Loading

```python
from document_loader import DocumentLoaderFactory, DirectoryLoader

# Load single document
loader = DocumentLoaderFactory.create_loader("document.pdf")
documents = loader.load()

# Load entire directory
dir_loader = DirectoryLoader("./documents", recursive=True)
all_documents = dir_loader.load()
```

### Text Splitting

```python
from text_splitting import TextSplitterFactory, ChunkingMethod

# Create semantic splitter
splitter = TextSplitterFactory.create_splitter(
    ChunkingMethod.SEMANTIC,
    chunk_size=1000,
    chunk_overlap=200
)

# Split documents
chunks = splitter.split_text(long_document)
```

### Vector Store Setup

```python
from vector_store_setup import VectorStoreFactory, VectorStoreConfig

# Configure store
config = VectorStoreConfig(
    store_type="chromadb",
    persist_directory="./vector_store",
    embedding_model="all-MiniLM-L6-v2"
)

# Create and use store
store = VectorStoreFactory.create_store(config)
store.add_documents(indexed_documents)
results = store.search("AI in healthcare", top_k=5)
```

## 🔧 Code Examples

### Complete Processing Pipeline

```python
from document_loader import DirectoryLoader
from text_splitting import RecursiveSplitter
from vector_store_setup import VectorStoreFactory, VectorStoreConfig
from sentence_transformers import SentenceTransformer

# 1. Load documents
loader = DirectoryLoader("./data")
documents = loader.load()

# 2. Split documents
splitter = RecursiveSplitter(chunk_size=500, chunk_overlap=100)
all_chunks = []
for doc in documents:
    chunks = splitter.split_text(doc.content)
    all_chunks.extend(chunks)

# 3. Setup vector store
config = VectorStoreConfig(store_type="chromadb")
store = VectorStoreFactory.create_store(config)

# 4. Index chunks
for chunk in all_chunks:
    metadata = DocumentMetadata(
        source=chunk.metadata.get('source'),
        title=chunk.metadata.get('title'),
        word_count=chunk.token_count
    )
    
    indexed_doc = IndexedDocument.from_text(
        text=chunk.text,
        metadata=metadata,
        embedding_model=store.embedding_model
    )
    store.add_documents([indexed_doc])
```

### Multi-Store Architecture

```python
from vector_store_setup import VectorStoreManager

# Create manager for multiple collections
manager = VectorStoreManager(base_config)

# Work with different collections
research_store = manager.get_store("research_papers")
news_store = manager.get_store("news_articles")
code_store = manager.get_store("code_documentation")

# Backup all collections
backups = manager.backup_all("./backups")
```

## 📈 Performance Optimization

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def process_documents_parallel(documents, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for doc in documents:
            future = executor.submit(process_single_document, doc)
            futures.append(future)
        
        results = []
        for future in as_completed(futures):
            results.extend(future.result())
        
        return results
```

### Batch Processing

```python
# Process embeddings in batches
batch_size = 32
embeddings = []

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    batch_embeddings = model.encode(batch)
    embeddings.extend(batch_embeddings)
```

### Memory Management

```python
# Use generators for large datasets
def document_generator(directory):
    for file_path in Path(directory).glob("**/*.txt"):
        with open(file_path, 'r') as f:
            yield f.read()

# Process one document at a time
for document in document_generator("./large_dataset"):
    chunks = splitter.split_text(document)
    # Process chunks incrementally
```

## 🧪 Testing

### Unit Tests

```bash
# Test document loading
python -m pytest test_document_loader.py -v

# Test text splitting
python -m pytest test_text_splitting.py -v

# Test vector stores
python -m pytest test_vector_store.py -v
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete RAG pipeline"""
    # 1. Load documents
    documents = load_documents("./test_data")
    
    # 2. Split into chunks
    chunks = split_documents(documents)
    
    # 3. Create embeddings
    embeddings = create_embeddings(chunks)
    
    # 4. Store in vector database
    store = create_vector_store(embeddings, chunks)
    
    # 5. Test retrieval
    results = store.search("test query", top_k=5)
    
    assert len(results) > 0
    assert all('score' in r for r in results)
```

## 📚 Additional Resources

### Documentation

- [PyPDF Documentation](https://pypdf.readthedocs.io/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers Documentation](https://www.sbert.net/)

### Best Practices

- **Document Loading:** Always validate file formats and handle encoding issues
- **Text Splitting:** Choose strategy based on document type and use case
- **Chunk Size:** Balance between context preservation and retrieval accuracy
- **Metadata:** Store rich metadata for better filtering and debugging
- **Backup:** Regular backups of vector stores are essential

### Production Considerations

- **Scalability:** Use batch processing and parallelization for large datasets
- **Memory:** Monitor memory usage, especially with large embeddings
- **Persistence:** Choose appropriate storage backend for your use case
- **Monitoring:** Track indexing performance and search accuracy

## 🎯 Success Checklist

### Day 25 Complete When:

- ✅ Can load documents from multiple formats
- ✅ Implement parallel processing for efficiency
- ✅ Handle web content and directory structures
- ✅ Understand factory pattern for extensibility

### Day 26 Complete When:

- ✅ Implement all 6 chunking strategies
- ✅ Understand trade-offs between different methods
- ✅ Can choose appropriate strategy for different document types
- ✅ Optimize chunking for semantic coherence

### Day 27 Complete When:

- ✅ Configure and use multiple vector databases
- ✅ Implement document indexing with metadata
- ✅ Create backup and restore functionality
- ✅ Manage multiple collections effectively

## 🚀 Next Steps

After completing Days 25-27:

1. **Review the pipeline** - Understand how all components work together
2. **Experiment with configurations** - Try different chunking strategies and vector stores
3. **Test with real documents** - Use your own documents or datasets
4. **Prepare for Days 28-30** - Build your first complete RAG system

Start with `document_loader.py` to build your document processing foundation!

## 📁 Complete Days 25-27 Structure

```
week5-6_day25-27/
├── document_loader.py      # Day 25: Multi-format document processing
├── text_splitting.py       # Day 26: Advanced text chunking strategies
├── vector_store_setup.py   # Day 27: Vector database configuration
└── README.md               # Complete documentation
```

## 🎯 What You've Built (Days 25-27)

### ✅ Complete Document Processing Pipeline:

1. **Multi-format Document Loading** - PDF, DOCX, TXT, CSV, PPTX, web pages
2. **Parallel Processing** - Efficient handling of large document collections
3. **Directory Scanning** - Recursive loading with file type filters
4. **Web Content Extraction** - BeautifulSoup integration for web pages

### ✅ Advanced Text Splitting System:

1. **6 Chunking Strategies** - Fixed, Sentence, Paragraph, Recursive, Semantic, Sliding Window
2. **Token-aware Splitting** - Accurate token counting for LLM context windows
3. **Performance Optimization** - Benchmarking and comparison tools
4. **Semantic Coherence** - Embedding-based natural breakpoint detection

### ✅ Production Vector Store Management:

1. **Multi-backend Support** - ChromaDB, FAISS, and Simple stores
2. **Rich Metadata Support** - Comprehensive document metadata tracking
3. **Backup & Restore** - Full backup functionality for data safety
4. **Collection Management** - Multiple collections with separate configurations
5. **Statistics & Monitoring** - Comprehensive performance tracking