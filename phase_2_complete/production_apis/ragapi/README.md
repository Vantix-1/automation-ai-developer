# 🔍 RAG System API

## 📋 Overview
This section covers building Retrieval-Augmented Generation (RAG) systems with document processing, vector search, and AI-powered question answering. You'll learn to create production-ready document intelligence APIs.

## 📁 File Structure
```
day_40_42_ragapi/
├── requirements.txt         # Python dependencies
├── rag_api.py              # Complete RAG system with document processing
├── file_upload_api.py      # Advanced file upload with chunking
└── api_security.py         # Comprehensive security implementation
```

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.11+
- OpenAI API key (for embeddings and LLM)
- ChromaDB (vector database)
- Sentence transformers (optional, for local embeddings)

### Installation
```bash
# Navigate to the directory
cd phase_2_complete/week_7_8_production/day_40_42_ragapi

# Install dependencies
pip install -r requirements.txt

# Install sentence transformers (optional, for local embeddings)
pip install sentence-transformers

# Set up environment variables
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "SECRET_KEY=your_secret_key_here" >> .env
```

## 📚 Learning Objectives

### Day 40: Document Processing
- ✅ Upload and process various document types (PDF, TXT, etc.)
- ✅ Implement text chunking and splitting strategies
- ✅ Create vector embeddings for semantic search
- ✅ Set up vector database (ChromaDB)

### Day 41: RAG Implementation
- ✅ Implement retrieval-augmented generation
- ✅ Create semantic search capabilities
- ✅ Build question-answering system with citations
- ✅ Manage document collections

### Day 42: Security & Production
- ✅ Implement JWT authentication
- ✅ Add rate limiting and security headers
- ✅ Create comprehensive file upload system
- ✅ Implement input validation and sanitization

## 🚦 Quick Start

### 1. Run the RAG API
```bash
python rag_api.py
```

Access the API at:
- **API:** http://localhost:8000
- **Documentation:** http://localhost:8000/docs
- **Default token:** rag-api-token

### 2. Upload Documents
```bash
# Upload a PDF document
curl -X POST http://localhost:8000/upload \
  -H "Authorization: Bearer rag-api-token" \
  -F "file=@/path/to/document.pdf" \
  -F "collection_name=research_papers"

# Response includes collection_id for querying
```

### 3. Query Documents
```bash
# Search documents
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer rag-api-token" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "collection_id": "your_collection_id",
    "top_k": 5,
    "include_sources": true
  }'
```

### 4. Run File Upload API
```bash
python file_upload_api.py
# Test uploads at: http://localhost:8000/docs
```

### 5. Test Security Features
```bash
python api_security.py
# Test endpoints with: http://localhost:8000/docs
```

## 🔧 Key Features

### 1. Document Processing
- Support for multiple file types (PDF, TXT, CSV, MD)
- Automatic text extraction and chunking
- Metadata extraction and storage
- Batch processing capabilities

### 2. Vector Search & Embeddings
- OpenAI embeddings integration
- Local sentence transformer support (fallback)
- Semantic similarity search
- Hybrid search (keyword + semantic)

### 3. RAG Pipeline
- Context retrieval from documents
- AI-powered answer generation
- Source citation and attribution
- Confidence scoring

### 4. File Upload System
- Chunked uploads for large files
- Resume capabilities
- Progress tracking
- File validation and virus scanning (conceptual)

### 5. Security Implementation
- JWT authentication with refresh tokens
- Password hashing with bcrypt
- Rate limiting and request throttling
- SQL injection and XSS protection
- Security headers and CORS

## 📖 Code Examples

### Upload Document
```python
import requests

url = "http://localhost:8000/upload"
headers = {"Authorization": "Bearer rag-api-token"}

with open("document.pdf", "rb") as f:
    files = {"file": ("document.pdf", f, "application/pdf")}
    data = {"collection_name": "research"}
    
    response = requests.post(url, headers=headers, files=files, data=data)
    print(response.json())
```

### Query RAG System
```python
import requests

url = "http://localhost:8000/query"
headers = {
    "Authorization": "Bearer rag-api-token",
    "Content-Type": "application/json"
}

payload = {
    "query": "What is machine learning?",
    "collection_id": "research",
    "top_k": 3,
    "include_sources": True
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()

print(f"Answer: {result['answer']}")
print(f"\nSources:")
for source in result['sources']:
    print(f"- {source['metadata']['filename']}")
    print(f"  {source['content'][:200]}...")
```

### Chunked File Upload
```python
import requests

def upload_large_file(file_path, chunk_size=1024*1024):  # 1MB chunks
    # Initialize upload
    init_response = requests.post(
        "http://localhost:8000/upload/chunked/init",
        data={
            "filename": "large_file.pdf",
            "file_size": os.path.getsize(file_path),
            "chunk_size": chunk_size
        },
        headers={"Authorization": "Bearer upload-token"}
    )
    
    upload_id = init_response.json()["upload_id"]
    
    # Upload chunks
    with open(file_path, "rb") as f:
        chunk_index = 0
        while chunk := f.read(chunk_size):
            files = {"chunk": (f"chunk_{chunk_index}", chunk)}
            data = {
                "upload_id": upload_id,
                "chunk_index": chunk_index,
                "total_chunks": -1  # Will be calculated server-side
            }
            
            response = requests.post(
                f"http://localhost:8000/upload/chunked/{upload_id}",
                files=files,
                data=data,
                headers={"Authorization": "Bearer upload-token"}
            )
            
            chunk_index += 1
            print(f"Uploaded chunk {chunk_index}: {response.json()}")
```

## 🧪 Testing Examples

### Test Document Processing
```python
from rag_api import process_document

# Test processing different file types
chunks = process_document("document.pdf", "document.pdf")
print(f"Created {len(chunks)} chunks")
print(f"First chunk: {chunks[0].page_content[:200]}...")
```

### Test Security Features
```python
from api_security import hash_password, verify_password

# Test password hashing
hashed = hash_password("SecurePass123!")
print(f"Hashed password: {hashed}")

# Test verification
is_valid = verify_password("SecurePass123!", hashed)
print(f"Password valid: {is_valid}")
```

## 🔍 API Reference

### Authentication
```
Authorization: Bearer your_token_here
```

### Main RAG Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | API information |
| GET | /health | Health check |
| POST | /upload | Upload document |
| POST | /query | Query documents |
| POST | /search | Semantic search |
| GET | /collections | List collections |
| GET | /collections/{id} | Get collection |
| DELETE | /collections/{id} | Delete collection |
| GET | /documents | List documents |

### File Upload Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload/single | Upload single file |
| POST | /upload/multiple | Upload multiple files |
| POST | /upload/chunked/init | Initialize chunked upload |
| POST | /upload/chunked/{id} | Upload chunk |
| GET | /upload/status/{id} | Get upload status |
| GET | /download/{id} | Download file |

### Security Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /auth/register | Register user |
| POST | /auth/login | Login and get tokens |
| POST | /auth/refresh | Refresh access token |
| POST | /auth/logout | Logout and blacklist token |
| GET | /secure/data | Get secure data |
| POST | /secure/data | Post secure data |

## 🎯 Best Practices Implemented

### Document Processing
- **Chunking Strategy:** Optimal chunk sizes with overlap
- **Metadata Preservation:** Keep document metadata with chunks
- **Error Handling:** Graceful handling of malformed documents
- **Progress Tracking:** Real-time upload progress

### RAG System
- **Source Citation:** Always include source documents
- **Confidence Scoring:** Return similarity scores
- **Hybrid Search:** Combine semantic and keyword search
- **Re-ranking:** Improve result quality with re-ranking

### Security
- **Defense in Depth:** Multiple security layers
- **Input Validation:** Validate and sanitize all inputs
- **Secure Defaults:** Safe configurations by default
- **Audit Logging:** Log security-relevant events

## 📈 Performance Optimization

### 1. Embedding Caching
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> List[float]:
    text_hash = hashlib.md5(text.encode()).hexdigest()
    # Check cache, compute if missing
    return compute_embedding(text)
```

### 2. Batch Processing
```python
# Process documents in batches
def process_documents_batch(documents: List[Document], batch_size: int = 10):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        # Process batch concurrently
        yield from process_batch_concurrently(batch)
```

### 3. Async Operations
```python
async def process_upload_async(file_path: str):
    # Use async for I/O operations
    async with aiofiles.open(file_path, "rb") as f:
        content = await f.read()
        # Process asynchronously
        chunks = await asyncio.to_thread(process_document, content)
        return chunks
```

## 🚨 Troubleshooting

### Common Issues

**ChromaDB connection errors**
```bash
# Ensure ChromaDB is installed
pip install chromadb

# Check persistence directory permissions
chmod 755 ./chroma_db
```

**Embedding model errors**
```bash
# Test OpenAI API
curl https://api.openai.com/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input": "test", "model": "text-embedding-3-small"}'

# Or use local embeddings
pip install sentence-transformers
```

**Large file upload issues**
```python
# Increase timeout for large files
import requests
requests.post(url, files=files, timeout=300)  # 5 minute timeout
```

## 📚 Resources
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)
- [RAG Best Practices](https://www.anthropic.com/index/retrieval-augmented-generation)
- [FastAPI File Uploads](https://fastapi.tiangolo.com/tutorial/request-files/)

## 🏆 Completion Checklist
- [ ] Implemented document processing pipeline
- [ ] Created vector embeddings and search
- [ ] Built complete RAG system
- [ ] Added file upload with chunking
- [ ] Implemented comprehensive security
- [ ] Added authentication and authorization
- [ ] Created error handling and validation
- [ ] Implemented performance optimizations