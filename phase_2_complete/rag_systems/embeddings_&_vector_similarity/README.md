# 📊 Embeddings & Vector Similarity

**Objective:** Master embedding concepts, vector mathematics, and semantic search implementation

## 🎯 Learning Objectives

### **Day 22: Embedding Fundamentals**
- ✅ Understand what embeddings are
- ✅ Learn different embedding models
- ✅ Practice creating embeddings
- ✅ Visualize embedding spaces

### **Day 23: Vector Mathematics**
- ✅ Master cosine similarity
- ✅ Understand Euclidean distance
- ✅ Practice vector operations
- ✅ Implement similarity calculations

### **Day 24: Semantic Search**
- ✅ Build semantic search engine
- ✅ Implement nearest neighbor search
- ✅ Practice with real datasets
- ✅ Optimize search performance

## 📁 Files

### **embeddings_demo.py**
Complete demonstration of embedding models:
- OpenAI embeddings
- Sentence Transformers (local)
- Hugging Face models
- Embedding visualization

### **vector_similarity.py**
Vector mathematics and similarity calculations:
- Cosine similarity implementation
- Euclidean distance
- Dot product calculations
- Batch similarity computations
- Performance benchmarking

### **semantic_search.py**
Production-ready semantic search engine:
- Multiple indexing backends (Simple, FAISS, ChromaDB)
- Document indexing and management
- Efficient similarity search
- Caching and performance optimization
- Interactive search interface

## 🚀 Quick Start

```bash
cd week5-6_day22-24

# Install dependencies (if not already installed)
pip install sentence-transformers chromadb faiss-cpu numpy scikit-learn matplotlib

# Run embeddings demo
python embeddings_demo.py

# Test vector similarity
python vector_similarity.py

# Build semantic search engine
python semantic_search.py
```

## 📚 Key Concepts

### 1. What are Embeddings?
- Numerical representations of text
- Capture semantic meaning
- Enable mathematical operations on text
- Dense vectors (e.g., 384, 768, 1536 dimensions)

### 2. Embedding Models
- **OpenAI:** text-embedding-ada-002 (1536 dim)
- **Sentence Transformers:** all-MiniLM-L6-v2 (384 dim)
- **Google:** Universal Sentence Encoder
- **Custom:** Train your own embeddings

### 3. Similarity Metrics
- **Cosine Similarity:** Angle between vectors (0 to 1)
- **Euclidean Distance:** Straight-line distance
- **Dot Product:** Magnitude-aware similarity
- **Manhattan Distance:** Grid-based distance

### 4. Semantic Search Architecture
```
Text Documents → Embeddings → Vector Store → Index
          ↓
        Query → Embedding → Similarity Search → Results
```

## 🛠️ Installation

### Required Packages
```bash
# Core packages
pip install sentence-transformers>=2.2.2,<3.0.0
pip install chromadb>=0.4.18,<0.5.0
pip install faiss-cpu>=1.7.4,<2.0.0

# Utilities
pip install numpy>=1.24.3,<2.0.0
pip install scikit-learn>=1.3.0,<2.0.0
pip install matplotlib>=3.7.0,<4.0.0

# Optional (for embeddings_demo.py)
pip install openai>=1.3.0,<2.0.0
pip install transformers>=4.36.0,<5.0.0
```

### Verifying Installation
```bash
python -c "
import sentence_transformers, chromadb, faiss, numpy
print('✅ All core packages installed')
print(f'Sentence Transformers: {sentence_transformers.__version__}')
print(f'ChromaDB: {chromadb.__version__}')
print(f'NumPy: {numpy.__version__}')
"
```

## 📊 Example Usage

### Create Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world", "Good morning"])
print(f"Embedding shape: {embeddings.shape}")  # (2, 384)
```

### Calculate Similarity
```python
from vector_similarity import VectorSimilarityCalculator

calculator = VectorSimilarityCalculator()
vec1 = np.random.randn(100)
vec2 = np.random.randn(100)

result = calculator.cosine_similarity_custom(vec1, vec2)
print(f"Cosine similarity: {result:.4f}")
```

### Semantic Search
```python
from semantic_search import SemanticSearchEngine

# Initialize engine
engine = SemanticSearchEngine(model_name="all-MiniLM-L6-v2")

# Index documents
documents = [
    {'text': 'Python is a programming language', 'metadata': {'category': 'programming'}},
    {'text': 'Machine learning is a subset of AI', 'metadata': {'category': 'ai'}}
]
engine.index_documents(documents)

# Search
results = engine.search("programming languages", top_k=5)
for result in results:
    print(f"Score: {result.score:.3f}, Text: {result.text[:50]}...")
```

## 🔧 Code Examples

### Basic Embedding Creation
```python
import numpy as np
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    input="Your text here",
    model="text-embedding-ada-002"
)
embedding = response.data[0].embedding
print(f"Embedding dimensions: {len(embedding)}")
```

### Vector Similarity Functions
```python
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2)

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))
```

### Simple Search Engine
```python
class SemanticSearch:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
    
    def add_documents(self, documents):
        self.documents.extend(documents)
        self.embeddings = self.model.encode(self.documents)
    
    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        indices = np.argsort(similarities)[::-1][:top_k]
        return [(self.documents[i], similarities[i]) for i in indices]
```

## 📈 Performance Optimization

### Batch Processing
```python
# Process documents in batches
batch_size = 32
embeddings = []
for i in range(0, len(documents), batch_size):
    batch = documents[i:i+batch_size]
    batch_embeddings = model.encode(batch)
    embeddings.extend(batch_embeddings)
```

### Memory Efficiency
```python
# Use 16-bit precision for memory savings
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
model.half()  # Convert to 16-bit
embeddings = model.encode(documents, precision='float16')
```

### Caching
```python
import hashlib
import pickle

def get_cached_embedding(text, model, cache_dir='./cache'):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    cache_path = f"{cache_dir}/{text_hash}.pkl"
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    embedding = model.encode([text])[0]
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(embedding, f)
    
    return embedding
```

## 🧪 Testing

### Unit Tests
```bash
# Run tests
python -m pytest test_embeddings.py -v

# Test with coverage
python -m pytest test_embeddings.py --cov=embeddings_demo --cov-report=html
```

### Benchmarking
```python
import time
from tqdm import tqdm

def benchmark_embeddings(model, documents, iterations=100):
    times = []
    for _ in tqdm(range(iterations)):
        start = time.time()
        model.encode(documents)
        times.append(time.time() - start)
    
    print(f"Average time: {np.mean(times):.4f}s")
    print(f"Documents per second: {len(documents)/np.mean(times):.1f}")
```

## 📚 Additional Resources

### Documentation
- [Sentence Transformers Documentation](https://www.sbert.net/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [Scikit-learn Similarity Metrics](https://scikit-learn.org/stable/modules/metrics.html)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [ChromaDB Documentation](https://docs.trychroma.com/)

### Research Papers
- "Attention is All You Need" (Transformer architecture)
- "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- "Efficient Neural Architecture Search for BERT"

### Tools
- **UMAP/t-SNE:** Embedding visualization
- **Weights & Biases:** Experiment tracking
- **Milvus/Weaviate:** Alternative vector databases

## 🎯 Success Checklist

### Day 22 Complete When:
- ✅ Can create embeddings with multiple models
- ✅ Understand embedding dimensions and quality tradeoffs
- ✅ Can visualize embeddings in 2D/3D space
- ✅ Know when to use different embedding models

### Day 23 Complete When:
- ✅ Can implement cosine similarity from scratch
- ✅ Understand different distance metrics
- ✅ Can perform vector operations efficiently
- ✅ Know when to use which similarity metric

### Day 24 Complete When:
- ✅ Built a working semantic search engine
- ✅ Can optimize search for performance
- ✅ Understand indexing strategies
- ✅ Can evaluate search quality