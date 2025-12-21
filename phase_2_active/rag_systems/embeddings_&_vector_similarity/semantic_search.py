"""
🔍 Semantic Search Engine
Day 24: Building production-ready semantic search with vector databases
"""

import os
import numpy as np
import json
import pickle
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Try to import required packages
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    print("⚠️ ChromaDB not installed. Install with: pip install chromadb")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("⚠️ FAISS not installed. Install with: pip install faiss-cpu")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️ Sentence Transformers not installed. Install with: pip install sentence-transformers")

load_dotenv()

@dataclass
class SearchResult:
    """Represents a single search result"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        return data

@dataclass
class SearchQuery:
    """Represents a search query"""
    text: str
    embedding: np.ndarray
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 10
    
    @classmethod
    def from_text(cls, text: str, model, filters: Optional[Dict] = None, top_k: int = 10):
        """Create search query from text"""
        embedding = model.encode([text])[0]
        return cls(text=text, embedding=embedding, filters=filters, top_k=top_k)

class DocumentIndex:
    """Manages document indexing and storage"""
    
    def __init__(self, index_dir: str = "./search_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(exist_ok=True)
        
        # Store documents and embeddings
        self.documents = []  # List of (id, text, metadata)
        self.embeddings = None  # Numpy array of embeddings
        self.document_ids = []
        
        # Load existing index if available
        self.load()
    
    def add_documents(self, documents: List[Tuple[str, Dict[str, Any]]], 
                     embeddings: np.ndarray):
        """Add documents with embeddings to index"""
        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents must match number of embeddings")
        
        for i, (text, metadata) in enumerate(documents):
            # Generate unique ID
            doc_id = hashlib.md5(f"{text}{datetime.now().timestamp()}".encode()).hexdigest()[:16]
            
            # Store document
            self.documents.append((doc_id, text, metadata))
            self.document_ids.append(doc_id)
        
        # Store embeddings
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Save index
        self.save()
        
        return self.document_ids[-len(documents):]
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10, 
                      filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search for similar documents using brute-force cosine similarity"""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Calculate cosine similarities
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        
        # Normalize query and document embeddings
        query_normalized = query_embedding / query_norm
        
        # Normalize all document embeddings
        doc_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        doc_norms[doc_norms == 0] = 1  # Avoid division by zero
        docs_normalized = self.embeddings / doc_norms
        
        # Calculate cosine similarities
        similarities = np.dot(docs_normalized, query_normalized)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter results if filters provided
        if filters:
            filtered_indices = []
            for idx in top_indices:
                doc_id, _, metadata = self.documents[idx]
                if self._matches_filters(metadata, filters):
                    filtered_indices.append(idx)
            top_indices = filtered_indices[:top_k]
        
        # Create results
        results = []
        for idx in top_indices:
            doc_id, text, metadata = self.documents[idx]
            results.append(SearchResult(
                id=doc_id,
                text=text,
                score=float(similarities[idx]),
                metadata=metadata,
                embedding=self.embeddings[idx]
            ))
        
        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches filters"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    def save(self):
        """Save index to disk"""
        index_path = self.index_dir / "index.pkl"
        
        index_data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'document_ids': self.document_ids,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"💾 Index saved: {len(self.documents)} documents")
    
    def load(self):
        """Load index from disk"""
        index_path = self.index_dir / "index.pkl"
        
        if index_path.exists():
            with open(index_path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data['documents']
            self.embeddings = index_data['embeddings']
            self.document_ids = index_data['document_ids']
            
            print(f"📖 Index loaded: {len(self.documents)} documents")
        else:
            print("📝 No existing index found, starting fresh")

class FAISSIndex(DocumentIndex):
    """Document index using FAISS for efficient similarity search"""
    
    def __init__(self, index_dir: str = "./faiss_index", dimension: int = 384):
        super().__init__(index_dir)
        
        if not HAS_FAISS:
            raise ImportError("FAISS not installed")
        
        self.dimension = dimension
        self.index = None
        self._build_index()
    
    def _build_index(self):
        """Build or load FAISS index"""
        index_path = self.index_dir / "faiss_index.bin"
        
        if self.embeddings is not None and len(self.embeddings) > 0:
            if index_path.exists():
                # Load existing index
                self.index = faiss.read_index(str(index_path))
                print(f"📖 FAISS index loaded: {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
                
                # Add existing embeddings if any
                if self.embeddings is not None:
                    # Normalize embeddings for cosine similarity
                    faiss.normalize_L2(self.embeddings)
                    self.index.add(self.embeddings)
                    print(f"🔨 FAISS index created: {self.index.ntotal} vectors")
    
    def add_documents(self, documents: List[Tuple[str, Dict[str, Any]]], 
                     embeddings: np.ndarray):
        """Add documents with embeddings to FAISS index"""
        # Add to parent index
        doc_ids = super().add_documents(documents, embeddings)
        
        # Add to FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings.copy()
        faiss.normalize_L2(embeddings_normalized)
        
        # Add to FAISS
        self.index.add(embeddings_normalized)
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_dir / "faiss_index.bin"))
        
        return doc_ids
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10,
                      filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search using FAISS"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Normalize query embedding
        query_normalized = query_embedding.copy().reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query_normalized)
        
        # Search in FAISS
        distances, indices = self.index.search(query_normalized, min(top_k * 2, self.index.ntotal))
        
        # Convert to results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            doc_id, text, metadata = self.documents[idx]
            
            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            results.append(SearchResult(
                id=doc_id,
                text=text,
                score=float(distances[0][i]),
                metadata=metadata,
                embedding=self.embeddings[idx]
            ))
            
            if len(results) >= top_k:
                break
        
        return results

class ChromaDBIndex:
    """Document index using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chromadb_index", 
                 collection_name: str = "documents"):
        if not HAS_CHROMADB:
            raise ImportError("ChromaDB not installed")
        
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        print(f"🔗 ChromaDB connected: {self.collection.count()} documents")
    
    def add_documents(self, documents: List[Tuple[str, Dict[str, Any]]], 
                     embeddings: np.ndarray):
        """Add documents to ChromaDB"""
        ids = []
        texts = []
        metadatas = []
        embeddings_list = []
        
        for i, (text, metadata) in enumerate(documents):
            # Generate unique ID
            doc_id = hashlib.md5(f"{text}{datetime.now().timestamp()}".encode()).hexdigest()[:16]
            
            ids.append(doc_id)
            texts.append(text)
            metadatas.append(metadata)
            embeddings_list.append(embeddings[i].tolist())
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
        
        print(f"✅ Added {len(documents)} documents to ChromaDB")
        return ids
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 10,
                      filters: Optional[Dict] = None) -> List[SearchResult]:
        """Search in ChromaDB"""
        # Convert filters to ChromaDB format
        where_filter = None
        if filters:
            where_filter = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    where_filter[key] = {"$in": value}
                else:
                    where_filter[key] = value
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to SearchResult objects
        search_results = []
        if results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                search_results.append(SearchResult(
                    id=doc_id,
                    text=results['documents'][0][i],
                    score=1.0 - results['distances'][0][i],  # Convert distance to similarity
                    metadata=results['metadatas'][0][i]
                ))
        
        return search_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name,
            "persist_directory": str(self.persist_directory)
        }

class SemanticSearchEngine:
    """Production-ready semantic search engine"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 index_type: str = "faiss",  # Options: "simple", "faiss", "chromadb"
                 cache_results: bool = True):
        
        # Load embedding model
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("Sentence Transformers not installed")
        
        print(f"🔤 Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Initialize index
        self.index_type = index_type
        if index_type == "simple":
            self.index = DocumentIndex("./simple_index")
        elif index_type == "faiss":
            self.index = FAISSIndex("./faiss_index", self.dimension)
        elif index_type == "chromadb":
            self.index = ChromaDBIndex("./chromadb_index")
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Setup caching
        self.cache_results = cache_results
        self.query_cache = {}
        self.cache_dir = Path("./search_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        print(f"🚀 Semantic Search Engine initialized")
        print(f"   Model: {model_name} ({self.dimension} dimensions)")
        print(f"   Index: {index_type}")
        print(f"   Cache: {'Enabled' if cache_results else 'Disabled'}")
    
    def index_documents(self, documents: List[Dict[str, Any]], 
                       batch_size: int = 32) -> List[str]:
        """Index a list of documents"""
        print(f"📚 Indexing {len(documents)} documents...")
        
        # Extract text and metadata
        doc_texts = []
        doc_metadatas = []
        
        for doc in documents:
            text = doc.get('text', '')
            metadata = doc.get('metadata', {})
            metadata['indexed_at'] = datetime.now().isoformat()
            
            if text:  # Only index non-empty documents
                doc_texts.append(text)
                doc_metadatas.append(metadata)
        
        # Process in batches
        all_doc_ids = []
        for i in range(0, len(doc_texts), batch_size):
            batch_texts = doc_texts[i:i+batch_size]
            batch_metadatas = doc_metadatas[i:i+batch_size]
            
            # Create embeddings
            print(f"   Creating embeddings for batch {i//batch_size + 1}...")
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
            
            # Add to index
            doc_pairs = [(text, metadata) for text, metadata in zip(batch_texts, batch_metadatas)]
            batch_doc_ids = self.index.add_documents(doc_pairs, batch_embeddings)
            all_doc_ids.extend(batch_doc_ids)
            
            print(f"   ✅ Batch {i//batch_size + 1} indexed: {len(batch_texts)} documents")
        
        print(f"📊 Total indexed: {len(all_doc_ids)} documents")
        return all_doc_ids
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None,
               use_cache: bool = True) -> List[SearchResult]:
        """Search for documents similar to query"""
        # Check cache
        cache_key = self._get_cache_key(query, top_k, filters)
        if use_cache and self.cache_results and cache_key in self.query_cache:
            print(f"💾 Using cached results for: '{query[:50]}...'")
            return self.query_cache[cache_key]
        
        # Create query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search in index
        print(f"🔍 Searching for: '{query[:50]}...'")
        results = self.index.search_similar(query_embedding, top_k, filters)
        
        # Cache results
        if use_cache and self.cache_results:
            self.query_cache[cache_key] = results
            self._save_to_cache(cache_key, results)
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = 10) -> List[List[SearchResult]]:
        """Search multiple queries efficiently"""
        print(f"🔍 Batch searching {len(queries)} queries...")
        
        # Create embeddings for all queries
        query_embeddings = self.model.encode(queries, show_progress_bar=True)
        
        # Search for each query
        all_results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for query, embedding in zip(queries, query_embeddings):
                future = executor.submit(self.index.search_similar, embedding, top_k)
                futures.append((query, future))
            
            for query, future in futures:
                try:
                    results = future.result(timeout=30)
                    all_results.append(results)
                    print(f"   ✅ '{query[:30]}...': {len(results)} results")
                except Exception as e:
                    print(f"   ❌ Error searching '{query[:30]}...': {e}")
                    all_results.append([])
        
        return all_results
    
    def _get_cache_key(self, query: str, top_k: int, filters: Optional[Dict]) -> str:
        """Generate cache key for query"""
        key_data = {
            'query': query,
            'top_k': top_k,
            'filters': filters or {},
            'model': self.model.get_sentence_embedding_dimension()
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _save_to_cache(self, cache_key: str, results: List[SearchResult]):
        """Save results to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'results': [r.to_dict() for r in results]
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    
    def clear_cache(self):
        """Clear search cache"""
        self.query_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        print("🧹 Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = {
            'model_dimension': self.dimension,
            'index_type': self.index_type,
            'cache_enabled': self.cache_results,
            'cache_size': len(self.query_cache),
            'cache_files': len(list(self.cache_dir.glob("*.json"))),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add index-specific stats
        if hasattr(self.index, 'get_stats'):
            stats.update(self.index.get_stats())
        
        return stats
    
    def export_results(self, results: List[SearchResult], format: str = 'json') -> str:
        """Export search results in specified format"""
        if format == 'json':
            return json.dumps([r.to_dict() for r in results], indent=2)
        elif format == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['ID', 'Score', 'Text', 'Metadata'])
            
            # Write rows
            for result in results:
                writer.writerow([
                    result.id,
                    result.score,
                    result.text[:100],  # Truncate for CSV
                    json.dumps(result.metadata)
                ])
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

def demo_semantic_search():
    """Demonstrate semantic search capabilities"""
    print("=" * 70)
    print("🔍 SEMANTIC SEARCH ENGINE DEMONSTRATION (Day 24)")
    print("=" * 70)
    
    # Create sample documents
    documents = [
        {
            'text': "Artificial intelligence is transforming every industry by automating complex tasks.",
            'metadata': {'category': 'technology', 'year': 2024, 'source': 'tech_blog'}
        },
        {
            'text': "Machine learning algorithms can analyze vast amounts of data to find patterns.",
            'metadata': {'category': 'technology', 'year': 2023, 'source': 'research_paper'}
        },
        {
            'text': "Deep learning models require powerful GPUs and large datasets for training.",
            'metadata': {'category': 'technology', 'year': 2024, 'source': 'conference'}
        },
        {
            'text': "Climate change is causing extreme weather events across the globe.",
            'metadata': {'category': 'environment', 'year': 2023, 'source': 'news'}
        },
        {
            'text': "Renewable energy sources like solar and wind are becoming more affordable.",
            'metadata': {'category': 'environment', 'year': 2024, 'source': 'report'}
        },
        {
            'text': "Python programming language is popular for data science and AI development.",
            'metadata': {'category': 'programming', 'year': 2024, 'source': 'tutorial'}
        },
        {
            'text': "JavaScript is essential for web development and interactive websites.",
            'metadata': {'category': 'programming', 'year': 2023, 'source': 'course'}
        },
        {
            'text': "Neural networks are inspired by the human brain's structure and function.",
            'metadata': {'category': 'technology', 'year': 2024, 'source': 'textbook'}
        },
        {
            'text': "Natural language processing enables computers to understand human language.",
            'metadata': {'category': 'technology', 'year': 2023, 'source': 'paper'}
        },
        {
            'text': "Sustainable agriculture practices can help reduce environmental impact.",
            'metadata': {'category': 'environment', 'year': 2024, 'source': 'magazine'}
        }
    ]
    
    # Test different index types
    index_types = []
    if HAS_SENTENCE_TRANSFORMERS:
        index_types.append("simple")
    if HAS_FAISS:
        index_types.append("faiss")
    if HAS_CHROMADB:
        index_types.append("chromadb")
    
    if not index_types:
        print("❌ No search indexes available. Install required packages.")
        return
    
    print(f"📊 Available index types: {index_types}")
    
    for index_type in index_types[:2]:  # Test first 2 index types
        print(f"\n{'='*60}")
        print(f"Testing {index_type.upper()} Index")
        print(f"{'='*60}")
        
        try:
            # Initialize search engine
            engine = SemanticSearchEngine(
                model_name="all-MiniLM-L6-v2",
                index_type=index_type,
                cache_results=True
            )
            
            # Index documents
            print("\n1️⃣ Indexing documents...")
            doc_ids = engine.index_documents(documents)
            print(f"   ✅ Indexed {len(doc_ids)} documents")
            
            # Test searches
            print("\n2️⃣ Testing searches...")
            
            test_queries = [
                "machine learning and artificial intelligence",
                "climate change and renewable energy",
                "programming languages for developers",
                "neural networks and deep learning"
            ]
            
            for query in test_queries:
                print(f"\n   🔍 Query: '{query}'")
                results = engine.search(query, top_k=3)
                
                if results:
                    for i, result in enumerate(results, 1):
                        print(f"     {i}. [{result.score:.3f}] {result.text[:60]}...")
                        print(f"        Category: {result.metadata.get('category', 'N/A')}, "
                              f"Year: {result.metadata.get('year', 'N/A')}")
                else:
                    print("     No results found")
            
            # Test with filters
            print("\n3️⃣ Testing with filters...")
            filters = {'category': 'technology', 'year': 2024}
            results = engine.search("AI and machine learning", top_k=5, filters=filters)
            print(f"   Filtered search (category=technology, year=2024): {len(results)} results")
            
            # Test batch search
            print("\n4️⃣ Testing batch search...")
            batch_results = engine.batch_search(test_queries, top_k=2)
            print(f"   Batch search completed: {len(batch_results)} query results")
            
            # Get stats
            print("\n5️⃣ Engine statistics:")
            stats = engine.get_stats()
            for key, value in stats.items():
                print(f"   {key}: {value}")
            
            # Clear cache
            engine.clear_cache()
            
        except Exception as e:
            print(f"❌ Error with {index_type} index: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Semantic Search Demonstration Complete!")
    print("📚 Next: Document Processing (Days 25-27)")

def interactive_search_demo():
    """Interactive search demonstration"""
    print("\n🎮 Interactive Search Demo")
    print("=" * 60)
    
    # Initialize engine
    try:
        engine = SemanticSearchEngine(
            model_name="all-MiniLM-L6-v2",
            index_type="simple",  # Use simple index for demo
            cache_results=True
        )
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        return
    
    # Sample documents for demo
    sample_docs = [
        {
            'text': "Python is a high-level programming language known for its readability.",
            'metadata': {'type': 'programming', 'difficulty': 'beginner'}
        },
        {
            'text': "Machine learning uses algorithms to parse data and make predictions.",
            'metadata': {'type': 'ai', 'difficulty': 'intermediate'}
        },
        {
            'text': "Web development involves building websites and web applications.",
            'metadata': {'type': 'web', 'difficulty': 'beginner'}
        },
        {
            'text': "Data science combines statistics, programming, and domain knowledge.",
            'metadata': {'type': 'data', 'difficulty': 'advanced'}
        },
        {
            'text': "Artificial intelligence simulates human intelligence in machines.",
            'metadata': {'type': 'ai', 'difficulty': 'advanced'}
        }
    ]
    
    print("\n📚 Indexing sample documents...")
    engine.index_documents(sample_docs)
    
    print("\n🔍 Ready to search! Commands:")
    print("  /search [query] - Search for documents")
    print("  /filter [field]=[value] - Add filter (e.g., /filter type=ai)")
    print("  /clear_filters - Clear all filters")
    print("  /stats - Show engine statistics")
    print("  /quit - Exit")
    
    current_filters = {}
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == '/quit':
                print("Goodbye!")
                break
            
            elif user_input.lower() == '/stats':
                stats = engine.get_stats()
                print("\n📊 Engine Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            elif user_input.lower() == '/clear_filters':
                current_filters = {}
                print("✅ Filters cleared")
            
            elif user_input.startswith('/filter '):
                try:
                    filter_str = user_input[8:].strip()
                    if '=' in filter_str:
                        key, value = filter_str.split('=', 1)
                        current_filters[key.strip()] = value.strip()
                        print(f"✅ Filter added: {key} = {value}")
                    else:
                        print("❌ Invalid filter format. Use: /filter field=value")
                except Exception as e:
                    print(f"❌ Error adding filter: {e}")
            
            elif user_input.startswith('/search '):
                query = user_input[8:].strip()
                if not query:
                    print("❌ Please provide a search query")
                    continue
                
                print(f"\n🔍 Searching for: '{query}'")
                if current_filters:
                    print(f"   Filters: {current_filters}")
                
                results = engine.search(query, top_k=5, filters=current_filters if current_filters else None)
                
                if results:
                    print(f"\n📄 Found {len(results)} results:")
                    for i, result in enumerate(results, 1):
                        print(f"\n{i}. Score: {result.score:.3f}")
                        print(f"   Text: {result.text}")
                        print(f"   Metadata: {result.metadata}")
                else:
                    print("❌ No results found")
            
            else:
                print("❌ Unknown command. Type /help for commands.")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run semantic search demonstrations"""
    print("🚀 Semantic Search Engine Implementation")
    print("Day 24: Building Production-Ready Search with Vector Databases")
    
    # Demo 1: Full semantic search demonstration
    demo_semantic_search()
    
    # Demo 2: Interactive search
    try:
        run_interactive = input("\n🎮 Run interactive search demo? (y/n): ").lower()
        if run_interactive == 'y':
            interactive_search_demo()
    except:
        print("Skipping interactive demo...")
    
    print("\n" + "=" * 70)
    print("✅ Day 24 Complete!")
    print("📚 You've built a production-ready semantic search engine!")
    print("   Next: Document Processing and RAG Architecture (Days 25-27)")

if __name__ == "__main__":
    main()