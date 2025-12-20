"""
🏪 Vector Store Setup & Management - OPTIMIZED VERSION
Day 27: Production-ready vector database configuration and management
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging

# ==================== DEPENDENCY CHECKS ====================
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    print("⚠️ ChromaDB not installed. Install with: pip install chromadb")

try:
    import faiss
    import numpy as np
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

@dataclass
class VectorStoreConfig:
    """Configuration for vector stores"""
    store_type: str = "chromadb"
    persist_directory: str = "./vector_store"
    collection_name: str = "documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    distance_metric: str = "cosine"
    batch_size: int = 100
    
    def __post_init__(self):
        """Validate configuration"""
        valid_stores = ["chromadb", "faiss", "simple"]
        if self.store_type not in valid_stores:
            raise ValueError(f"Unknown store type: {self.store_type}. Choose from {valid_stores}")
        
        valid_metrics = ["cosine", "l2", "ip"]
        if self.distance_metric not in valid_metrics:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}. Choose from {valid_metrics}")
        
        os.makedirs(self.persist_directory, exist_ok=True)

@dataclass
class DocumentMetadata:
    """Document metadata for vector stores"""
    source: str
    title: Optional[str] = None
    document_type: Optional[str] = None
    created_date: Optional[str] = None
    word_count: Optional[int] = None
    chunk_index: Optional[int] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, removing None values"""
        result = {
            'source': self.source,
            'title': self.title,
            'document_type': self.document_type,
            'created_date': self.created_date,
            'word_count': self.word_count,
            'chunk_index': self.chunk_index,
            'indexed_at': datetime.now().isoformat()
        }
        result.update(self.custom_fields)
        return {k: v for k, v in result.items() if v is not None}

@dataclass
class IndexedDocument:
    """Document with embedding for indexing"""
    id: str
    text: str
    embedding: List[float]
    metadata: DocumentMetadata
    
    @classmethod
    def from_text(cls, text: str, metadata: DocumentMetadata, 
                  embedding_model, doc_id: Optional[str] = None):
        """Create from text with automatic embedding"""
        if not doc_id:
            text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            doc_id = f"{text_hash}_{timestamp}"
        
        embedding = embedding_model.encode([text])[0].tolist()
        
        return cls(id=doc_id, text=text, embedding=embedding, metadata=metadata)

# ==================== BASE VECTOR STORE ====================

class BaseVectorStore:
    """Base class for vector stores"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("Sentence Transformers required. Install: pip install sentence-transformers")
        
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        actual_dim = self.embedding_model.get_sentence_embedding_dimension()
        if self.config.embedding_dimension != actual_dim:
            logger.warning(f"Adjusting embedding dimension: {self.config.embedding_dimension} → {actual_dim}")
            self.config.embedding_dimension = actual_dim
    
    def add_documents(self, documents: List[IndexedDocument]) -> List[str]:
        raise NotImplementedError
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    def backup(self, backup_path: str) -> Optional[str]:
        raise NotImplementedError

# ==================== CHROMADB IMPLEMENTATION ====================

class ChromaDBStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        if not HAS_CHROMADB:
            raise ImportError("ChromaDB not installed. Install: pip install chromadb")
        
        super().__init__(config)
        
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )
        
        self.collection = self._get_or_create_collection()
        logger.info(f"ChromaDB initialized: {self.collection.count()} documents")
    
    def _get_or_create_collection(self):
        """Get or create collection"""
        try:
            collection = self.client.get_collection(name=self.config.collection_name)
            logger.info(f"Using existing collection: {self.config.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={
                    "hnsw:space": self.config.distance_metric,
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": self.config.embedding_model
                }
            )
            logger.info(f"Created collection: {self.config.collection_name}")
        return collection
    
    def add_documents(self, documents: List[IndexedDocument]) -> List[str]:
        """Add documents to ChromaDB"""
        if not documents:
            return []
        
        ids = [doc.id for doc in documents]
        texts = [doc.text for doc in documents]
        embeddings = [doc.embedding for doc in documents]
        metadatas = [doc.metadata.to_dict() for doc in documents]
        
        added_ids = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(ids), batch_size):
            try:
                batch_slice = slice(i, i + batch_size)
                self.collection.add(
                    ids=ids[batch_slice],
                    documents=texts[batch_slice],
                    embeddings=embeddings[batch_slice],
                    metadatas=metadatas[batch_slice]
                )
                added_ids.extend(ids[batch_slice])
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
        
        logger.info(f"Added {len(added_ids)} documents to ChromaDB")
        return added_ids
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search in ChromaDB"""
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        where_filter = None
        if filters:
            where_filter = {k: {"$in": v} if isinstance(v, list) else v for k, v in filters.items()}
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            search_results = []
            if results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity = 1.0 - distance if self.config.distance_metric == "cosine" else 1.0 / (1.0 + distance)
                    
                    search_results.append({
                        'id': doc_id,
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': float(similarity),
                        'distance': float(distance)
                    })
            
            return search_results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from ChromaDB"""
        try:
            self.collection.delete(ids=document_ids)
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ChromaDB statistics"""
        return {
            'store_type': 'chromadb',
            'document_count': self.collection.count(),
            'collection_name': self.config.collection_name,
            'embedding_model': self.config.embedding_model,
            'embedding_dimension': self.config.embedding_dimension,
            'distance_metric': self.config.distance_metric,
            'timestamp': datetime.now().isoformat()
        }
    
    def backup(self, backup_path: str) -> Optional[str]:
        """Create backup of ChromaDB"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"chromadb_backup_{timestamp}"
            target_dir = backup_dir / backup_name
            
            source_dir = Path(self.config.persist_directory)
            if source_dir.exists():
                shutil.copytree(source_dir, target_dir)
                
                metadata = {
                    'backup_date': timestamp,
                    'collection_name': self.config.collection_name,
                    'document_count': self.collection.count()
                }
                
                with open(target_dir / "backup_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Backup created: {target_dir}")
                return str(target_dir)
            return None
        except Exception as e:
            logger.error(f"Backup error: {e}")
            return None

# ==================== FAISS IMPLEMENTATION ====================

class FAISSStore(BaseVectorStore):
    """FAISS vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        if not HAS_FAISS:
            raise ImportError("FAISS not installed. Install: pip install faiss-cpu")
        
        super().__init__(config)
        
        self.documents = {}
        self.index_file = Path(self.config.persist_directory) / "faiss_index.bin"
        self.metadata_file = Path(self.config.persist_directory) / "metadata.json"
        
        self._load_or_create_index()
        logger.info(f"FAISS initialized: {len(self.documents)} documents")
    
    def _load_or_create_index(self):
        """Load or create FAISS index"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.documents = {k: (v['text'], v['metadata']) for k, v in metadata.items()}
        
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
        else:
            self.index = self._create_new_index()
    
    def _create_new_index(self):
        """Create new FAISS index"""
        if self.config.distance_metric == "cosine":
            return faiss.IndexFlatIP(self.config.embedding_dimension)
        elif self.config.distance_metric == "l2":
            return faiss.IndexFlatL2(self.config.embedding_dimension)
        else:
            return faiss.IndexFlatIP(self.config.embedding_dimension)
    
    def add_documents(self, documents: List[IndexedDocument]) -> List[str]:
        """Add documents to FAISS"""
        if not documents:
            return []
        
        added_ids = []
        embeddings = []
        
        for doc in documents:
            self.documents[doc.id] = (doc.text, doc.metadata.to_dict())
            added_ids.append(doc.id)
            
            embedding = doc.embedding
            if self.config.distance_metric == "cosine":
                norm = np.linalg.norm(embedding)
                embedding = [e / norm for e in embedding] if norm > 0 else embedding
            
            embeddings.append(embedding)
        
        embeddings_np = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings_np)
        
        self._save()
        logger.info(f"Added {len(added_ids)} documents to FAISS")
        return added_ids
    
    def _save(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, str(self.index_file))
        
        metadata = {doc_id: {'text': text, 'metadata': meta} 
                   for doc_id, (text, meta) in self.documents.items()}
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search in FAISS"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        if self.config.distance_metric == "cosine":
            norm = np.linalg.norm(query_embedding)
            query_embedding = query_embedding / norm if norm > 0 else query_embedding
        
        query_np = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_np, min(top_k * 2, self.index.ntotal))
        
        search_results = []
        doc_items = list(self.documents.items())
        
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(doc_items):
                continue
            
            doc_id, (text, metadata) = doc_items[idx]
            
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            distance = distances[0][i]
            similarity = distance if self.config.distance_metric == "cosine" else 1.0 / (1.0 + distance)
            
            search_results.append({
                'id': doc_id,
                'text': text,
                'metadata': metadata,
                'score': float(similarity),
                'distance': float(distance)
            })
            
            if len(search_results) >= top_k:
                break
        
        return search_results
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
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
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from FAISS (requires rebuild)"""
        try:
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
            
            if self.documents:
                remaining_embeddings = []
                for doc_id, (text, _) in self.documents.items():
                    embedding = self.embedding_model.encode([text])[0]
                    
                    if self.config.distance_metric == "cosine":
                        norm = np.linalg.norm(embedding)
                        embedding = embedding / norm if norm > 0 else embedding
                    
                    remaining_embeddings.append(embedding)
                
                self.index = self._create_new_index()
                embeddings_np = np.array(remaining_embeddings, dtype=np.float32)
                self.index.add(embeddings_np)
            else:
                self.index = self._create_new_index()
            
            self._save()
            logger.info(f"Deleted {len(document_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get FAISS statistics"""
        return {
            'store_type': 'faiss',
            'document_count': len(self.documents),
            'index_size': self.index.ntotal,
            'embedding_dimension': self.config.embedding_dimension,
            'distance_metric': self.config.distance_metric,
            'timestamp': datetime.now().isoformat()
        }
    
    def backup(self, backup_path: str) -> Optional[str]:
        """Create backup of FAISS store"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_dir = backup_dir / f"faiss_backup_{timestamp}"
            target_dir.mkdir()
            
            if self.index_file.exists():
                shutil.copy2(self.index_file, target_dir / "faiss_index.bin")
            if self.metadata_file.exists():
                shutil.copy2(self.metadata_file, target_dir / "metadata.json")
            
            logger.info(f"Backup created: {target_dir}")
            return str(target_dir)
        except Exception as e:
            logger.error(f"Backup error: {e}")
            return None

# ==================== SIMPLE STORE IMPLEMENTATION ====================

class SimpleVectorStore(BaseVectorStore):
    """Simple in-memory vector store"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        
        self.documents = {}
        self.metadata_file = Path(self.config.persist_directory) / "simple_store.json"
        
        self._load()
        logger.info(f"Simple store initialized: {len(self.documents)} documents")
    
    def _load(self):
        """Load from disk"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.documents = data.get('documents', {})
    
    def _save(self):
        """Save to disk"""
        data = {'documents': self.documents, 'timestamp': datetime.now().isoformat()}
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_documents(self, documents: List[IndexedDocument]) -> List[str]:
        """Add documents"""
        added_ids = []
        for doc in documents:
            self.documents[doc.id] = {
                'text': doc.text,
                'embedding': doc.embedding,
                'metadata': doc.metadata.to_dict()
            }
            added_ids.append(doc.id)
        
        self._save()
        logger.info(f"Added {len(added_ids)} documents")
        return added_ids
    
    def search(self, query: str, top_k: int = 10, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search documents"""
        if not self.documents:
            return []
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        similarities = []
        for doc_id, doc_data in self.documents.items():
            if filters and not self._matches_filters(doc_data['metadata'], filters):
                continue
            
            doc_embedding = np.array(doc_data['embedding'])
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            similarities.append((doc_id, float(similarity)))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, similarity in similarities[:top_k]:
            doc_data = self.documents[doc_id]
            results.append({
                'id': doc_id,
                'text': doc_data['text'],
                'metadata': doc_data['metadata'],
                'score': similarity
            })
        
        return results
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """Check filter match"""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            elif metadata[key] != value:
                return False
        return True
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents"""
        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
        
        self._save()
        logger.info(f"Deleted {len(document_ids)} documents")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            'store_type': 'simple',
            'document_count': len(self.documents),
            'timestamp': datetime.now().isoformat()
        }
    
    def backup(self, backup_path: str) -> Optional[str]:
        """Create backup"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            target_file = backup_dir / f"simple_backup_{timestamp}.json"
            
            shutil.copy2(self.metadata_file, target_file)
            logger.info(f"Backup created: {target_file}")
            return str(target_file)
        except Exception as e:
            logger.error(f"Backup error: {e}")
            return None

# ==================== FACTORY ====================

class VectorStoreFactory:
    """Factory for creating vector stores"""
    
    @staticmethod
    def create_store(config: VectorStoreConfig) -> BaseVectorStore:
        """Create vector store"""
        if config.store_type == "chromadb":
            if not HAS_CHROMADB:
                raise ImportError("ChromaDB not installed")
            return ChromaDBStore(config)
        elif config.store_type == "faiss":
            if not HAS_FAISS:
                raise ImportError("FAISS not installed")
            return FAISSStore(config)
        elif config.store_type == "simple":
            return SimpleVectorStore(config)
        else:
            raise ValueError(f"Unknown store type: {config.store_type}")
    
    @staticmethod
    def get_available_stores() -> List[str]:
        """Get available stores"""
        stores = ["simple"]
        if HAS_CHROMADB:
            stores.append("chromadb")
        if HAS_FAISS:
            stores.append("faiss")
        return stores

# ==================== DEMO FUNCTIONS ====================

def demo_vector_stores():
    """Demonstrate vector store capabilities"""
    print("=" * 70)
    print("🏪 VECTOR STORE DEMONSTRATION (Day 27)")
    print("=" * 70)
    
    available_stores = VectorStoreFactory.get_available_stores()
    print(f"\n🔧 Available stores: {available_stores}")
    
    for store_type in available_stores[:2]:
        print(f"\n{'='*60}")
        print(f"Testing {store_type.upper()} Store")
        print(f"{'='*60}")
        
        try:
            config = VectorStoreConfig(
                store_type=store_type,
                persist_directory=f"./test_{store_type}_store",
                collection_name="test_docs"
            )
            
            print(f"\n1️⃣ Creating {store_type} store...")
            store = VectorStoreFactory.create_store(config)
            
            print("\n2️⃣ Creating test documents...")
            test_docs = [
                "AI is transforming healthcare with diagnostic tools.",
                "Machine learning predicts patient outcomes from data.",
                "NLP extracts insights from clinical notes.",
                "Computer vision analyzes medical images.",
                "Robotic surgery uses AI for precision."
            ]
            
            indexed_docs = []
            for i, text in enumerate(test_docs):
                metadata = DocumentMetadata(
                    source="test",
                    title=f"AI Doc {i+1}",
                    document_type="article",
                    word_count=len(text.split())
                )
                
                doc = IndexedDocument.from_text(
                    text=text,
                    metadata=metadata,
                    embedding_model=store.embedding_model,
                    doc_id=f"doc_{i+1}"
                )
                indexed_docs.append(doc)
            
            print(f"\n3️⃣ Adding {len(indexed_docs)} documents...")
            added_ids = store.add_documents(indexed_docs)
            print(f"   ✅ Added: {len(added_ids)} documents")
            
            print("\n4️⃣ Testing search...")
            queries = ["medical AI", "machine learning healthcare", "computer vision"]
            
            for query in queries:
                results = store.search(query, top_k=2)
                print(f"   🔍 '{query}': {len(results)} results")
                if results:
                    print(f"      Top: {results[0]['text'][:60]}...")
            
            print("\n5️⃣ Statistics:")
            stats = store.get_stats()
            for key, value in stats.items():
                if key not in ['collection_metadata']:
                    print(f"   {key}: {value}")
            
            print("\n6️⃣ Testing backup...")
            backup_dir = store.backup("./backups")
            if backup_dir:
                print(f"   ✅ Backup: {backup_dir}")
            
            # Cleanup
            test_dir = Path(f"./test_{store_type}_store")
            if test_dir.exists():
                shutil.rmtree(test_dir)
            
            print(f"\n✅ {store_type.upper()} test completed!")
        
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run demonstrations"""
    print("🚀 Production Vector Store Management")
    print("Day 27: Multi-backend Vector Database Configuration")
    
    demo_vector_stores()
    
    print("\n" + "=" * 70)
    print("✅ Day 27 Complete!")
    print("📚 Production-ready vector store system built!")
    print("   Next: First RAG System (Days 28-30)")

if __name__ == "__main__":
    main()