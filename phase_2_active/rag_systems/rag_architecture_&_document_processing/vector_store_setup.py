"""
🏪 Vector Store Setup & Management
Day 27: Production-ready vector database configuration and management
"""

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Try to import required packages
try:
    import chromadb
    from chromadb.config import Settings
    from chromadb.errors import InvalidDimensionException
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class VectorStoreConfig:
    """Configuration for vector stores"""
    store_type: str = "chromadb"  # Options: chromadb, faiss, simple
    persist_directory: str = "./vector_store"
    collection_name: str = "documents"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    distance_metric: str = "cosine"  # Options: cosine, l2, ip
    batch_size: int = 100
    max_retries: int = 3
    cleanup_interval: int = 3600  # Cleanup every hour (seconds)
    
    def __post_init__(self):
        """Validate configuration"""
        if self.store_type not in ["chromadb", "faiss", "simple"]:
            raise ValueError(f"Unknown store type: {self.store_type}")
        
        if self.distance_metric not in ["cosine", "l2", "ip"]:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Create persist directory
        os.makedirs(self.persist_directory, exist_ok=True)

@dataclass
class DocumentMetadata:
    """Enhanced document metadata for vector stores"""
    source: str
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[str] = None
    modified_date: Optional[str] = None
    document_type: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'source': self.source,
            'title': self.title,
            'author': self.author,
            'created_date': self.created_date,
            'modified_date': self.modified_date,
            'document_type': self.document_type,
            'language': self.language,
            'page_count': self.page_count,
            'word_count': self.word_count,
            'chunk_index': self.chunk_index,
            'total_chunks': self.total_chunks,
            'indexed_at': datetime.now().isoformat()
        }
        
        # Add custom fields
        result.update(self.custom_fields)
        
        # Remove None values
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
            # Generate ID from text hash and timestamp
            text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            doc_id = f"{text_hash}_{timestamp}"
        
        # Generate embedding
        embedding = embedding_model.encode([text])[0].tolist()
        
        return cls(
            id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata
        )

class BaseVectorStore:
    """Base class for vector stores"""
    
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.embedding_model = None
        self._initialize_embedding_model()
    
    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("Sentence Transformers required for embedding model")
        
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # Verify dimension matches config
        actual_dim = self.embedding_model.get_sentence_embedding_dimension()
        if self.config.embedding_dimension != actual_dim:
            logger.warning(f"Embedding dimension mismatch: config={self.config.embedding_dimension}, actual={actual_dim}")
            self.config.embedding_dimension = actual_dim
    
    def add_documents(self, documents: List[IndexedDocument]) -> List[str]:
        """Add documents to vector store"""
        raise NotImplementedError
    
    def search(self, query: str, top_k: int = 10, 
               filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        raise NotImplementedError
    
    def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from store"""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        raise NotImplementedError
    
    def cleanup(self):
        """Clean up store resources"""
        pass
    
    def backup(self, backup_path: str):
        """Create backup of vector store"""
        raise NotImplementedError
    
    def restore(self, backup_path: str):
        """Restore from backup"""
        raise NotImplementedError

class ChromaDBStore(BaseVectorStore):
    """ChromaDB vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        
        if not HAS_CHROMADB:
            raise ImportError("ChromaDB not installed")
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"ChromaDB store initialized: {self.collection.count()} documents")
    
    def _get_or_create_collection(self):
        """Get or create collection with proper configuration"""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.config.collection_name
            )
            logger.info(f"Using existing collection: {self.config.collection_name}")
            return collection
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={
                    "hnsw:space": self.config.distance_metric,
                    "description": "Document embeddings for semantic search",
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": self.config.embedding_model,
                    "embedding_dimension": self.config.embedding_dimension
                }
            )
            logger.info(f"Created new collection: {self.config.collection_name}")
            return collection
    
    def add_documents(self, documents: List[IndexedDocument]) -> List[str]:
        """Add documents to ChromaDB"""
        if not documents:
            return []
        
        # Prepare batch data
        ids = []
        texts = []
        embeddings = []
        metadatas = []
        
        for doc in documents:
            ids.append(doc.id)
            texts.append(doc.text)
            embeddings.append(doc.embedding)
            metadatas.append(doc.metadata.to_dict())
        
        # Add in batches
        added_ids = []
        for i in range(0, len(ids), self.config.batch_size):
            batch_ids = ids[i:i + self.config.batch_size]
            batch_texts = texts[i:i + self.config.batch_size]
            batch_embeddings = embeddings[i:i + self.config.batch_size]
            batch_metadatas = metadatas[i:i + self.config.batch_size]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
                added_ids.extend(batch_ids)
                logger.debug(f"Added batch {i//self.config.batch_size + 1}: {len(batch_ids)} documents")
            except InvalidDimensionException as e:
                logger.error(f"Dimension error in batch {i//self.config.batch_size + 1}: {e}")
                # Try with correct dimension
                actual_dim = len(batch_embeddings[0])
                if actual_dim != self.config.embedding_dimension:
                    logger.warning(f"Adjusting embedding dimension to {actual_dim}")
                    self.config.embedding_dimension = actual_dim
            
            except Exception as e:
                logger.error(f"Error adding batch {i//self.config.batch_size + 1}: {e}")
        
        logger.info(f"Added {len(added_ids)} documents to ChromaDB")
        return added_ids
    
    def search(self, query: str, top_k: int = 10, 
               filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search in ChromaDB"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Convert filters to ChromaDB format
        where_filter = None
        if filters:
            where_filter = {}
            for key, value in filters.items():
                if isinstance(value, list):
                    where_filter[key] = {"$in": value}
                else:
                    where_filter[key] = value
        
        try:
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            if results['ids'][0]:
                for i, doc_id in enumerate(results['ids'][0]):
                    # Convert distance to similarity score
                    distance = results['distances'][0][i]
                    if self.config.distance_metric == "cosine":
                        similarity = 1.0 - distance
                    else:
                        similarity = 1.0 / (1.0 + distance)  # Convert to similarity-like score
                    
                    search_results.append({
                        'id': doc_id,
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': similarity,
                        'distance': distance
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
        try:
            count = self.collection.count()
            
            # Get metadata about collection
            collection_metadata = self.collection.metadata or {}
            
            return {
                'store_type': 'chromadb',
                'document_count': count,
                'collection_name': self.config.collection_name,
                'embedding_model': self.config.embedding_model,
                'embedding_dimension': self.config.embedding_dimension,
                'distance_metric': self.config.distance_metric,
                'collection_metadata': collection_metadata,
                'persist_directory': self.config.persist_directory,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {'error': str(e)}
    
    def backup(self, backup_path: str):
        """Create backup of ChromaDB"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy entire persist directory
            source_dir = Path(self.config.persist_directory)
            if source_dir.exists():
                # Create timestamped backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"chromadb_backup_{timestamp}"
                target_dir = backup_dir / backup_name
                
                shutil.copytree(source_dir, target_dir)
                
                # Save metadata
                metadata = {
                    'backup_date': timestamp,
                    'collection_name': self.config.collection_name,
                    'document_count': self.collection.count(),
                    'source_directory': str(source_dir)
                }
                
                metadata_file = target_dir / "backup_metadata.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Backup created: {target_dir}")
                return str(target_dir)
            
            return None
        
        except Exception as e:
            logger.error(f"Backup error: {e}")
            return None
    
    def restore(self, backup_path: str):
        """Restore from backup"""
        try:
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                logger.error(f"Backup directory not found: {backup_path}")
                return False
            
            # Clear existing data
            self.client.reset()
            
            # Copy backup to persist directory
            source_dir = backup_path
            target_dir = Path(self.config.persist_directory)
            
            # Remove existing directory
            if target_dir.exists():
                shutil.rmtree(target_dir)
            
            # Copy backup
            shutil.copytree(source_dir, target_dir)
            
            logger.info(f"Restored from backup: {backup_path}")
            return True
        
        except Exception as e:
            logger.error(f"Restore error: {e}")
            return False

class FAISSStore(BaseVectorStore):
    """FAISS vector store implementation"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        
        if not HAS_FAISS:
            raise ImportError("FAISS not installed")
        
        self.index = None
        self.documents = {}  # id -> (text, metadata)
        self.index_file = Path(self.config.persist_directory) / "faiss_index.bin"
        self.metadata_file = Path(self.config.persist_directory) / "metadata.json"
        
        self._load_or_create_index()
        
        logger.info(f"FAISS store initialized: {len(self.documents)} documents")
    
    def _load_or_create_index(self):
        """Load existing index or create new one"""
        # Load metadata if exists
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                self.documents = {k: (v['text'], v['metadata']) for k, v in metadata.items()}
            logger.info(f"Loaded metadata: {len(self.documents)} documents")
        
        # Load or create index
        if self.index_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
            
            # Verify dimension matches
            if self.index.d != self.config.embedding_dimension:
                logger.warning(f"Dimension mismatch: index={self.index.d}, config={self.config.embedding_dimension}")
                # Recreate index with correct dimension
                self.index = self._create_new_index()
        else:
            self.index = self._create_new_index()
            logger.info("Created new FAISS index")
    
    def _create_new_index(self):
        """Create new FAISS index"""
        if self.config.distance_metric == "cosine":
            # For cosine similarity, use inner product with normalized vectors
            index = faiss.IndexFlatIP(self.config.embedding_dimension)
        elif self.config.distance_metric == "l2":
            index = faiss.IndexFlatL2(self.config.embedding_dimension)
        else:  # ip (inner product)
            index = faiss.IndexFlatIP(self.config.embedding_dimension)
        
        return index
    
    def add_documents(self, documents: List[IndexedDocument]) -> List[str]:
        """Add documents to FAISS"""
        if not documents:
            return []
        
        added_ids = []
        embeddings = []
        
        for doc in documents:
            # Store document
            self.documents[doc.id] = (doc.text, doc.metadata.to_dict())
            added_ids.append(doc.id)
            
            # Prepare embedding
            embedding = doc.embedding
            
            # Normalize for cosine similarity
            if self.config.distance_metric == "cosine":
                import numpy as np
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = [e / norm for e in embedding]
            
            embeddings.append(embedding)
        
        # Convert to numpy array
        import numpy as np
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        self.index.add(embeddings_np)
        
        # Save index and metadata
        self._save()
        
        logger.info(f"Added {len(added_ids)} documents to FAISS")
        return added_ids
    
    def _save(self):
        """Save index and metadata"""
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_file))
        
        # Save metadata
        metadata = {}
        for doc_id, (text, meta) in self.documents.items():
            metadata[doc_id] = {
                'text': text,
                'metadata': meta
            }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def search(self, query: str, top_k: int = 10, 
               filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search in FAISS"""
        if self.index.ntotal == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Normalize for cosine similarity
        if self.config.distance_metric == "cosine":
            import numpy as np
            norm = np.linalg.norm(query_embedding)
            if norm > 0:
                query_embedding = query_embedding / norm
        
        # Prepare query for FAISS
        query_np = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_np, min(top_k * 2, self.index.ntotal))
        
        # Format results
        search_results = []
        doc_items = list(self.documents.items())  # Convert to list for indexing
        
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(doc_items):
                continue
            
            doc_id, (text, metadata) = doc_items[idx]
            
            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            # Convert distance to similarity score
            distance = distances[0][i]
            if self.config.distance_metric == "cosine":
                similarity = distance  # Already cosine similarity
            else:
                similarity = 1.0 / (1.0 + distance)
            
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
        """Delete documents from FAISS (Note: FAISS doesn't support deletion, so we recreate)"""
        try:
            # Remove documents from metadata
            for doc_id in document_ids:
                if doc_id in self.documents:
                    del self.documents[doc_id]
            
            # Recreate index without deleted documents
            if self.documents:
                # Extract embeddings for remaining documents
                remaining_embeddings = []
                for doc_id, (text, metadata) in self.documents.items():
                    embedding = self.embedding_model.encode([text])[0]
                    
                    # Normalize for cosine similarity
                    if self.config.distance_metric == "cosine":
                        import numpy as np
                        norm = np.linalg.norm(embedding)
                        if norm > 0:
                            embedding = embedding / norm
                    
                    remaining_embeddings.append(embedding)
                
                # Create new index
                self.index = self._create_new_index()
                embeddings_np = np.array(remaining_embeddings, dtype=np.float32)
                self.index.add(embeddings_np)
            else:
                # Empty index
                self.index = self._create_new_index()
            
            # Save
            self._save()
            
            logger.info(f"Deleted {len(document_ids)} documents from FAISS")
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
            'persist_directory': self.config.persist_directory,
            'timestamp': datetime.now().isoformat()
        }
    
    def backup(self, backup_path: str):
        """Create backup of FAISS store"""
        try:
            backup_dir = Path(backup_path)
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy index and metadata files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"faiss_backup_{timestamp}"
            target_dir = backup_dir / backup_name
            target_dir.mkdir()
            
            # Copy files
            if self.index_file.exists():
                shutil.copy2(self.index_file, target_dir / "faiss_index.bin")
            if self.metadata_file.exists():
                shutil.copy2(self.metadata_file, target_dir / "metadata.json")
            
            # Save backup metadata
            metadata = {
                'backup_date': timestamp,
                'document_count': len(self.documents),
                'index_size': self.index.ntotal
            }
            
            with open(target_dir / "backup_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"FAISS backup created: {target_dir}")
            return str(target_dir)
        
        except Exception as e:
            logger.error(f"Backup error: {e}")
            return None

class SimpleVectorStore(BaseVectorStore):
    """Simple in-memory vector store for testing"""
    
    def __init__(self, config: VectorStoreConfig):
        super().__init__(config)
        
        self.documents = {}  # id -> (text, embedding, metadata)
        self.metadata_file = Path(self.config.persist_directory) / "simple_store.json"
        
        self._load()
        
        logger.info(f"Simple store initialized: {len(self.documents)} documents")
    
    def _load(self):
        """Load from disk"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                self.documents = data.get('documents', {})
            logger.info(f"Loaded {len(self.documents)} documents from disk")
    
    def _save(self):
        """Save to disk"""
        data = {
            'documents': self.documents,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_documents(self, documents: List[IndexedDocument]) -> List[str]:
        """Add documents to simple store"""
        if not documents:
            return []
        
        added_ids = []
        for doc in documents:
            self.documents[doc.id] = {
                'text': doc.text,
                'embedding': doc.embedding,
                'metadata': doc.metadata.to_dict()
            }
            added_ids.append(doc.id)
        
        self._save()
        logger.info(f"Added {len(added_ids)} documents to simple store")
        return added_ids
    
    def search(self, query: str, top_k: int = 10, 
               filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search in simple store"""
        if not self.documents:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate similarities
        similarities = []
        for doc_id, doc_data in self.documents.items():
            # Apply filters
            if filters and not self._matches_filters(doc_data['metadata'], filters):
                continue
            
            # Calculate cosine similarity
            import numpy as np
            doc_embedding = np.array(doc_data['embedding'])
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            
            similarities.append((doc_id, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-k results
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
        """Delete documents from simple store"""
        deleted_count = 0
        for doc_id in document_ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                deleted_count += 1
        
        self._save()
        logger.info(f"Deleted {deleted_count} documents from simple store")
        return deleted_count > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simple store statistics"""
        return {
            'store_type': 'simple',
            'document_count': len(self.documents),
            'persist_directory': self.config.persist_directory,
            'timestamp': datetime.now().isoformat()
        }

class VectorStoreFactory:
    """Factory for creating vector stores"""
    
    @staticmethod
    def create_store(config: VectorStoreConfig) -> BaseVectorStore:
        """Create vector store based on configuration"""
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
        """Get list of available vector store types"""
        stores = ["simple"]
        
        if HAS_CHROMADB:
            stores.append("chromadb")
        if HAS_FAISS:
            stores.append("faiss")
        
        return stores

class VectorStoreManager:
    """Manager for multiple vector stores"""
    
    def __init__(self, base_config: VectorStoreConfig):
        self.base_config = base_config
        self.stores: Dict[str, BaseVectorStore] = {}
    
    def get_store(self, collection_name: Optional[str] = None) -> BaseVectorStore:
        """Get or create vector store for collection"""
        name = collection_name or self.base_config.collection_name
        
        if name not in self.stores:
            # Create new config for this collection
            config = VectorStoreConfig(**self.base_config.__dict__)
            config.collection_name = name
            config.persist_directory = os.path.join(
                self.base_config.persist_directory,
                name
            )
            
            # Create store
            self.stores[name] = VectorStoreFactory.create_store(config)
        
        return self.stores[name]
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        collections = []
        base_dir = Path(self.base_config.persist_directory)
        
        if base_dir.exists():
            for item in base_dir.iterdir():
                if item.is_dir():
                    collections.append(item.name)
        
        return collections
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        try:
            # Remove from memory
            if collection_name in self.stores:
                del self.stores[collection_name]
            
            # Remove from disk
            collection_dir = Path(self.base_config.persist_directory) / collection_name
            if collection_dir.exists():
                shutil.rmtree(collection_dir)
            
            logger.info(f"Deleted collection: {collection_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {e}")
            return False
    
    def backup_all(self, backup_path: str) -> Dict[str, str]:
        """Backup all collections"""
        backups = {}
        
        for name, store in self.stores.items():
            backup_dir = store.backup(backup_path)
            if backup_dir:
                backups[name] = backup_dir
        
        return backups
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """Get statistics for all stores"""
        stats = {
            'total_collections': len(self.stores),
            'collections': [],
            'available_stores': VectorStoreFactory.get_available_stores(),
            'base_config': self.base_config.__dict__
        }
        
        for name, store in self.stores.items():
            store_stats = store.get_stats()
            store_stats['collection_name'] = name
            stats['collections'].append(store_stats)
        
        return stats

def demo_vector_stores():
    """Demonstrate vector store capabilities"""
    print("=" * 70)
    print("🏪 VECTOR STORE DEMONSTRATION (Day 27)")
    print("=" * 70)
    
    # Check available stores
    available_stores = VectorStoreFactory.get_available_stores()
    print(f"\n🔧 Available vector stores: {available_stores}")
    
    # Test each available store
    for store_type in available_stores[:2]:  # Test first 2 available stores
        print(f"\n{'='*60}")
        print(f"Testing {store_type.upper()} Store")
        print(f"{'='*60}")
        
        try:
            # Create configuration
            config = VectorStoreConfig(
                store_type=store_type,
                persist_directory=f"./test_{store_type}_store",
                collection_name="test_documents",
                embedding_model="all-MiniLM-L6-v2",
                distance_metric="cosine"
            )
            
            # Create store
            print(f"\n1️⃣ Creating {store_type} store...")
            store = VectorStoreFactory.create_store(config)
            
            # Create test documents
            print("\n2️⃣ Creating test documents...")
            test_documents = [
                "Artificial intelligence is transforming healthcare with diagnostic tools.",
                "Machine learning algorithms can predict patient outcomes from medical data.",
                "Natural language processing enables doctors to extract insights from clinical notes.",
                "Computer vision systems assist in analyzing medical images like X-rays and MRIs.",
                "Robotic surgery systems use AI to enhance precision and reduce recovery time."
            ]
            
            indexed_docs = []
            for i, text in enumerate(test_documents):
                metadata = DocumentMetadata(
                    source="test_source",
                    title=f"AI in Healthcare Document {i+1}",
                    document_type="article",
                    language="en",
                    word_count=len(text.split())
                )
                
                doc = IndexedDocument.from_text(
                    text=text,
                    metadata=metadata,
                    embedding_model=store.embedding_model,
                    doc_id=f"doc_{i+1}"
                )
                indexed_docs.append(doc)
            
            # Add documents
            print(f"\n3️⃣ Adding {len(indexed_docs)} documents...")
            added_ids = store.add_documents(indexed_docs)
            print(f"   ✅ Added documents: {len(added_ids)}")
            
            # Search test
            print("\n4️⃣ Testing search...")
            queries = [
                "medical diagnosis with AI",
                "machine learning in healthcare",
                "computer vision for medical images"
            ]
            
            for query in queries:
                results = store.search(query, top_k=2)
                print(f"   🔍 '{query}': found {len(results)} results")
                if results:
                    print(f"      Top result: {results[0]['text'][:80]}...")
            
            # Get stats
            print("\n5️⃣ Store statistics:")
            stats = store.get_stats()
            for key, value in stats.items():
                if key not in ['collection_metadata', 'custom_fields']:
                    print(f"   {key}: {value}")
            
            # Test backup (if supported)
            if hasattr(store, 'backup'):
                print("\n6️⃣ Testing backup...")
                backup_dir = store.backup("./backups")
                if backup_dir:
                    print(f"   ✅ Backup created: {backup_dir}")
            
            # Cleanup test directory
            import shutil
            test_dir = Path(f"./test_{store_type}_store")
            if test_dir.exists():
                shutil.rmtree(test_dir)
            
            print(f"\n✅ {store_type.upper()} store test completed successfully!")
        
        except Exception as e:
            print(f"❌ Error with {store_type} store: {e}")

def benchmark_vector_stores():
    """Benchmark different vector stores"""
    print("\n⏱️ VECTOR STORE BENCHMARK")
    print("=" * 60)
    
    import time
    import numpy as np
    
    # Generate test data
    print("\n📊 Generating test data...")
    
    n_documents = 100
    doc_length = 100  # words per document
    
    test_documents = []
    for i in range(n_documents):
        words = [f"word_{j}" for j in range(doc_length)]
        text = f"Document {i}: " + " ".join(words)
        test_documents.append(text)
    
    print(f"   Generated {n_documents} documents, ~{doc_length} words each")
    
    # Test configurations
    test_configs = []
    
    if "simple" in VectorStoreFactory.get_available_stores():
        test_configs.append(("Simple", "simple"))
    
    if "chromadb" in VectorStoreFactory.get_available_stores():
        test_configs.append(("ChromaDB", "chromadb"))
    
    if "faiss" in VectorStoreFactory.get_available_stores():
        test_configs.append(("FAISS", "faiss"))
    
    results = []
    
    for store_name, store_type in test_configs:
        print(f"\n⏱️ Benchmarking {store_name}...")
        
        try:
            # Create temporary store
            config = VectorStoreConfig(
                store_type=store_type,
                persist_directory=f"./benchmark_{store_type}",
                collection_name="benchmark",
                embedding_model="all-MiniLM-L6-v2"
            )
            
            store = VectorStoreFactory.create_store(config)
            
            # Benchmark: Adding documents
            print("   Testing document addition...")
            
            start_time = time.perf_counter()
            
            indexed_docs = []
            for i, text in enumerate(test_documents):
                metadata = DocumentMetadata(
                    source="benchmark",
                    title=f"Benchmark Doc {i}",
                    word_count=len(text.split())
                )
                
                doc = IndexedDocument.from_text(
                    text=text,
                    metadata=metadata,
                    embedding_model=store.embedding_model
                )
                indexed_docs.append(doc)
            
            added_ids = store.add_documents(indexed_docs)
            add_time = time.perf_counter() - start_time
            
            # Benchmark: Search
            print("   Testing search...")
            
            search_times = []
            test_queries = ["document analysis", "word processing", "test search"]
            
            for query in test_queries:
                start = time.perf_counter()
                results = store.search(query, top_k=10)
                search_times.append(time.perf_counter() - start)
            
            avg_search_time = np.mean(search_times)
            
            # Get stats
            stats = store.get_stats()
            
            results.append({
                'store': store_name,
                'type': store_type,
                'add_time': add_time,
                'add_rate': n_documents / add_time if add_time > 0 else 0,
                'avg_search_time': avg_search_time,
                'document_count': stats.get('document_count', 0),
                'embedding_dimension': stats.get('embedding_dimension', 0)
            })
            
            print(f"   ✅ Add time: {add_time:.3f}s ({n_documents/add_time:.1f} docs/s)")
            print(f"   ✅ Avg search time: {avg_search_time:.3f}s")
            
            # Cleanup
            import shutil
            store_dir = Path(f"./benchmark_{store_type}")
            if store_dir.exists():
                shutil.rmtree(store_dir)
        
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results.append({
                'store': store_name,
                'type': store_type,
                'add_time': 0,
                'add_rate': 0,
                'avg_search_time': 0,
                'document_count': 0,
                'embedding_dimension': 0
            })
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\n{'Store':<12} {'Type':<12} {'Add Time':<12} {'Add Rate':<12} {'Search Time':<12} {'Docs':<8}")
    print("-" * 68)
    
    for result in results:
        print(f"{result['store']:<12} "
              f"{result['type']:<12} "
              f"{result['add_time']:<12.3f} "
              f"{result['add_rate']:<12.1f} "
              f"{result['avg_search_time']:<12.3f} "
              f"{result['document_count']:<8}")

def interactive_store_demo():
    """Interactive vector store demonstration"""
    print("\n🎮 Interactive Vector Store Manager")
    print("=" * 60)
    
    # Create manager
    base_config = VectorStoreConfig(
        store_type="chromadb" if HAS_CHROMADB else "simple",
        persist_directory="./interactive_store",
        collection_name="default"
    )
    
    manager = VectorStoreManager(base_config)
    
    print(f"\n🔧 Using {base_config.store_type} as default store")
    print(f"   Persist directory: {base_config.persist_directory}")
    
    print("\n⚙️ Available commands:")
    print("  /create [name] - Create new collection")
    print("  /list - List all collections")
    print("  /use [name] - Switch to collection")
    print("  /add [text] - Add document to current collection")
    print("  /search [query] - Search in current collection")
    print("  /stats - Show collection statistics")
    print("  /delete [name] - Delete collection")
    print("  /backup - Backup all collections")
    print("  /quit - Exit")
    
    current_collection = "default"
    
    while True:
        try:
            command = input(f"\n[{current_collection}]> ").strip()
            
            if command.lower() == '/quit':
                print("Goodbye!")
                break
            
            elif command.lower() == '/list':
                collections = manager.list_collections()
                if collections:
                    print(f"\n📚 Collections:")
                    for coll in collections:
                        print(f"  - {coll}")
                else:
                    print("❌ No collections found")
            
            elif command.lower().startswith('/create '):
                try:
                    name = command[8:].strip()
                    if not name:
                        print("❌ Collection name required")
                        continue
                    
                    # Create collection by getting store
                    store = manager.get_store(name)
                    print(f"✅ Created collection: {name}")
                    current_collection = name
                
                except Exception as e:
                    print(f"❌ Error creating collection: {e}")
            
            elif command.lower().startswith('/use '):
                try:
                    name = command[5:].strip()
                    if not name:
                        print("❌ Collection name required")
                        continue
                    
                    # Check if collection exists
                    collections = manager.list_collections()
                    if name in collections or name == "default":
                        current_collection = name
                        print(f"✅ Switched to collection: {name}")
                    else:
                        print(f"❌ Collection not found: {name}")
                
                except Exception as e:
                    print(f"❌ Error switching collection: {e}")
            
            elif command.lower().startswith('/add '):
                try:
                    text = command[5:].strip()
                    if not text:
                        print("❌ Text required")
                        continue
                    
                    # Get current store
                    store = manager.get_store(current_collection)
                    
                    # Create document
                    metadata = DocumentMetadata(
                        source="interactive",
                        title=f"Interactive Doc",
                        created_date=datetime.now().isoformat(),
                        word_count=len(text.split())
                    )
                    
                    doc = IndexedDocument.from_text(
                        text=text,
                        metadata=metadata,
                        embedding_model=store.embedding_model
                    )
                    
                    # Add to store
                    added_ids = store.add_documents([doc])
                    
                    print(f"✅ Added document to {current_collection}")
                    print(f"   Document ID: {added_ids[0]}")
                    print(f"   Text: {text[:80]}...")
                
                except Exception as e:
                    print(f"❌ Error adding document: {e}")
            
            elif command.lower().startswith('/search '):
                try:
                    query = command[8:].strip()
                    if not query:
                        print("❌ Search query required")
                        continue
                    
                    # Get current store
                    store = manager.get_store(current_collection)
                    
                    # Search
                    results = store.search(query, top_k=5)
                    
                    if results:
                        print(f"\n🔍 Found {len(results)} results for '{query}':")
                        for i, result in enumerate(results, 1):
                            print(f"\n{i}. Score: {result['score']:.3f}")
                            print(f"   Text: {result['text'][:100]}...")
                            print(f"   Source: {result['metadata'].get('source', 'N/A')}")
                    else:
                        print(f"❌ No results found for '{query}'")
                
                except Exception as e:
                    print(f"❌ Error searching: {e}")
            
            elif command.lower() == '/stats':
                try:
                    # Get current store stats
                    store = manager.get_store(current_collection)
                    stats = store.get_stats()
                    
                    print(f"\n📊 Statistics for {current_collection}:")
                    for key, value in stats.items():
                        if key not in ['collection_metadata', 'custom_fields']:
                            print(f"  {key}: {value}")
                
                except Exception as e:
                    print(f"❌ Error getting stats: {e}")
            
            elif command.lower().startswith('/delete '):
                try:
                    name = command[8:].strip()
                    if not name:
                        print("❌ Collection name required")
                        continue
                    
                    if name == current_collection:
                        print("⚠️ Cannot delete current collection. Switch to another first.")
                        continue
                    
                    confirm = input(f"⚠️ Delete collection '{name}'? (y/n): ").lower()
                    if confirm == 'y':
                        success = manager.delete_collection(name)
                        if success:
                            print(f"✅ Deleted collection: {name}")
                        else:
                            print(f"❌ Failed to delete collection: {name}")
                
                except Exception as e:
                    print(f"❌ Error deleting collection: {e}")
            
            elif command.lower() == '/backup':
                try:
                    backup_dir = "./backups"
                    backups = manager.backup_all(backup_dir)
                    
                    if backups:
                        print(f"\n💾 Backups created:")
                        for name, path in backups.items():
                            print(f"  {name}: {path}")
                    else:
                        print("❌ No backups created")
                
                except Exception as e:
                    print(f"❌ Error creating backups: {e}")
            
            else:
                print("❌ Unknown command. Type /help for commands.")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run vector store demonstrations"""
    print("🚀 Production Vector Store Management")
    print("Day 27: Multi-backend Vector Database Configuration")
    
    # Demo 1: Vector store demonstration
    demo_vector_stores()
    
    # Demo 2: Benchmarking
    benchmark_vector_stores()
    
    # Demo 3: Interactive manager
    try:
        run_interactive = input("\n🎮 Run interactive vector store manager? (y/n): ").lower()
        if run_interactive == 'y':
            interactive_store_demo()
    except:
        print("Skipping interactive demo...")
    
    print("\n" + "=" * 70)
    print("✅ Day 27 Complete!")
    print("📚 You've built a production-ready vector store management system!")
    print("   Next: First RAG System Implementation (Days 28-30)")

if __name__ == "__main__":
    main()