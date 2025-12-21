"""
Hybrid Search System - Advanced RAG Techniques
Combines semantic, keyword, and dense retrieval for superior search results.
"""

import os
import json
import math
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
import openai
from dotenv import load_dotenv
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Load environment variables
load_dotenv()

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

# ========== Enums and Data Classes ==========

class SearchType(Enum):
    """Types of search algorithms"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    DENSE = "dense"
    SPARSE = "sparse"
    BM25 = "bm25"

class SearchResult:
    """Enhanced search result with multiple scores"""
    
    def __init__(
        self,
        document: Document,
        semantic_score: float = 0.0,
        keyword_score: float = 0.0,
        bm25_score: float = 0.0,
        hybrid_score: float = 0.0,
        final_score: float = 0.0,
        rank: int = 0
    ):
        self.document = document
        self.semantic_score = semantic_score
        self.keyword_score = keyword_score
        self.bm25_score = bm25_score
        self.hybrid_score = hybrid_score
        self.final_score = final_score
        self.rank = rank
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "content": self.document.page_content[:200] + "...",
            "source": self.document.metadata.get("source", "unknown"),
            "page": self.document.metadata.get("page", 0),
            "semantic_score": round(self.semantic_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "bm25_score": round(self.bm25_score, 4),
            "hybrid_score": round(self.hybrid_score, 4),
            "final_score": round(self.final_score, 4),
            "rank": self.rank
        }
    
    def __repr__(self):
        return f"SearchResult(score={self.final_score:.4f}, source={self.document.metadata.get('source', 'unknown')})"

@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search"""
    semantic_weight: float = 0.6
    keyword_weight: float = 0.3
    bm25_weight: float = 0.1
    alpha: float = 0.5  # For hybrid fusion
    k: int = 10  # Number of results to retrieve
    score_threshold: float = 0.1
    rerank_top_k: int = 5  # Top K to re-rank
    use_reranking: bool = True
    
    def validate(self):
        """Validate configuration"""
        weights_sum = self.semantic_weight + self.keyword_weight + self.bm25_weight
        if abs(weights_sum - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {weights_sum}")
        
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"Alpha must be between 0 and 1, got {self.alpha}")
        
        if self.k <= 0:
            raise ValueError(f"K must be positive, got {self.k}")

# ========== Tokenizer and Preprocessor ==========

class TextPreprocessor:
    """Advanced text preprocessing for search"""
    
    def __init__(self, language: str = "english"):
        self.language = language
        self.stop_words = set(stopwords.words(language))
        
        # Try to load spaCy model
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("⚠️  spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    
    def preprocess(self, text: str) -> str:
        """Preprocess text for search"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Lemmatize if spaCy is available
        if self.nlp:
            doc = self.nlp(" ".join(tokens))
            tokens = [token.lemma_ for token in doc]
        
        return " ".join(tokens)
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract keywords from text"""
        # Preprocess text
        processed_text = self.preprocess(text)
        
        # Use TF-IDF to extract keywords
        vectorizer = TfidfVectorizer(max_features=top_n * 2)
        try:
            tfidf_matrix = vectorizer.fit_transform([processed_text])
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix.toarray()[0]
            
            # Get top N keywords
            top_indices = tfidf_scores.argsort()[-top_n:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            return keywords
        except:
            # Fallback: return most frequent words
            words = processed_text.split()
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in sorted_words[:top_n]]
    
    def expand_query(self, query: str, expansion_terms: int = 3) -> str:
        """Expand query with related terms"""
        if not self.nlp:
            return query
        
        doc = self.nlp(query)
        expanded_terms = []
        
        for token in doc:
            expanded_terms.append(token.text)
            
            # Add synonyms
            if token.pos_ in ["NOUN", "ADJ", "VERB"]:
                for syn in token._.wordnet.synsets()[:2]:
                    for lemma in syn.lemmas()[:2]:
                        if lemma.name() != token.text:
                            expanded_terms.append(lemma.name())
        
        # Remove duplicates and limit
        expanded_terms = list(set(expanded_terms))[:expansion_terms * 2]
        return query + " " + " ".join(expanded_terms)

# ========== Search Algorithms ==========

class SemanticSearch:
    """Semantic search using embeddings"""
    
    def __init__(self, embedding_model: str = "openai"):
        self.embedding_model = embedding_model
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if self.embedding_model == "openai":
            return OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                model="text-embedding-ada-002"
            )
        else:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def embed_query(self, query: str) -> List[float]:
        """Embed a query"""
        return self.embeddings.embed_query(query)
    
    def search(
        self,
        query: str,
        documents: List[Document],
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Perform semantic search"""
        # Embed query
        query_embedding = self.embed_query(query)
        
        # Embed documents
        document_texts = [doc.page_content for doc in documents]
        document_embeddings = self.embeddings.embed_documents(document_texts)
        
        # Calculate similarities
        similarities = []
        for i, doc_embedding in enumerate(document_embeddings):
            similarity = cosine_similarity(
                [query_embedding],
                [doc_embedding]
            )[0][0]
            similarities.append((documents[i], similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]

class KeywordSearch:
    """Keyword search using TF-IDF"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)  # Include unigrams and bigrams
        )
        self.fitted = False
    
    def fit(self, documents: List[Document]):
        """Fit vectorizer on documents"""
        texts = [doc.page_content for doc in documents]
        self.vectorizer.fit(texts)
        self.fitted = True
    
    def search(
        self,
        query: str,
        documents: List[Document],
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Perform keyword search"""
        if not self.fitted:
            self.fit(documents)
        
        # Transform query and documents
        query_vec = self.vectorizer.transform([query])
        doc_texts = [doc.page_content for doc in documents]
        doc_vecs = self.vectorizer.transform(doc_texts)
        
        # Calculate similarities
        similarities = cosine_similarity(query_vec, doc_vecs)[0]
        
        # Pair with documents and sort
        results = list(zip(documents, similarities))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]

class BM25Search:
    """BM25 search algorithm"""
    
    def __init__(self, preprocessor: TextPreprocessor = None):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.bm25 = None
        self.documents = []
    
    def fit(self, documents: List[Document]):
        """Fit BM25 on documents"""
        self.documents = documents
        
        # Tokenize documents
        tokenized_docs = []
        for doc in documents:
            processed = self.preprocessor.preprocess(doc.page_content)
            tokens = processed.split()
            tokenized_docs.append(tokens)
        
        # Create BM25 model
        self.bm25 = BM25Okapi(tokenized_docs)
    
    def search(
        self,
        query: str,
        documents: List[Document] = None,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """Perform BM25 search"""
        if self.bm25 is None and documents:
            self.fit(documents)
        elif self.bm25 is None:
            raise ValueError("BM25 not fitted and no documents provided")
        
        # Tokenize query
        processed_query = self.preprocessor.preprocess(query)
        query_tokens = processed_query.split()
        
        # Get scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Use provided documents or fitted documents
        target_docs = documents if documents else self.documents
        
        # Pair with documents and sort
        results = list(zip(target_docs, scores))
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]

# ========== Hybrid Search System ==========

class HybridSearchSystem:
    """
    Advanced hybrid search system combining multiple algorithms.
    
    Features:
    - Semantic search (embeddings)
    - Keyword search (TF-IDF)
    - BM25 search
    - Weighted fusion
    - Query expansion
    - Result re-ranking
    """
    
    def __init__(
        self,
        config: HybridSearchConfig = None,
        embedding_model: str = "openai"
    ):
        self.config = config or HybridSearchConfig()
        self.config.validate()
        
        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.semantic_search = SemanticSearch(embedding_model)
        self.keyword_search = KeywordSearch()
        self.bm25_search = BM25Search(self.preprocessor)
        
        # State
        self.documents: List[Document] = []
        self.fitted = False
        self.vector_store = None
        
        print(f"✅ Hybrid Search System initialized")
        print(f"   Weights: Semantic={self.config.semantic_weight}, "
              f"Keyword={self.config.keyword_weight}, "
              f"BM25={self.config.bm25_weight}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to search system"""
        self.documents.extend(documents)
        
        # Fit search algorithms
        if documents:
            self.keyword_search.fit(self.documents)
            self.bm25_search.fit(self.documents)
            self.fitted = True
        
        print(f"✅ Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def create_vector_store(self, persist_directory: str = "./data/hybrid_search"):
        """Create vector store for semantic search"""
        if not self.documents:
            raise ValueError("No documents to create vector store")
        
        os.makedirs(persist_directory, exist_ok=True)
        
        # Create ChromaDB vector store
        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=self.semantic_search.embeddings,
            persist_directory=persist_directory,
            collection_name="hybrid_search"
        )
        
        self.vector_store.persist()
        print(f"✅ Vector store created with {len(self.documents)} documents")
    
    def search(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        k: Optional[int] = None,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Perform search using specified algorithm.
        
        Args:
            query: Search query
            search_type: Type of search to perform
            k: Number of results (overrides config)
            filters: Metadata filters
            
        Returns:
            List of SearchResult objects
        """
        if not self.documents:
            raise ValueError("No documents available for search")
        
        k = k or self.config.k
        
        # Filter documents if filters provided
        target_docs = self.documents
        if filters:
            target_docs = self._filter_documents(filters)
        
        # Expand query for better results
        expanded_query = self.preprocessor.expand_query(query)
        
        # Perform search based on type
        if search_type == SearchType.SEMANTIC:
            results = self._semantic_search(expanded_query, target_docs, k)
        
        elif search_type == SearchType.KEYWORD:
            results = self._keyword_search(expanded_query, target_docs, k)
        
        elif search_type == SearchType.BM25:
            results = self._bm25_search(expanded_query, target_docs, k)
        
        elif search_type == SearchType.HYBRID:
            results = self._hybrid_search(expanded_query, target_docs, k)
        
        elif search_type == SearchType.DENSE:
            results = self._dense_search(expanded_query, target_docs, k)
        
        else:
            raise ValueError(f"Unsupported search type: {search_type}")
        
        # Apply score threshold
        results = [r for r in results if r.final_score >= self.config.score_threshold]
        
        # Re-rank if enabled
        if self.config.use_reranking and len(results) > 1:
            results = self._rerank_results(results, query)
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results
    
    def _semantic_search(
        self,
        query: str,
        documents: List[Document],
        k: int
    ) -> List[SearchResult]:
        """Perform semantic search"""
        # Use vector store if available
        if self.vector_store:
            search_results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            results = []
            for doc, score in search_results:
                result = SearchResult(
                    document=doc,
                    semantic_score=score,
                    final_score=score
                )
                results.append(result)
            
            return results
        
        # Fallback to direct semantic search
        search_results = self.semantic_search.search(query, documents, k)
        
        results = []
        for doc, score in search_results:
            result = SearchResult(
                document=doc,
                semantic_score=score,
                final_score=score
            )
            results.append(result)
        
        return results
    
    def _keyword_search(
        self,
        query: str,
        documents: List[Document],
        k: int
    ) -> List[SearchResult]:
        """Perform keyword search"""
        search_results = self.keyword_search.search(query, documents, k)
        
        results = []
        for doc, score in search_results:
            result = SearchResult(
                document=doc,
                keyword_score=score,
                final_score=score
            )
            results.append(result)
        
        return results
    
    def _bm25_search(
        self,
        query: str,
        documents: List[Document],
        k: int
    ) -> List[SearchResult]:
        """Perform BM25 search"""
        search_results = self.bm25_search.search(query, documents, k)
        
        results = []
        for doc, score in search_results:
            result = SearchResult(
                document=doc,
                bm25_score=score,
                final_score=score
            )
            results.append(result)
        
        return results
    
    def _hybrid_search(
        self,
        query: str,
        documents: List[Document],
        k: int
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining multiple algorithms.
        
        Uses Reciprocal Rank Fusion (RRF) for combining results.
        """
        # Get results from each algorithm
        semantic_results = self._semantic_search(query, documents, k * 2)
        keyword_results = self._keyword_search(query, documents, k * 2)
        bm25_results = self._bm25_search(query, documents, k * 2)
        
        # Create result dictionaries
        result_dict = {}
        
        # Add semantic results
        for rank, result in enumerate(semantic_results):
            doc_id = id(result.document)
            if doc_id not in result_dict:
                result_dict[doc_id] = {
                    'document': result.document,
                    'semantic_score': result.semantic_score,
                    'semantic_rank': rank + 1,
                    'keyword_score': 0,
                    'keyword_rank': len(keyword_results) + 1,
                    'bm25_score': 0,
                    'bm25_rank': len(bm25_results) + 1
                }
            else:
                result_dict[doc_id]['semantic_score'] = result.semantic_score
                result_dict[doc_id]['semantic_rank'] = rank + 1
        
        # Add keyword results
        for rank, result in enumerate(keyword_results):
            doc_id = id(result.document)
            if doc_id not in result_dict:
                result_dict[doc_id] = {
                    'document': result.document,
                    'semantic_score': 0,
                    'semantic_rank': len(semantic_results) + 1,
                    'keyword_score': result.keyword_score,
                    'keyword_rank': rank + 1,
                    'bm25_score': 0,
                    'bm25_rank': len(bm25_results) + 1
                }
            else:
                result_dict[doc_id]['keyword_score'] = result.keyword_score
                result_dict[doc_id]['keyword_rank'] = rank + 1
        
        # Add BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = id(result.document)
            if doc_id not in result_dict:
                result_dict[doc_id] = {
                    'document': result.document,
                    'semantic_score': 0,
                    'semantic_rank': len(semantic_results) + 1,
                    'keyword_score': 0,
                    'keyword_rank': len(keyword_results) + 1,
                    'bm25_score': result.bm25_score,
                    'bm25_rank': rank + 1
                }
            else:
                result_dict[doc_id]['bm25_score'] = result.bm25_score
                result_dict[doc_id]['bm25_rank'] = rank + 1
        
        # Calculate hybrid scores using RRF
        results = []
        for doc_id, data in result_dict.items():
            # Reciprocal Rank Fusion
            rrf_score = (
                1 / (60 + data['semantic_rank']) +
                1 / (60 + data['keyword_rank']) +
                1 / (60 + data['bm25_rank'])
            )
            
            # Weighted average
            weighted_score = (
                data['semantic_score'] * self.config.semantic_weight +
                data['keyword_score'] * self.config.keyword_weight +
                data['bm25_score'] * self.config.bm25_weight
            )
            
            # Combined score
            combined_score = self.config.alpha * rrf_score + (1 - self.config.alpha) * weighted_score
            
            result = SearchResult(
                document=data['document'],
                semantic_score=data['semantic_score'],
                keyword_score=data['keyword_score'],
                bm25_score=data['bm25_score'],
                hybrid_score=combined_score,
                final_score=combined_score
            )
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results[:k]
    
    def _dense_search(
        self,
        query: str,
        documents: List[Document],
        k: int
    ) -> List[SearchResult]:
        """
        Perform dense search (semantic + keyword fusion).
        
        Combines embeddings and keyword matching for better results.
        """
        # Get semantic and keyword results
        semantic_results = self._semantic_search(query, documents, k * 2)
        keyword_results = self._keyword_search(query, documents, k * 2)
        
        # Create combined scores
        combined_dict = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = id(result.document)
            combined_dict[doc_id] = {
                'document': result.document,
                'semantic_score': result.semantic_score,
                'keyword_score': 0
            }
        
        # Add keyword results
        for result in keyword_results:
            doc_id = id(result.document)
            if doc_id in combined_dict:
                combined_dict[doc_id]['keyword_score'] = result.keyword_score
            else:
                combined_dict[doc_id] = {
                    'document': result.document,
                    'semantic_score': 0,
                    'keyword_score': result.keyword_score
                }
        
        # Calculate dense scores
        results = []
        for data in combined_dict.values():
            # Geometric mean of scores
            if data['semantic_score'] > 0 and data['keyword_score'] > 0:
                dense_score = math.sqrt(data['semantic_score'] * data['keyword_score'])
            else:
                dense_score = max(data['semantic_score'], data['keyword_score'])
            
            result = SearchResult(
                document=data['document'],
                semantic_score=data['semantic_score'],
                keyword_score=data['keyword_score'],
                final_score=dense_score
            )
            results.append(result)
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results[:k]
    
    def _rerank_results(
        self,
        results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """
        Re-rank results using cross-encoder or advanced scoring.
        
        Args:
            results: Initial search results
            query: Original query
            
        Returns:
            Re-ranked results
        """
        if len(results) <= 1:
            return results
        
        # Simple re-ranking based on query-document relevance
        for result in results:
            # Calculate additional relevance factors
            doc_text = result.document.page_content.lower()
            query_terms = query.lower().split()
            
            # Term frequency in document
            term_freq_score = 0
            for term in query_terms:
                if len(term) > 2:  # Ignore short terms
                    term_freq_score += doc_text.count(term)
            
            # Position of first occurrence
            first_pos = float('inf')
            for term in query_terms:
                if len(term) > 2:
                    pos = doc_text.find(term)
                    if pos != -1:
                        first_pos = min(first_pos, pos)
            
            position_score = 1.0 / (1.0 + first_pos) if first_pos != float('inf') else 0
            
            # Update final score with re-ranking factors
            result.final_score = result.final_score * 0.7 + term_freq_score * 0.2 + position_score * 0.1
        
        # Re-sort by updated final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results[:self.config.rerank_top_k]
    
    def _filter_documents(self, filters: Dict) -> List[Document]:
        """Filter documents based on metadata"""
        filtered_docs = []
        
        for doc in self.documents:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata or doc.metadata[key] != value:
                    match = False
                    break
            
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def search_with_explain(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Perform search with detailed explanation of results.
        
        Returns:
            Dictionary with results and explanation
        """
        results = self.search(query, search_type, k)
        
        explanation = {
            "query": query,
            "search_type": search_type.value,
            "total_documents": len(self.documents),
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "keyword_weight": self.config.keyword_weight,
                "bm25_weight": self.config.bm25_weight,
                "alpha": self.config.alpha
            },
            "results": [],
            "statistics": {
                "avg_semantic_score": 0.0,
                "avg_keyword_score": 0.0,
                "avg_final_score": 0.0,
                "score_distribution": []
            }
        }
        
        # Add results
        for result in results:
            explanation["results"].append(result.to_dict())
        
        # Calculate statistics
        if results:
            explanation["statistics"]["avg_semantic_score"] = np.mean([r.semantic_score for r in results])
            explanation["statistics"]["avg_keyword_score"] = np.mean([r.keyword_score for r in results])
            explanation["statistics"]["avg_final_score"] = np.mean([r.final_score for r in results])
            
            # Score distribution
            score_ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            for low, high in score_ranges:
                count = sum(1 for r in results if low <= r.final_score < high)
                explanation["statistics"]["score_distribution"].append({
                    "range": f"{low:.1f}-{high:.1f}",
                    "count": count,
                    "percentage": (count / len(results)) * 100
                })
        
        return explanation
    
    def benchmark_search(
        self,
        queries: List[str],
        search_types: List[SearchType] = None
    ) -> Dict[str, Any]:
        """
        Benchmark different search algorithms.
        
        Args:
            queries: List of test queries
            search_types: List of search types to benchmark
            
        Returns:
            Benchmark results
        """
        if search_types is None:
            search_types = [
                SearchType.SEMANTIC,
                SearchType.KEYWORD,
                SearchType.BM25,
                SearchType.HYBRID,
                SearchType.DENSE
            ]
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(queries),
            "search_types": [st.value for st in search_types],
            "results": {}
        }
        
        import time
        
        for search_type in search_types:
            print(f"📊 Benchmarking {search_type.value}...")
            
            type_results = {
                "query_times": [],
                "avg_scores": [],
                "total_results": []
            }
            
            for query in queries:
                start_time = time.time()
                results = self.search(query, search_type, k=5)
                query_time = time.time() - start_time
                
                type_results["query_times"].append(query_time)
                if results:
                    type_results["avg_scores"].append(np.mean([r.final_score for r in results]))
                type_results["total_results"].append(len(results))
            
            # Calculate statistics
            benchmark_results["results"][search_type.value] = {
                "avg_query_time_ms": np.mean(type_results["query_times"]) * 1000,
                "std_query_time_ms": np.std(type_results["query_times"]) * 1000,
                "avg_score": np.mean(type_results["avg_scores"]) if type_results["avg_scores"] else 0,
                "avg_results": np.mean(type_results["total_results"]),
                "total_queries_completed": len(queries)
            }
        
        return benchmark_results

# ========== Example Usage ==========

def create_sample_documents():
    """Create sample documents for demonstration"""
    samples = [
        {
            "content": """
            Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
            Machine learning is a subset of AI that focuses on building systems that learn from data.
            Deep learning uses neural networks with many layers to model complex patterns.
            """,
            "metadata": {"source": "ai_basics.txt", "category": "technology", "year": 2023}
        },
        {
            "content": """
            Python is a high-level programming language known for its simplicity and readability.
            It's widely used in data science, web development, and automation.
            Python supports multiple programming paradigms including object-oriented and functional programming.
            """,
            "metadata": {"source": "python_guide.txt", "category": "programming", "year": 2023}
        },
        {
            "content": """
            Vector databases store data as high-dimensional vectors for efficient similarity search.
            They are essential for applications like recommendation systems and semantic search.
            Popular vector databases include Pinecone, Weaviate, and ChromaDB.
            """,
            "metadata": {"source": "vector_databases.txt", "category": "database", "year": 2024}
        },
        {
            "content": """
            Large Language Models (LLMs) like GPT-4 are trained on vast amounts of text data.
            They can generate human-like text, translate languages, and answer questions.
            Fine-tuning adapts pre-trained models to specific tasks or domains.
            """,
            "metadata": {"source": "llm_overview.txt", "category": "ai", "year": 2024}
        },
        {
            "content": """
            Retrieval-Augmented Generation (RAG) combines retrieval and generation for better AI answers.
            It retrieves relevant documents and uses them to generate accurate, sourced responses.
            RAG reduces hallucinations and improves factuality in AI systems.
            """,
            "metadata": {"source": "rag_explained.txt", "category": "ai", "year": 2024}
        }
    ]
    
    documents = []
    for sample in samples:
        doc = Document(
            page_content=sample["content"],
            metadata=sample["metadata"]
        )
        documents.append(doc)
    
    return documents

def demo_hybrid_search():
    """Demonstrate hybrid search system"""
    print("=" * 60)
    print("🧪 DEMO: Hybrid Search System")
    print("=" * 60)
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Configure hybrid search
    config = HybridSearchConfig(
        semantic_weight=0.5,
        keyword_weight=0.3,
        bm25_weight=0.2,
        alpha=0.7,
        k=5,
        use_reranking=True
    )
    
    # Initialize system
    search_system = HybridSearchSystem(config)
    search_system.add_documents(documents)
    search_system.create_vector_store()
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How do vector databases work?",
        "Explain retrieval-augmented generation",
        "Python programming language features"
    ]
    
    # Test different search types
    search_types = [
        (SearchType.SEMANTIC, "🔍 Semantic Search"),
        (SearchType.KEYWORD, "🔤 Keyword Search"),
        (SearchType.BM25, "📊 BM25 Search"),
        (SearchType.HYBRID, "🤝 Hybrid Search"),
        (SearchType.DENSE, "🎯 Dense Search")
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        for search_type, label in search_types:
            results = search_system.search(query, search_type, k=3)
            
            print(f"\n{label}:")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.document.metadata['source']}")
                print(f"     Score: {result.final_score:.4f}")
                print(f"     Preview: {result.document.page_content[:80]}...")
        
        print("-" * 40)
    
    # Demonstrate detailed explanation
    print("\n📊 Detailed Search Explanation:")
    print("=" * 40)
    
    explanation = search_system.search_with_explain(
        "What are large language models?",
        SearchType.HYBRID,
        k=3
    )
    
    print(f"Query: {explanation['query']}")
    print(f"Search Type: {explanation['search_type']}")
    print(f"Config: {explanation['config']}")
    
    print("\nResults:")
    for result in explanation['results']:
        print(f"  • {result['source']} (Score: {result['final_score']})")
        print(f"    Semantic: {result['semantic_score']}, Keyword: {result['keyword_score']}")
    
    print(f"\nStatistics:")
    print(f"  Average Final Score: {explanation['statistics']['avg_final_score']:.4f}")
    for dist in explanation['statistics']['score_distribution']:
        print(f"  {dist['range']}: {dist['count']} results ({dist['percentage']:.1f}%)")
    
    # Run benchmark
    print("\n📈 Running Benchmark...")
    benchmark = search_system.benchmark_search(test_queries)
    
    print("\nBenchmark Results:")
    for search_type, results in benchmark['results'].items():
        print(f"\n  {search_type.upper()}:")
        print(f"    Avg Query Time: {results['avg_query_time_ms']:.2f}ms")
        print(f"    Avg Score: {results['avg_score']:.4f}")
        print(f"    Avg Results: {results['avg_results']:.1f}")
    
    print("\n✅ Demo completed!")
    print("=" * 60)

def interactive_search():
    """Interactive search demo"""
    print("🤖 Interactive Hybrid Search Demo")
    print("=" * 50)
    
    # Initialize with sample documents
    search_system = HybridSearchSystem()
    search_system.add_documents(create_sample_documents())
    
    print("\nAvailable search types:")
    print("  1. semantic - Search using embeddings")
    print("  2. keyword - Search using keyword matching")
    print("  3. bm25 - Search using BM25 algorithm")
    print("  4. hybrid - Combined search (default)")
    print("  5. dense - Dense semantic search")
    
    print("\nCommands:")
    print("  Type 'quit' to exit")
    print("  Type 'config' to see current configuration")
    print("  Type 'stats' to see document statistics")
    print("  Type 'benchmark' to run benchmark")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🔍 Enter search query: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'config':
                config = search_system.config
                print(f"\n📋 Current Configuration:")
                print(f"  Semantic Weight: {config.semantic_weight}")
                print(f"  Keyword Weight: {config.keyword_weight}")
                print(f"  BM25 Weight: {config.bm25_weight}")
                print(f"  Alpha: {config.alpha}")
                print(f"  K: {config.k}")
                print(f"  Use Re-ranking: {config.use_reranking}")
                continue
            
            elif user_input.lower() == 'stats':
                print(f"\n📊 Document Statistics:")
                print(f"  Total Documents: {len(search_system.documents)}")
                
                # Count by category
                categories = {}
                for doc in search_system.documents:
                    category = doc.metadata.get('category', 'unknown')
                    categories[category] = categories.get(category, 0) + 1
                
                for category, count in categories.items():
                    print(f"  {category}: {count} documents")
                continue
            
            elif user_input.lower() == 'benchmark':
                queries = [
                    "artificial intelligence",
                    "machine learning",
                    "vector databases",
                    "python programming"
                ]
                
                print("\n📈 Running benchmark with 4 queries...")
                benchmark = search_system.benchmark_search(queries)
                
                print("\nBenchmark Results:")
                for search_type, results in benchmark['results'].items():
                    print(f"\n  {search_type}:")
                    print(f"    Avg Time: {results['avg_query_time_ms']:.2f}ms")
                    print(f"    Avg Score: {results['avg_score']:.4f}")
                continue
            
            if not user_input:
                continue
            
            # Parse search type if specified
            search_type = SearchType.HYBRID
            query = user_input
            
            if user_input.startswith('/'):
                parts = user_input[1:].split(' ', 1)
                if len(parts) == 2:
                    type_str, query = parts
                    try:
                        search_type = SearchType(type_str.lower())
                    except:
                        print(f"⚠️  Unknown search type: {type_str}")
                        print("   Available: semantic, keyword, bm25, hybrid, dense")
                        continue
            
            # Perform search
            print(f"\n🔎 Searching for: {query}")
            print(f"   Type: {search_type.value}")
            
            results = search_system.search(query, search_type, k=5)
            
            if not results:
                print("   No results found.")
                continue
            
            print(f"\n📄 Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                print(f"\n  {i}. {result.document.metadata['source']}")
                print(f"     Score: {result.final_score:.4f}")
                print(f"     Category: {result.document.metadata.get('category', 'N/A')}")
                
                # Show scores breakdown for hybrid search
                if search_type == SearchType.HYBRID:
                    print(f"     Breakdown: Semantic={result.semantic_score:.3f}, "
                          f"Keyword={result.keyword_score:.3f}, "
                          f"BM25={result.bm25_score:.3f}")
                
                # Show preview
                preview = result.document.page_content[:100].replace('\n', ' ')
                print(f"     Preview: {preview}...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# ========== Main Execution ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid Search System")
    parser.add_argument("--mode", choices=["demo", "interactive", "benchmark"], 
                       default="demo", help="Mode to run")
    parser.add_argument("--folder", type=str, help="Folder with documents to load")
    parser.add_argument("--query", type=str, help="Single query to test")
    parser.add_argument("--type", type=str, default="hybrid", 
                       help="Search type: semantic, keyword, bm25, hybrid, dense")
    
    args = parser.parse_args()
    
    # Check OpenAI API key for semantic search
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  OPENAI_API_KEY not found. Some features may be limited.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
    
    # Run based on mode
    if args.mode == "demo":
        demo_hybrid_search()
    
    elif args.mode == "interactive":
        interactive_search()
    
    elif args.mode == "benchmark":
        # Initialize with sample documents
        search_system = HybridSearchSystem()
        search_system.add_documents(create_sample_documents())
        
        # Run benchmark
        queries = [
            "artificial intelligence and machine learning",
            "vector databases for similarity search",
            "python programming language features",
            "large language models and RAG systems"
        ]
        
        print("📊 Running comprehensive benchmark...")
        benchmark = search_system.benchmark_search(queries)
        
        # Print results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        for search_type, results in benchmark['results'].items():
            print(f"\n{search_type.upper()}:")
            print(f"  Average Query Time: {results['avg_query_time_ms']:.2f}ms")
            print(f"  Standard Deviation: {results['std_query_time_ms']:.2f}ms")
            print(f"  Average Score: {results['avg_score']:.4f}")
            print(f"  Average Results per Query: {results['avg_results']:.1f}")
        
        # Determine winner
        winner = min(
            benchmark['results'].items(),
            key=lambda x: x[1]['avg_query_time_ms']
        )
        
        print(f"\n🏆 Fastest: {winner[0]} ({winner[1]['avg_query_time_ms']:.2f}ms)")
    
    # Handle single query
    if args.query:
        search_system = HybridSearchSystem()
        
        if args.folder:
            # Load documents from folder
            import glob
            from langchain.document_loaders import TextLoader
            
            documents = []
            for file_path in glob.glob(os.path.join(args.folder, "*.txt")):
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = os.path.basename(file_path)
                    documents.extend(docs)
                except Exception as e:
                    print(f"⚠️  Error loading {file_path}: {e}")
            
            search_system.add_documents(documents)
        else:
            # Use sample documents
            search_system.add_documents(create_sample_documents())
        
        # Perform search
        try:
            search_type = SearchType(args.type.lower())
        except:
            print(f"⚠️  Invalid search type: {args.type}")
            print("   Using hybrid search instead")
            search_type = SearchType.HYBRID
        
        results = search_system.search(args.query, search_type, k=5)
        
        print(f"\n🔍 Query: {args.query}")
        print(f"🔧 Search Type: {search_type.value}")
        print(f"📄 Results Found: {len(results)}")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.document.metadata.get('source', 'Unknown')}")
            print(f"   Score: {result.final_score:.4f}")
            print(f"   Content: {result.document.page_content[:150]}...")