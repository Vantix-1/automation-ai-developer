"""
Advanced Re-ranking System for RAG
Improves search results quality with sophisticated re-ranking algorithms.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
from collections import Counter, defaultdict
import hashlib
from scipy import stats

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# ========== Enums and Data Classes ==========

class RerankerType(Enum):
    """Types of re-ranking algorithms"""
    CROSS_ENCODER = "cross_encoder"
    BERT_SCORE = "bert_score"
    TFIDF_RERANK = "tfidf_rerank"
    SEMANTIC_RERANK = "semantic_rerank"
    HYBRID_RERANK = "hybrid_rerank"
    POSITIONAL_RERANK = "positional_rerank"
    DIVERSITY_RERANK = "diversity_rerank"
    ENSEMBLE = "ensemble"

@dataclass
class RerankerConfig:
    """Configuration for re-ranking"""
    reranker_type: RerankerType = RerankerType.CROSS_ENCODER
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k: int = 5  # Re-rank top K results
    diversity_weight: float = 0.3
    position_weight: float = 0.2
    freshness_weight: float = 0.1
    similarity_weight: float = 0.4
    min_score: float = 0.0
    max_results: int = 10
    use_cache: bool = True
    cache_size: int = 1000
    
    def __post_init__(self):
        weights_sum = (
            self.diversity_weight + 
            self.position_weight + 
            self.freshness_weight + 
            self.similarity_weight
        )
        if abs(weights_sum - 1.0) > 0.01:
            raise ValueError(f"Reranking weights must sum to 1.0, got {weights_sum}")

@dataclass
class RerankedResult:
    """Enhanced result after re-ranking"""
    document_id: str
    content: str
    original_score: float
    reranked_score: float
    improvement: float
    rank_change: int  # Positive = improved, Negative = worsened
    factors: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "document_id": self.document_id,
            "content_preview": self.content[:200] + "...",
            "original_score": round(self.original_score, 4),
            "reranked_score": round(self.reranked_score, 4),
            "improvement": round(self.improvement, 4),
            "rank_change": self.rank_change,
            "factors": {k: round(v, 4) for k, v in self.factors.items()},
            "metadata": self.metadata
        }

# ========== Base Reranker Class ==========

class BaseReranker:
    """Base class for all re-ranking algorithms"""
    
    def __init__(self, config: RerankerConfig = None):
        self.config = config or RerankerConfig()
        self.cache = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _get_cache_key(self, query: str, documents: List[str]) -> str:
        """Generate cache key for query-documents pair"""
        content = query + "||" + "||".join(documents)
        return hashlib.md5(content.encode()).hexdigest()
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RerankedResult]:
        """
        Re-rank documents based on query relevance.
        
        Args:
            query: Search query
            documents: List of document dictionaries with 'content' and 'metadata'
            original_scores: Original similarity scores
            
        Returns:
            List of re-ranked results
        """
        raise NotImplementedError("Subclasses must implement rerank method")
    
    def _create_results(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        new_scores: List[float],
        original_scores: List[float],
        factor_breakdowns: Optional[List[Dict[str, float]]] = None
    ) -> List[RerankedResult]:
        """Create RerankedResult objects from scores"""
        # Pair documents with scores
        doc_items = []
        for i, (doc, new_score, orig_score) in enumerate(zip(documents, new_scores, original_scores)):
            factors = factor_breakdowns[i] if factor_breakdowns else {}
            
            # Generate document ID if not present
            doc_id = doc.get('id') or doc.get('document_id') or f"doc_{i}"
            
            doc_items.append({
                'index': i,
                'document': doc,
                'new_score': new_score,
                'orig_score': orig_score,
                'factors': factors,
                'doc_id': doc_id
            })
        
        # Sort by new scores
        doc_items.sort(key=lambda x: x['new_score'], reverse=True)
        
        # Create results with rank changes
        results = []
        for new_rank, item in enumerate(doc_items):
            # Find original rank
            original_rank = next(
                i for i, d in enumerate(sorted(
                    doc_items,
                    key=lambda x: x['orig_score'],
                    reverse=True
                ))
                if d['index'] == item['index']
            )
            
            rank_change = original_rank - new_rank  # Positive = improved
            
            result = RerankedResult(
                document_id=item['doc_id'],
                content=item['document'].get('content', ''),
                original_score=item['orig_score'],
                reranked_score=item['new_score'],
                improvement=item['new_score'] - item['orig_score'],
                rank_change=rank_change,
                factors=item['factors'],
                metadata=item['document'].get('metadata', {})
            )
            
            results.append(result)
        
        return results[:self.config.max_results]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get re-ranking statistics"""
        return {
            "cache_hits": self.hit_count,
            "cache_misses": self.miss_count,
            "cache_hit_rate": self.hit_count / max(self.hit_count + self.miss_count, 1),
            "cache_size": len(self.cache)
        }

# ========== Cross-Encoder Reranker ==========

class CrossEncoderReranker(BaseReranker):
    """Re-ranker using Sentence Transformers Cross-Encoder"""
    
    def __init__(self, config: RerankerConfig = None):
        super().__init__(config)
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load cross-encoder model"""
        try:
            self.model = CrossEncoder(self.config.model_name)
            print(f"✅ Loaded Cross-Encoder model: {self.config.model_name}")
        except Exception as e:
            print(f"❌ Failed to load Cross-Encoder: {e}")
            print("   Using TF-IDF fallback")
            self.model = None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RerankedResult]:
        """Re-rank using cross-encoder"""
        if self.model is None:
            # Fallback to TF-IDF
            fallback = TFIDFReranker(self.config)
            return fallback.rerank(query, documents, original_scores)
        
        # Check cache
        if self.config.use_cache:
            cache_key = self._get_cache_key(query, [d.get('content', '') for d in documents])
            if cache_key in self.cache:
                self.hit_count += 1
                return self.cache[cache_key]
            self.miss_count += 1
        
        # Prepare document texts
        doc_texts = [doc.get('content', '') for doc in documents]
        
        # Create query-document pairs
        pairs = [(query, doc_text) for doc_text in doc_texts]
        
        # Get scores from cross-encoder
        try:
            scores = self.model.predict(pairs)
        except Exception as e:
            print(f"❌ Cross-encoder prediction failed: {e}")
            scores = original_scores if original_scores else [0.5] * len(documents)
        
        # Normalize scores to [0, 1]
        if len(scores) > 0:
            min_score, max_score = min(scores), max(scores)
            if max_score > min_score:
                scores = [(s - min_score) / (max_score - min_score) for s in scores]
        
        # Use original scores if provided, else use default
        if original_scores is None:
            original_scores = scores.copy()
        
        # Create results
        results = self._create_results(query, documents, scores, original_scores)
        
        # Cache results
        if self.config.use_cache:
            self.cache[cache_key] = results
            # Limit cache size
            if len(self.cache) > self.config.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return results

# ========== TF-IDF Reranker ==========

class TFIDFReranker(BaseReranker):
    """Re-ranker using TF-IDF and advanced text features"""
    
    def __init__(self, config: RerankerConfig = None):
        super().__init__(config)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.stop_words = set(stopwords.words('english'))
        self.fitted = False
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_features(self, query: str, document: str) -> Dict[str, float]:
        """Extract advanced text features"""
        features = {}
        
        # Preprocess
        query_processed = self._preprocess_text(query)
        doc_processed = self._preprocess_text(document)
        
        # 1. Term overlap
        query_terms = set(query_processed.split())
        doc_terms = set(doc_processed.split())
        overlap = len(query_terms.intersection(doc_terms))
        features['term_overlap'] = overlap / max(len(query_terms), 1)
        
        # 2. Query term density
        total_terms = len(doc_terms)
        if total_terms > 0:
            features['query_density'] = overlap / total_terms
        else:
            features['query_density'] = 0
        
        # 3. Bigram overlap
        query_bigrams = set(zip(query_processed.split()[:-1], query_processed.split()[1:]))
        doc_bigrams = set(zip(doc_processed.split()[:-1], doc_processed.split()[1:]))
        bigram_overlap = len(query_bigrams.intersection(doc_bigrams))
        features['bigram_overlap'] = bigram_overlap / max(len(query_bigrams), 1)
        
        # 4. Position of first query term
        if query_terms:
            first_positions = []
            for term in query_terms:
                pos = doc_processed.find(term)
                if pos != -1:
                    first_positions.append(pos)
            
            if first_positions:
                avg_position = sum(first_positions) / len(first_positions)
                features['avg_first_position'] = 1.0 / (1.0 + avg_position / 100)
            else:
                features['avg_first_position'] = 0
        else:
            features['avg_first_position'] = 0
        
        # 5. Document length factor
        doc_length = len(doc_processed.split())
        # Prefer medium-length documents (not too short, not too long)
        if doc_length < 50:
            features['length_factor'] = 0.7
        elif doc_length < 500:
            features['length_factor'] = 1.0
        else:
            features['length_factor'] = 0.8
        
        return features
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RerankedResult]:
        """Re-rank using TF-IDF and text features"""
        # Check cache
        if self.config.use_cache:
            cache_key = self._get_cache_key(query, [d.get('content', '') for d in documents])
            if cache_key in self.cache:
                self.hit_count += 1
                return self.cache[cache_key]
            self.miss_count += 1
        
        # Extract document texts
        doc_texts = [doc.get('content', '') for doc in documents]
        
        # Fit vectorizer if needed
        if not self.fitted:
            self.vectorizer.fit(doc_texts)
            self.fitted = True
        
        # Transform query and documents
        query_vec = self.vectorizer.transform([query])
        doc_vecs = self.vectorizer.transform(doc_texts)
        
        # Calculate TF-IDF similarities
        tfidf_similarities = cosine_similarity(query_vec, doc_vecs)[0]
        
        # Calculate feature scores
        feature_scores = []
        factor_breakdowns = []
        
        for i, doc in enumerate(documents):
            features = self._extract_features(query, doc_texts[i])
            
            # Calculate weighted feature score
            feature_score = (
                features.get('term_overlap', 0) * 0.3 +
                features.get('query_density', 0) * 0.2 +
                features.get('bigram_overlap', 0) * 0.2 +
                features.get('avg_first_position', 0) * 0.2 +
                features.get('length_factor', 0) * 0.1
            )
            
            # Combine with TF-IDF similarity
            combined_score = (
                tfidf_similarities[i] * 0.6 +
                feature_score * 0.4
            )
            
            feature_scores.append(combined_score)
            factor_breakdowns.append(features)
        
        # Normalize scores
        if len(feature_scores) > 0:
            min_score, max_score = min(feature_scores), max(feature_scores)
            if max_score > min_score:
                feature_scores = [(s - min_score) / (max_score - min_score) for s in feature_scores]
        
        # Use original scores if provided
        if original_scores is None:
            original_scores = feature_scores.copy()
        
        # Create results
        results = self._create_results(query, documents, feature_scores, original_scores, factor_breakdowns)
        
        # Cache results
        if self.config.use_cache:
            self.cache[cache_key] = results
            if len(self.cache) > self.config.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return results

# ========== Semantic Reranker ==========

class SemanticReranker(BaseReranker):
    """Re-ranker using semantic embeddings"""
    
    def __init__(self, config: RerankerConfig = None):
        super().__init__(config)
        self.embeddings = self._load_embeddings()
    
    def _load_embeddings(self):
        """Load embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Loaded Sentence Transformer model")
            return model
        except Exception as e:
            print(f"❌ Failed to load Sentence Transformer: {e}")
            return None
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RerankedResult]:
        """Re-rank using semantic embeddings"""
        if self.embeddings is None:
            # Fallback
            fallback = TFIDFReranker(self.config)
            return fallback.rerank(query, documents, original_scores)
        
        # Check cache
        if self.config.use_cache:
            cache_key = self._get_cache_key(query, [d.get('content', '') for d in documents])
            if cache_key in self.cache:
                self.hit_count += 1
                return self.cache[cache_key]
            self.miss_count += 1
        
        # Encode query and documents
        doc_texts = [doc.get('content', '') for doc in documents]
        
        try:
            query_embedding = self.embeddings.encode([query])[0]
            doc_embeddings = self.embeddings.encode(doc_texts)
            
            # Calculate cosine similarities
            similarities = []
            for doc_embedding in doc_embeddings:
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append(similarity)
            
            # Normalize to [0, 1]
            if len(similarities) > 0:
                min_sim, max_sim = min(similarities), max(similarities)
                if max_sim > min_sim:
                    similarities = [(s - min_sim) / (max_sim - min_sim) for s in similarities]
            
            semantic_scores = similarities
            
        except Exception as e:
            print(f"❌ Semantic encoding failed: {e}")
            semantic_scores = original_scores if original_scores else [0.5] * len(documents)
        
        # Use original scores if provided
        if original_scores is None:
            original_scores = semantic_scores.copy()
        
        # Create results
        results = self._create_results(query, documents, semantic_scores, original_scores)
        
        # Cache results
        if self.config.use_cache:
            self.cache[cache_key] = results
            if len(self.cache) > self.config.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
        
        return results

# ========== Hybrid Reranker ==========

class HybridReranker(BaseReranker):
    """Hybrid re-ranker combining multiple algorithms"""
    
    def __init__(self, config: RerankerConfig = None):
        super().__init__(config)
        
        # Initialize sub-rerankers
        self.rerankers = {
            RerankerType.CROSS_ENCODER: CrossEncoderReranker(config),
            RerankerType.TFIDF_RERANK: TFIDFReranker(config),
            RerankerType.SEMANTIC_RERANK: SemanticReranker(config)
        }
        
        # Default weights for ensemble
        self.ensemble_weights = {
            RerankerType.CROSS_ENCODER: 0.5,
            RerankerType.TFIDF_RERANK: 0.3,
            RerankerType.SEMANTIC_RERANK: 0.2
        }
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RerankedResult]:
        """Re-rank using hybrid approach"""
        if self.config.reranker_type == RerankerType.ENSEMBLE:
            return self._ensemble_rerank(query, documents, original_scores)
        elif self.config.reranker_type == RerankerType.HYBRID_RERANK:
            return self._hybrid_rerank(query, documents, original_scores)
        else:
            # Use single reranker
            reranker = self.rerankers.get(self.config.reranker_type)
            if reranker:
                return reranker.rerank(query, documents, original_scores)
            else:
                # Fallback to TF-IDF
                return self.rerankers[RerankerType.TFIDF_RERANK].rerank(
                    query, documents, original_scores
                )
    
    def _ensemble_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RerankedResult]:
        """Ensemble re-ranking combining multiple algorithms"""
        # Get scores from each reranker
        all_scores = []
        
        for reranker_type, reranker in self.rerankers.items():
            if reranker_type in self.ensemble_weights:
                results = reranker.rerank(query, documents, original_scores)
                scores = [r.reranked_score for r in results]
                all_scores.append((self.ensemble_weights[reranker_type], scores))
        
        # Weighted average of scores
        if not all_scores:
            return []
        
        weighted_scores = np.zeros(len(documents))
        total_weight = 0
        
        for weight, scores in all_scores:
            if len(scores) == len(documents):
                weighted_scores += np.array(scores) * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_scores = weighted_scores / total_weight
        else:
            ensemble_scores = np.array([0.5] * len(documents))
        
        # Normalize
        if len(ensemble_scores) > 0:
            min_score, max_score = min(ensemble_scores), max(ensemble_scores)
            if max_score > min_score:
                ensemble_scores = (ensemble_scores - min_score) / (max_score - min_score)
        
        # Use original scores if provided
        if original_scores is None:
            original_scores = ensemble_scores.copy()
        
        # Create results
        return self._create_results(query, documents, ensemble_scores.tolist(), original_scores)
    
    def _hybrid_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RerankedResult]:
        """Advanced hybrid re-ranking with multiple factors"""
        # Extract metadata and content
        doc_texts = [doc.get('content', '') for doc in documents]
        metadata_list = [doc.get('metadata', {}) for doc in documents]
        
        # Calculate multiple factors
        factors = []
        factor_breakdowns = []
        
        for i, (text, metadata) in enumerate(zip(doc_texts, metadata_list)):
            factor_scores = {}
            
            # 1. Content relevance (TF-IDF)
            relevance_score = self._calculate_relevance(query, text)
            factor_scores['relevance'] = relevance_score
            
            # 2. Freshness (based on metadata)
            freshness_score = self._calculate_freshness(metadata)
            factor_scores['freshness'] = freshness_score
            
            # 3. Authority (based on source metadata)
            authority_score = self._calculate_authority(metadata)
            factor_scores['authority'] = authority_score
            
            # 4. Diversity (avoid similar documents)
            diversity_score = self._calculate_diversity(text, doc_texts[:i])
            factor_scores['diversity'] = diversity_score
            
            # 5. Length factor
            length_score = self._calculate_length_factor(text)
            factor_scores['length'] = length_score
            
            # Combined score with weights
            combined_score = (
                relevance_score * self.config.similarity_weight +
                freshness_score * self.config.freshness_weight +
                authority_score * 0.2 +  # Fixed weight for authority
                diversity_score * self.config.diversity_weight +
                length_score * 0.1  # Fixed weight for length
            )
            
            factors.append(combined_score)
            factor_breakdowns.append(factor_scores)
        
        # Normalize scores
        if len(factors) > 0:
            min_score, max_score = min(factors), max(factors)
            if max_score > min_score:
                factors = [(s - min_score) / (max_score - min_score) for s in factors]
        
        # Use original scores if provided
        if original_scores is None:
            original_scores = factors.copy()
        
        # Create results
        return self._create_results(query, documents, factors, original_scores, factor_breakdowns)
    
    def _calculate_relevance(self, query: str, document: str) -> float:
        """Calculate relevance score using TF-IDF"""
        # Simple implementation - in production use proper TF-IDF
        query_terms = set(query.lower().split())
        doc_terms = set(document.lower().split())
        
        overlap = len(query_terms.intersection(doc_terms))
        return overlap / max(len(query_terms), 1)
    
    def _calculate_freshness(self, metadata: Dict) -> float:
        """Calculate freshness score based on date"""
        date_str = metadata.get('date') or metadata.get('created_at') or metadata.get('timestamp')
        
        if date_str:
            try:
                # Parse date and calculate freshness
                import dateutil.parser
                doc_date = dateutil.parser.parse(date_str)
                now = datetime.now()
                
                # Days since publication
                days_old = (now - doc_date).days
                
                # Score: 1.0 for today, decreasing over time
                freshness = 1.0 / (1.0 + days_old / 365)  # Half-life of 1 year
                return min(max(freshness, 0), 1)
            except:
                pass
        
        return 0.5  # Default score
    
    def _calculate_authority(self, metadata: Dict) -> float:
        """Calculate authority score based on source"""
        source = metadata.get('source', '').lower()
        
        # Simple authority scoring
        authority_scores = {
            'research': 0.9,
            'academic': 0.9,
            'government': 0.8,
            'official': 0.8,
            'news': 0.7,
            'blog': 0.5,
            'forum': 0.3,
            'unknown': 0.5
        }
        
        for key, score in authority_scores.items():
            if key in source:
                return score
        
        return 0.5  # Default
    
    def _calculate_diversity(self, text: str, previous_texts: List[str]) -> float:
        """Calculate diversity score (avoid redundancy)"""
        if not previous_texts:
            return 1.0  # First document is always diverse
        
        # Calculate similarity with previous documents
        similarities = []
        for prev_text in previous_texts[-3:]:  # Check last 3 documents
            # Simple Jaccard similarity
            terms1 = set(text.lower().split()[:50])  # First 50 terms
            terms2 = set(prev_text.lower().split()[:50])
            
            if terms1 and terms2:
                similarity = len(terms1.intersection(terms2)) / len(terms1.union(terms2))
                similarities.append(similarity)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            return 1.0 - avg_similarity  # Diversity = 1 - similarity
        else:
            return 1.0
    
    def _calculate_length_factor(self, text: str) -> float:
        """Calculate optimal length factor"""
        word_count = len(text.split())
        
        # Prefer documents with 50-500 words
        if word_count < 20:
            return 0.3  # Too short
        elif word_count < 50:
            return 0.7  # A bit short
        elif word_count < 500:
            return 1.0  # Optimal
        elif word_count < 1000:
            return 0.8  # A bit long
        else:
            return 0.5  # Too long

# ========== Positional Reranker ==========

class PositionalReranker(BaseReranker):
    """Re-ranker that considers document position and structure"""
    
    def __init__(self, config: RerankerConfig = None):
        super().__init__(config)
        self.stop_words = set(stopwords.words('english'))
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        original_scores: Optional[List[float]] = None
    ) -> List[RerankedResult]:
        """Re-rank considering positional information"""
        doc_texts = [doc.get('content', '') for doc in documents]
        
        # Calculate positional scores
        positional_scores = []
        factor_breakdowns = []
        
        for i, text in enumerate(doc_texts):
            factors = {}
            
            # 1. Query term position
            factors['position'] = self._calculate_position_score(query, text)
            
            # 2. Query term distribution
            factors['distribution'] = self._calculate_distribution_score(query, text)
            
            # 3. Title/heading presence (simulated)
            factors['structure'] = self._calculate_structure_score(text)
            
            # Combined score
            combined_score = (
                factors['position'] * 0.4 +
                factors['distribution'] * 0.4 +
                factors['structure'] * 0.2
            )
            
            positional_scores.append(combined_score)
            factor_breakdowns.append(factors)
        
        # Normalize
        if len(positional_scores) > 0:
            min_score, max_score = min(positional_scores), max(positional_scores)
            if max_score > min_score:
                positional_scores = [(s - min_score) / (max_score - min_score) for s in positional_scores]
        
        # Use original scores if provided
        if original_scores is None:
            original_scores = positional_scores.copy()
        
        # Create results
        return self._create_results(query, documents, positional_scores, original_scores, factor_breakdowns)
    
    def _calculate_position_score(self, query: str, document: str) -> float:
        """Score based on position of query terms"""
        query_terms = [term.lower() for term in query.split() if term.lower() not in self.stop_words]
        
        if not query_terms:
            return 0.5
        
        sentences = sent_tokenize(document)
        if not sentences:
            return 0.5
        
        # Check position in first sentence
        first_sentence = sentences[0].lower()
        term_in_first = any(term in first_sentence for term in query_terms)
        
        # Check position in title (first 100 chars)
        title_region = document[:100].lower()
        term_in_title = any(term in title_region for term in query_terms)
        
        # Calculate score
        score = 0.0
        if term_in_title:
            score += 0.6
        if term_in_first:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_distribution_score(self, query: str, document: str) -> float:
        """Score based on distribution of query terms"""
        query_terms = [term.lower() for term in query.split() if term.lower() not in self.stop_words]
        
        if not query_terms:
            return 0.5
        
        document_lower = document.lower()
        sentences = sent_tokenize(document)
        
        if not sentences:
            return 0.5
        
        # Count sentences containing query terms
        relevant_sentences = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in query_terms):
                relevant_sentences += 1
        
        # Calculate distribution score
        distribution_ratio = relevant_sentences / len(sentences)
        
        # Also consider term frequency
        total_terms = len(document_lower.split())
        if total_terms == 0:
            return 0.5
        
        term_freq = sum(document_lower.count(term) for term in query_terms)
        freq_ratio = term_freq / total_terms
        
        # Combined score
        score = (distribution_ratio * 0.6 + freq_ratio * 0.4)
        return min(score, 1.0)
    
    def _calculate_structure_score(self, document: str) -> float:
        """Score based on document structure"""
        # Simple heuristic: documents with headings, lists, etc. are better structured
        lines = document.split('\n')
        
        structure_indicators = 0
        for line in lines[:20]:  # Check first 20 lines
            line_stripped = line.strip()
            
            # Check for headings (short lines, ends with colon, all caps, etc.)
            if len(line_stripped) < 100 and len(line_stripped) > 0:
                if (line_stripped.endswith(':') or 
                    line_stripped.isupper() or
                    line_stripped.startswith('#') or
                    line_stripped.startswith('•') or
                    line_stripped.startswith('-')):
                    structure_indicators += 1
        
        # Normalize score
        score = min(structure_indicators / 5, 1.0)  # Max 5 indicators
        return score

# ========== Reranker Factory ==========

class RerankerFactory:
    """Factory for creating re-rankers"""
    
    @staticmethod
    def create_reranker(
        reranker_type: Union[str, RerankerType],
        config: RerankerConfig = None
    ) -> BaseReranker:
        """Create a re-ranker instance"""
        if isinstance(reranker_type, str):
            try:
                reranker_type = RerankerType(reranker_type.lower())
            except ValueError:
                print(f"⚠️  Unknown reranker type: {reranker_type}, using default")
                reranker_type = RerankerType.CROSS_ENCODER
        
        if config is None:
            config = RerankerConfig(reranker_type=reranker_type)
        
        if reranker_type == RerankerType.CROSS_ENCODER:
            return CrossEncoderReranker(config)
        elif reranker_type == RerankerType.TFIDF_RERANK:
            return TFIDFReranker(config)
        elif reranker_type == RerankerType.SEMANTIC_RERANK:
            return SemanticReranker(config)
        elif reranker_type == RerankerType.HYBRID_RERANK:
            return HybridReranker(config)
        elif reranker_type == RerankerType.POSITIONAL_RERANK:
            return PositionalReranker(config)
        elif reranker_type == RerankerType.ENSEMBLE:
            return HybridReranker(config)  # Ensemble uses HybridReranker
        else:
            print(f"⚠️  Unsupported reranker type: {reranker_type}, using TF-IDF")
            return TFIDFReranker(config)

# ========== Advanced Reranking System ==========

class AdvancedRerankingSystem:
    """Complete re-ranking system with evaluation capabilities"""
    
    def __init__(self):
        self.rerankers = {}
        self.evaluation_results = {}
    
    def add_reranker(self, name: str, reranker: BaseReranker):
        """Add a re-ranker to the system"""
        self.rerankers[name] = reranker
        print(f"✅ Added re-ranker: {name}")
    
    def evaluate_rerankers(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        ground_truth: Optional[List[int]] = None,
        original_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Evaluate all re-rankers"""
        evaluation = {
            "query": query,
            "total_documents": len(documents),
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        for name, reranker in self.rerankers.items():
            print(f"📊 Evaluating: {name}")
            
            # Perform re-ranking
            results = reranker.rerank(query, documents, original_scores)
            
            # Calculate metrics
            metrics = self._calculate_metrics(results, ground_truth)
            
            # Add stats
            metrics.update(reranker.get_stats())
            
            evaluation["results"][name] = {
                "metrics": metrics,
                "top_results": [r.to_dict() for r in results[:3]]
            }
        
        # Store evaluation
        eval_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.evaluation_results[eval_id] = evaluation
        
        return evaluation
    
    def _calculate_metrics(
        self,
        results: List[RerankedResult],
        ground_truth: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # 1. Average improvement
        if results:
            improvements = [r.improvement for r in results]
            metrics["avg_improvement"] = np.mean(improvements)
            metrics["max_improvement"] = np.max(improvements)
            metrics["improved_count"] = sum(1 for r in results if r.improvement > 0)
            metrics["worsened_count"] = sum(1 for r in results if r.improvement < 0)
        
        # 2. Rank change statistics
        rank_changes = [r.rank_change for r in results]
        if rank_changes:
            metrics["avg_rank_change"] = np.mean(rank_changes)
            metrics["max_rank_improvement"] = np.max(rank_changes)
            metrics["max_rank_worsening"] = abs(min(rank_changes, default=0))
        
        # 3. Score statistics
        if results:
            original_scores = [r.original_score for r in results]
            reranked_scores = [r.reranked_score for r in results]
            
            metrics["avg_original_score"] = np.mean(original_scores)
            metrics["avg_reranked_score"] = np.mean(reranked_scores)
            metrics["score_variance"] = np.var(reranked_scores)
        
        # 4. Ground truth metrics (if available)
        if ground_truth and results:
            # Calculate NDCG@k
            ndcg_scores = []
            for k in [3, 5, 10]:
                ndcg = self._calculate_ndcg(results, ground_truth, k)
                metrics[f"ndcg@{k}"] = ndcg
            
            # Calculate precision@k
            for k in [3, 5, 10]:
                precision = self._calculate_precision(results, ground_truth, k)
                metrics[f"precision@{k}"] = precision
        
        return metrics
    
    def _calculate_ndcg(
        self,
        results: List[RerankedResult],
        ground_truth: List[int],
        k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        # Simplified implementation
        # In production, use proper relevance scores
        dcg = 0.0
        idcg = 0.0
        
        for i in range(min(k, len(results))):
            # Relevance based on ground truth (simplified)
            relevance = 1.0 if i in ground_truth else 0.0
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0
        
        # Ideal DCG (ground truth sorted by relevance)
        ideal_relevance = [1.0] * min(k, len(ground_truth))
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_precision(
        self,
        results: List[RerankedResult],
        ground_truth: List[int],
        k: int
    ) -> float:
        """Calculate precision@k"""
        relevant_count = 0
        for i in range(min(k, len(results))):
            if i in ground_truth:
                relevant_count += 1
        
        return relevant_count / k
    
    def generate_report(self, evaluation: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        report = f"""
Re-ranking Evaluation Report
============================

Query: {evaluation['query']}
Total Documents: {evaluation['total_documents']}
Timestamp: {evaluation['timestamp']}

Results:
"""
        
        for name, result in evaluation['results'].items():
            metrics = result['metrics']
            report += f"\n{name.upper()}:\n"
            report += "-" * 40 + "\n"
            
            report += f"Average Improvement: {metrics.get('avg_improvement', 0):.4f}\n"
            report += f"Improved Documents: {metrics.get('improved_count', 0)}\n"
            report += f"Worsened Documents: {metrics.get('worsened_count', 0)}\n"
            report += f"Average Rank Change: {metrics.get('avg_rank_change', 0):.2f}\n"
            
            if 'ndcg@5' in metrics:
                report += f"NDCG@5: {metrics['ndcg@5']:.4f}\n"
                report += f"Precision@5: {metrics['precision@5']:.4f}\n"
            
            report += f"Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.2%}\n"
        
        # Determine best reranker
        if evaluation['results']:
            best_reranker = max(
                evaluation['results'].items(),
                key=lambda x: x[1]['metrics'].get('avg_improvement', 0)
            )
            report += f"\n🏆 Best Reranker: {best_reranker[0]} "
            report += f"(Improvement: {best_reranker[1]['metrics'].get('avg_improvement', 0):.4f})\n"
        
        return report

# ========== Example Usage ==========

def create_sample_documents():
    """Create sample documents for re-ranking"""
    samples = [
        {
            "id": "doc_1",
            "content": "Artificial Intelligence (AI) is transforming industries worldwide. Machine learning algorithms enable computers to learn from data and make predictions.",
            "metadata": {"source": "tech_report.pdf", "date": "2024-01-15", "category": "technology"}
        },
        {
            "id": "doc_2", 
            "content": "Python programming language is widely used for AI development. Its simplicity and extensive libraries make it ideal for machine learning projects.",
            "metadata": {"source": "python_guide.txt", "date": "2023-11-20", "category": "programming"}
        },
        {
            "id": "doc_3",
            "content": "Deep learning neural networks require large amounts of data and computational power. They excel at tasks like image recognition and natural language processing.",
            "metadata": {"source": "ai_research.pdf", "date": "2024-02-10", "category": "research"}
        },
        {
            "id": "doc_4",
            "content": "Vector databases store embeddings for efficient similarity search. They are essential for building recommendation systems and semantic search engines.",
            "metadata": {"source": "database_guide.md", "date": "2024-01-05", "category": "database"}
        },
        {
            "id": "doc_5",
            "content": "Retrieval-Augmented Generation (RAG) combines retrieval and generation for accurate AI responses. It reduces hallucinations and improves factuality.",
            "metadata": {"source": "rag_paper.pdf", "date": "2024-02-28", "category": "ai"}
        }
    ]
    
    return samples

def demo_reranking():
    """Demonstrate re-ranking system"""
    print("=" * 60)
    print("🧪 DEMO: Advanced Re-ranking System")
    print("=" * 60)
    
    # Create sample documents
    documents = create_sample_documents()
    
    # Create re-rankers
    reranker_types = [
        ("cross_encoder", RerankerType.CROSS_ENCODER),
        ("tfidf", RerankerType.TFIDF_RERANK),
        ("semantic", RerankerType.SEMANTIC_RERANK),
        ("hybrid", RerankerType.HYBRID_RERANK),
        ("positional", RerankerType.POSITIONAL_RERANK),
        ("ensemble", RerankerType.ENSEMBLE)
    ]
    
    # Create advanced system
    system = AdvancedRerankingSystem()
    
    for name, r_type in reranker_types:
        config = RerankerConfig(
            reranker_type=r_type,
            top_k=3,
            use_cache=True
        )
        reranker = RerankerFactory.create_reranker(r_type, config)
        system.add_reranker(name, reranker)
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Explain vector databases"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 40)
        
        # Simulate original scores (e.g., from first-stage retrieval)
        original_scores = [0.8, 0.7, 0.6, 0.5, 0.4]  # Decreasing relevance
        
        # Ground truth (indices of actually relevant documents)
        ground_truth = [0, 2, 4]  # Documents 1, 3, 5 are relevant
        
        # Evaluate all re-rankers
        evaluation = system.evaluate_rerankers(
            query, documents, ground_truth, original_scores
        )
        
        # Print results for this query
        for name, result in evaluation['results'].items():
            metrics = result['metrics']
            print(f"\n{name.upper()}:")
            print(f"  Avg Improvement: {metrics.get('avg_improvement', 0):.4f}")
            print(f"  Improved: {metrics.get('improved_count', 0)} documents")
            
            # Show top result
            if result['top_results']:
                top = result['top_results'][0]
                print(f"  Top Result: {top['content_preview'][:50]}...")
    
    # Generate comprehensive report for last query
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE EVALUATION REPORT")
    print("=" * 60)
    
    report = system.generate_report(evaluation)
    print(report)
    
    # Save report to file
    with open("reranking_report.txt", "w") as f:
        f.write(report)
    
    print("✅ Report saved to: reranking_report.txt")
    print("=" * 60)

def interactive_reranking():
    """Interactive re-ranking demo"""
    print("🤖 Interactive Re-ranking Demo")
    print("=" * 50)
    
    # Create sample documents
    documents = create_sample_documents()
    
    print(f"\n📚 Loaded {len(documents)} sample documents:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc['metadata']['source']}")
        print(f"     {doc['content'][:80]}...")
    
    print("\nAvailable re-rankers:")
    print("  1. cross_encoder - Cross-Encoder (most accurate)")
    print("  2. tfidf - TF-IDF based (fast)")
    print("  3. semantic - Semantic embeddings")
    print("  4. hybrid - Hybrid approach")
    print("  5. positional - Position-aware")
    print("  6. ensemble - Ensemble of all")
    print("  7. compare - Compare all re-rankers")
    
    print("\nCommands:")
    print("  Type 'quit' to exit")
    print("  Type 'docs' to show documents")
    print("  Type 'stats' to show statistics")
    print("=" * 50)
    
    while True:
        try:
            user_input = input("\n🔍 Enter query: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            elif user_input.lower() == 'docs':
                print(f"\n📚 Documents:")
                for i, doc in enumerate(documents, 1):
                    print(f"\n  {i}. {doc['metadata']['source']}")
                    print(f"     Category: {doc['metadata']['category']}")
                    print(f"     Date: {doc['metadata'].get('date', 'N/A')}")
                    print(f"     Content: {doc['content'][:100]}...")
                continue
            
            elif user_input.lower() == 'stats':
                # Create a reranker and show stats
                reranker = RerankerFactory.create_reranker("cross_encoder")
                results = reranker.rerank("test", documents)
                stats = reranker.get_stats()
                
                print(f"\n📊 Statistics:")
                print(f"  Cache Hits: {stats['cache_hits']}")
                print(f"  Cache Misses: {stats['cache_misses']}")
                print(f"  Hit Rate: {stats['cache_hit_rate']:.2%}")
                print(f"  Cache Size: {stats['cache_size']}")
                continue
            
            if not user_input:
                continue
            
            # Parse reranker type if specified
            reranker_type = "cross_encoder"
            query = user_input
            
            if user_input.startswith('/'):
                parts = user_input[1:].split(' ', 1)
                if len(parts) == 2:
                    type_str, query = parts
                    reranker_type = type_str.lower()
            
            if reranker_type == 'compare':
                # Compare all re-rankers
                print(f"\n🔍 Comparing all re-rankers for: {query}")
                print("-" * 50)
                
                types_to_compare = [
                    "cross_encoder",
                    "tfidf", 
                    "semantic",
                    "hybrid",
                    "positional",
                    "ensemble"
                ]
                
                for r_type in types_to_compare:
                    reranker = RerankerFactory.create_reranker(r_type)
                    results = reranker.rerank(query, documents)
                    
                    if results:
                        top_result = results[0]
                        print(f"\n{r_type.upper()}:")
                        print(f"  Top Score: {top_result.reranked_score:.4f}")
                        print(f"  Improvement: {top_result.improvement:+.4f}")
                        print(f"  Document: {top_result.metadata.get('source', 'Unknown')}")
                        print(f"  Preview: {top_result.content[:80]}...")
                
                continue
            
            # Use specified reranker
            print(f"\n🔧 Using re-ranker: {reranker_type}")
            print(f"🔍 Query: {query}")
            
            reranker = RerankerFactory.create_reranker(reranker_type)
            results = reranker.rerank(query, documents)
            
            if not results:
                print("   No results found.")
                continue
            
            print(f"\n📄 Re-ranked Results:")
            for i, result in enumerate(results[:3], 1):
                print(f"\n  {i}. {result.metadata.get('source', 'Unknown')}")
                print(f"     Score: {result.reranked_score:.4f} (Original: {result.original_score:.4f})")
                print(f"     Improvement: {result.improvement:+.4f}")
                print(f"     Rank Change: {result.rank_change:+d}")
                
                if result.factors:
                    print(f"     Factors: {', '.join(f'{k}: {v:.3f}' for k, v in result.factors.items())}")
                
                print(f"     Content: {result.content[:100]}...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# ========== Main Execution ==========

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Re-ranking System")
    parser.add_argument("--mode", choices=["demo", "interactive", "benchmark"], 
                       default="demo", help="Mode to run")
    parser.add_argument("--query", type=str, help="Query to test")
    parser.add_argument("--reranker", type=str, default="cross_encoder", 
                       help="Reranker type: cross_encoder, tfidf, semantic, hybrid, positional, ensemble")
    
    args = parser.parse_args()
    
    # Run based on mode
    if args.mode == "demo":
        demo_reranking()
    
    elif args.mode == "interactive":
        interactive_reranking()
    
    elif args.mode == "benchmark":
        # Create comprehensive benchmark
        from datetime import datetime
        import time
        
        print("📊 Running Re-ranking Benchmark")
        print("=" * 60)
        
        # Create documents
        documents = create_sample_documents()
        
        # Test all reranker types
        reranker_types = [
            "cross_encoder",
            "tfidf",
            "semantic", 
            "hybrid",
            "positional",
            "ensemble"
        ]
        
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "total_documents": len(documents),
            "rerankers": {}
        }
        
        test_queries = [
            "artificial intelligence machine learning",
            "python programming language",
            "vector databases similarity search"
        ]
        
        for r_type in reranker_types:
            print(f"\n🔧 Testing: {r_type}")
            
            # Create reranker
            reranker = RerankerFactory.create_reranker(r_type)
            
            # Measure performance
            query_times = []
            improvements = []
            
            for query in test_queries:
                start_time = time.time()
                results = reranker.rerank(query, documents)
                query_time = time.time() - start_time
                
                query_times.append(query_time)
                
                if results:
                    avg_improvement = np.mean([r.improvement for r in results])
                    improvements.append(avg_improvement)
            
            # Store results
            benchmark_results["rerankers"][r_type] = {
                "avg_query_time_ms": np.mean(query_times) * 1000,
                "avg_improvement": np.mean(improvements) if improvements else 0,
                "cache_stats": reranker.get_stats()
            }
        
        # Print results
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        
        for r_type, results in benchmark_results["rerankers"].items():
            print(f"\n{r_type.upper()}:")
            print(f"  Avg Query Time: {results['avg_query_time_ms']:.2f}ms")
            print(f"  Avg Improvement: {results['avg_improvement']:.4f}")
            print(f"  Cache Hit Rate: {results['cache_stats']['cache_hit_rate']:.2%}")
        
        # Save results
        with open("reranking_benchmark.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\n✅ Benchmark saved to: reranking_benchmark.json")
    
    # Handle single query
    if args.query:
        documents = create_sample_documents()
        
        reranker = RerankerFactory.create_reranker(args.reranker)
        results = reranker.rerank(args.query, documents)
        
        print(f"\n🔍 Query: {args.query}")
        print(f"🔧 Reranker: {args.reranker}")
        print(f"📄 Results Found: {len(results)}")
        print("-" * 60)
        
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. {result.metadata.get('source', 'Unknown')}")
            print(f"   Score: {result.reranked_score:.4f} (Original: {result.original_score:.4f})")
            print(f"   Improvement: {result.improvement:+.4f}")
            print(f"   Content: {result.content[:150]}...")