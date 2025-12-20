"""
✂️ Text Splitting & Chunking Strategies
Day 26: Advanced text segmentation for RAG systems
"""

import re
import math
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("⚠️ NLTK not installed. Install with: pip install nltk")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    print("⚠️ spaCy not installed. Install with: pip install spacy")

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    print("⚠️ tiktoken not installed. Install with: pip install tiktoken")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChunkingMethod(Enum):
    """Available text chunking methods"""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"

@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    chunk_number: int = 0
    token_count: int = 0
    char_count: int = 0
    
    def __post_init__(self):
        """Initialize chunk with computed values"""
        self.char_count = len(self.text)
        self.token_count = len(self.text.split())  # Approximate
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:16]
        
        # Ensure metadata has basic info
        if 'chunk_number' not in self.metadata:
            self.metadata['chunk_number'] = self.chunk_number
        if 'char_count' not in self.metadata:
            self.metadata['char_count'] = self.char_count
        if 'token_count' not in self.metadata:
            self.metadata['token_count'] = self.token_count

class TextSplitter:
    """Base class for text splitters"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ensure_overlap_valid()
    
    def ensure_overlap_valid(self):
        """Ensure chunk overlap is valid"""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap ({self.chunk_overlap}) must be smaller than chunk size ({self.chunk_size})")
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text into chunks (to be implemented by subclasses)"""
        raise NotImplementedError
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (approximate)"""
        return len(text.split())
    
    def get_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        total_chars = sum(chunk.char_count for chunk in chunks)
        total_tokens = sum(chunk.token_count for chunk in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'total_tokens': total_tokens,
            'avg_chars_per_chunk': total_chars / len(chunks),
            'avg_tokens_per_chunk': total_tokens / len(chunks),
            'min_chars': min(chunk.char_count for chunk in chunks),
            'max_chars': max(chunk.char_count for chunk in chunks),
            'chunk_size_target': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }

class FixedSizeSplitter(TextSplitter):
    """Split text by fixed character count"""
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text into fixed-size chunks"""
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_number = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the first chunk, add overlap
            if start > 0:
                start = start - self.chunk_overlap
            
            # Ensure we don't go beyond text length
            end = min(end, len(text))
            
            # Get chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'start_position': start,
                        'end_position': end,
                        'chunking_method': ChunkingMethod.FIXED_SIZE.value
                    },
                    chunk_number=chunk_number,
                    token_count=self.count_tokens(chunk_text)
                )
                chunks.append(chunk)
                chunk_number += 1
            
            # Move start position
            start = end
        
        return chunks

class SentenceSplitter(TextSplitter):
    """Split text by sentences"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 language: str = 'english'):
        super().__init__(chunk_size, chunk_overlap)
        self.language = language
        
        # Download NLTK data if needed
        if HAS_NLTK:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text by sentences"""
        if not text:
            return []
        
        if not HAS_NLTK:
            logger.warning("NLTK not available, falling back to simple sentence splitting")
            sentences = re.split(r'(?<=[.!?])\s+', text)
        else:
            sentences = sent_tokenize(text, language=self.language)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_number = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed chunk size, start new chunk
            if current_size + sentence_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunk = TextChunk(
                        text=chunk_text,
                        metadata={
                            'sentence_count': len(current_chunk),
                            'chunking_method': ChunkingMethod.SENTENCE.value,
                            'language': self.language
                        },
                        chunk_number=chunk_number,
                        token_count=self.count_tokens(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_number += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_size = 0
                    for s in reversed(current_chunk):
                        if overlap_size + len(s) <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_size += len(s)
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_size = overlap_size
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'sentence_count': len(current_chunk),
                        'chunking_method': ChunkingMethod.SENTENCE.value,
                        'language': self.language
                    },
                    chunk_number=chunk_number,
                    token_count=self.count_tokens(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks

class ParagraphSplitter(TextSplitter):
    """Split text by paragraphs"""
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text by paragraphs"""
        if not text:
            return []
        
        # Split by multiple newlines
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_number = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed chunk size, start new chunk
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = '\n\n'.join(current_chunk).strip()
                if chunk_text:
                    chunk = TextChunk(
                        text=chunk_text,
                        metadata={
                            'paragraph_count': len(current_chunk),
                            'chunking_method': ChunkingMethod.PARAGRAPH.value
                        },
                        chunk_number=chunk_number,
                        token_count=self.count_tokens(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_number += 1
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Keep last paragraph for overlap
                    if current_chunk:
                        current_chunk = [current_chunk[-1]]
                        current_size = len(current_chunk[0])
                    else:
                        current_chunk = []
                        current_size = 0
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size
        
        # Add the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk).strip()
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'paragraph_count': len(current_chunk),
                        'chunking_method': ChunkingMethod.PARAGRAPH.value
                    },
                    chunk_number=chunk_number,
                    token_count=self.count_tokens(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks

class SemanticSplitter(TextSplitter):
    """Split text semantically using sentence embeddings"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 similarity_threshold: float = 0.5):
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        
        # Try to load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_embeddings = True
        except ImportError:
            self.has_embeddings = False
            logger.warning("Sentence Transformers not available, falling back to sentence splitting")
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text semantically"""
        if not text:
            return []
        
        if not self.has_embeddings:
            # Fall back to sentence splitting
            fallback_splitter = SentenceSplitter(self.chunk_size, self.chunk_overlap)
            return fallback_splitter.split_text(text)
        
        # Split into sentences first
        if HAS_NLTK:
            sentences = sent_tokenize(text)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return []
        
        # Create embeddings for sentences
        sentence_embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_number = 0
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            sentence_size = len(sentence)
            
            # Check if we should start a new chunk
            should_start_new = False
            
            if current_size + sentence_size > self.chunk_size:
                should_start_new = True
            elif i > 0 and current_chunk:
                # Check semantic similarity with previous sentence
                prev_embedding = sentence_embeddings[i-1]
                similarity = self.cosine_similarity(embedding, prev_embedding)
                if similarity < self.similarity_threshold:
                    should_start_new = True
            
            if should_start_new and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunk = TextChunk(
                        text=chunk_text,
                        metadata={
                            'sentence_count': len(current_chunk),
                            'chunking_method': ChunkingMethod.SEMANTIC.value,
                            'similarity_threshold': self.similarity_threshold
                        },
                        chunk_number=chunk_number,
                        token_count=self.count_tokens(chunk_text)
                    )
                    chunks.append(chunk)
                    chunk_number += 1
                
                # Start new chunk with overlap
                current_chunk = [current_chunk[-1]] if current_chunk and self.chunk_overlap > 0 else []
                current_size = len(current_chunk[0]) if current_chunk else 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'sentence_count': len(current_chunk),
                        'chunking_method': ChunkingMethod.SEMANTIC.value,
                        'similarity_threshold': self.similarity_threshold
                    },
                    chunk_number=chunk_number,
                    token_count=self.count_tokens(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

class RecursiveSplitter(TextSplitter):
    """Recursively split text using multiple separators"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        
        # Define separators in order of priority
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            "? ",
            "! ",
            " ",     # Words
            "",      # Characters
        ]
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Recursively split text"""
        return self._split_recursive(text, 0)
    
    def _split_recursive(self, text: str, separator_index: int) -> List[TextChunk]:
        """Recursive splitting helper"""
        if not text:
            return []
        
        # Get current separator
        separator = self.separators[separator_index]
        
        if separator:
            # Split by current separator
            splits = text.split(separator)
        else:
            # Last separator: split by character
            splits = list(text)
        
        # Combine splits into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split_text = split + (separator if separator else "")
            split_size = len(split_text)
            
            # If a single split is too large, recurse with next separator
            if split_size > self.chunk_size:
                if separator_index < len(self.separators) - 1:
                    sub_chunks = self._split_recursive(split, separator_index + 1)
                    chunks.extend(sub_chunks)
                else:
                    # Split by character if nothing else works
                    char_chunks = self._split_by_fixed_size(split, self.chunk_size)
                    chunks.extend(char_chunks)
                continue
            
            # If adding this split would exceed chunk size, start new chunk
            if current_size + split_size > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ''.join(current_chunk).strip()
                if chunk_text:
                    chunk = TextChunk(
                        text=chunk_text,
                        metadata={
                            'chunking_method': ChunkingMethod.RECURSIVE.value,
                            'separator_used': separator if separator else "character"
                        },
                        token_count=self.count_tokens(chunk_text)
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0 and current_chunk:
                    # Keep last part for overlap
                    overlap_text = ""
                    for s in reversed(current_chunk):
                        if len(overlap_text) + len(s) <= self.chunk_overlap:
                            overlap_text = s + overlap_text
                        else:
                            break
                    current_chunk = [overlap_text] if overlap_text else []
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add split to current chunk
            current_chunk.append(split_text)
            current_size += split_size
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ''.join(current_chunk).strip()
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'chunking_method': ChunkingMethod.RECURSIVE.value,
                        'separator_used': separator if separator else "character"
                    },
                    token_count=self.count_tokens(chunk_text)
                )
                chunks.append(chunk)
        
        # Add chunk numbers
        for i, chunk in enumerate(chunks):
            chunk.chunk_number = i
        
        return chunks
    
    def _split_by_fixed_size(self, text: str, chunk_size: int) -> List[TextChunk]:
        """Split text by fixed character size"""
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunk_text = text[i:i + chunk_size]
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'chunking_method': 'fixed_size_fallback',
                        'start_position': i,
                        'end_position': i + len(chunk_text)
                    },
                    token_count=self.count_tokens(chunk_text)
                )
                chunks.append(chunk)
        return chunks

class SlidingWindowSplitter(TextSplitter):
    """Split text using a sliding window approach"""
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text with sliding window"""
        if not text:
            return []
        
        chunks = []
        step_size = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(text), step_size):
            start = max(0, i - self.chunk_overlap if i > 0 else 0)
            end = min(len(text), start + self.chunk_size)
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'chunking_method': ChunkingMethod.SLIDING_WINDOW.value,
                        'window_start': start,
                        'window_end': end,
                        'step_size': step_size
                    },
                    chunk_number=len(chunks),
                    token_count=self.count_tokens(chunk_text)
                )
                chunks.append(chunk)
        
        return chunks

class TokenAwareSplitter(TextSplitter):
    """Split text based on token count (e.g., for LLM context windows)"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200,
                 encoding_name: str = "cl100k_base"):  # GPT-4 encoding
        super().__init__(chunk_size, chunk_overlap)
        self.encoding_name = encoding_name
        
        if HAS_TIKTOKEN:
            self.encoder = tiktoken.get_encoding(encoding_name)
        else:
            self.encoder = None
            logger.warning("tiktoken not available, falling back to word count")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens accurately"""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Fallback: approximate with word count
            return len(text.split())
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text based on token count"""
        if not text:
            return []
        
        if not self.encoder:
            # Fall back to fixed size splitting
            fallback_splitter = FixedSizeSplitter(self.chunk_size, self.chunk_overlap)
            return fallback_splitter.split_text(text)
        
        # Encode text to tokens
        tokens = self.encoder.encode(text)
        
        chunks = []
        start = 0
        chunk_number = 0
        
        while start < len(tokens):
            # Calculate end position
            end = start + self.chunk_size
            
            # If this is not the first chunk, add overlap
            if start > 0:
                start = start - self.chunk_overlap
            
            # Ensure we don't go beyond tokens
            end = min(end, len(tokens))
            
            # Decode chunk tokens back to text
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens).strip()
            
            if chunk_text:
                chunk = TextChunk(
                    text=chunk_text,
                    metadata={
                        'chunking_method': 'token_aware',
                        'token_count': len(chunk_tokens),
                        'encoding': self.encoding_name,
                        'token_start': start,
                        'token_end': end
                    },
                    chunk_number=chunk_number,
                    token_count=len(chunk_tokens),
                    char_count=len(chunk_text)
                )
                chunks.append(chunk)
                chunk_number += 1
            
            # Move start position
            start = end
        
        return chunks

class TextSplitterFactory:
    """Factory for creating text splitters"""
    
    @staticmethod
    def create_splitter(method: ChunkingMethod, **kwargs) -> TextSplitter:
        """Create text splitter based on method"""
        method_map = {
            ChunkingMethod.FIXED_SIZE: FixedSizeSplitter,
            ChunkingMethod.SENTENCE: SentenceSplitter,
            ChunkingMethod.PARAGRAPH: ParagraphSplitter,
            ChunkingMethod.SEMANTIC: SemanticSplitter,
            ChunkingMethod.RECURSIVE: RecursiveSplitter,
            ChunkingMethod.SLIDING_WINDOW: SlidingWindowSplitter,
        }
        
        if method not in method_map:
            raise ValueError(f"Unknown chunking method: {method}")
        
        return method_map[method](**kwargs)
    
    @staticmethod
    def create_token_aware_splitter(**kwargs) -> TokenAwareSplitter:
        """Create token-aware splitter"""
        return TokenAwareSplitter(**kwargs)

def compare_chunking_methods():
    """Compare different chunking methods"""
    print("=" * 70)
    print("✂️ CHUNKING METHODS COMPARISON (Day 26)")
    print("=" * 70)
    
    # Sample text for testing
    sample_text = """
Artificial Intelligence (AI) is revolutionizing how we interact with technology. 
Machine learning, a subset of AI, enables computers to learn from data without explicit programming.

Deep learning models, particularly neural networks, have achieved remarkable success in areas like computer vision and natural language processing. 
These models require substantial computational resources and large datasets for training.

Natural Language Processing (NLP) allows machines to understand, interpret, and generate human language. 
Applications include chatbots, translation services, and sentiment analysis.

Computer vision enables machines to interpret and understand visual information from the world. 
This technology powers facial recognition, autonomous vehicles, and medical image analysis.

The field of AI ethics has emerged to address concerns about bias, fairness, and transparency in AI systems. 
Researchers are developing techniques to make AI more interpretable and accountable.

Reinforcement learning involves training agents to make decisions through trial and error. 
This approach has been successful in game playing, robotics, and recommendation systems.

Transfer learning allows models trained on one task to be adapted for related tasks with limited data. 
This technique has accelerated AI development across various domains.

Explainable AI (XAI) focuses on making AI decisions understandable to humans. 
This is crucial for building trust in AI systems, especially in high-stakes applications.
"""
    
    print(f"\n📄 Sample text: {len(sample_text):,} characters, ~{len(sample_text.split()):,} words")
    
    # Test different methods
    methods_to_test = [
        (ChunkingMethod.FIXED_SIZE, FixedSizeSplitter, {"chunk_size": 300, "chunk_overlap": 50}),
        (ChunkingMethod.SENTENCE, SentenceSplitter, {"chunk_size": 300, "chunk_overlap": 50}),
        (ChunkingMethod.PARAGRAPH, ParagraphSplitter, {"chunk_size": 300, "chunk_overlap": 50}),
        (ChunkingMethod.RECURSIVE, RecursiveSplitter, {"chunk_size": 300, "chunk_overlap": 50}),
        (ChunkingMethod.SLIDING_WINDOW, SlidingWindowSplitter, {"chunk_size": 300, "chunk_overlap": 50}),
    ]
    
    # Add semantic splitter if available
    try:
        from sentence_transformers import SentenceTransformer
        methods_to_test.append(
            (ChunkingMethod.SEMANTIC, SemanticSplitter, {"chunk_size": 300, "chunk_overlap": 50})
        )
    except ImportError:
        print("\n⚠️ Semantic splitter skipped (Sentence Transformers not installed)")
    
    results = []
    
    for method, splitter_class, kwargs in methods_to_test:
        print(f"\n🔧 Testing {method.value.replace('_', ' ').title()}...")
        
        try:
            splitter = splitter_class(**kwargs)
            chunks = splitter.split_text(sample_text)
            stats = splitter.get_stats(chunks)
            
            results.append({
                'method': method.value,
                'chunks': len(chunks),
                'avg_chars': stats['avg_chars_per_chunk'],
                'avg_tokens': stats['avg_tokens_per_chunk'],
                'min_chars': stats['min_chars'],
                'max_chars': stats['max_chars']
            })
            
            print(f"   ✓ Created {len(chunks)} chunks")
            print(f"     Avg size: {stats['avg_chars_per_chunk']:.0f} chars, {stats['avg_tokens_per_chunk']:.0f} tokens")
            
            # Show sample chunks
            if chunks and len(chunks) >= 2:
                print(f"     Sample 1: {chunks[0].text[:80]}...")
                print(f"     Sample 2: {chunks[1].text[:80]}...")
        
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results.append({
                'method': method.value,
                'chunks': 0,
                'avg_chars': 0,
                'avg_tokens': 0,
                'min_chars': 0,
                'max_chars': 0
            })
    
    # Display comparison table
    print("\n" + "=" * 70)
    print("📊 CHUNKING METHODS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'Chunks':<10} {'Avg Chars':<12} {'Avg Tokens':<12} {'Min':<10} {'Max':<10}")
    print("-" * 74)
    
    for result in results:
        print(f"{result['method'].replace('_', ' '):<20} "
              f"{result['chunks']:<10} "
              f"{result['avg_chars']:<12.0f} "
              f"{result['avg_tokens']:<12.0f} "
              f"{result['min_chars']:<10} "
              f"{result['max_chars']:<10}")

def demo_text_splitting():
    """Demonstrate text splitting capabilities"""
    print("\n🎯 TEXT SPLITTING DEMONSTRATION")
    print("=" * 60)
    
    # Create a more complex sample text
    sample_text = """
# Artificial Intelligence Overview

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.

## Machine Learning

Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.

### Types of Machine Learning

1. Supervised Learning: The algorithm learns from labeled training data.
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
3. Reinforcement Learning: The algorithm learns through trial and error.

## Deep Learning

Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to analyze various factors of data. It has driven many recent advances in AI.

### Applications

- Computer Vision: Image recognition, object detection
- Natural Language Processing: Language translation, sentiment analysis
- Speech Recognition: Voice assistants, transcription services

## Challenges and Future Directions

While AI has made significant progress, challenges remain including data privacy concerns, algorithmic bias, and the need for transparency in decision-making processes. Future research focuses on making AI systems more robust, fair, and interpretable.
"""
    
    print(f"\n📄 Sample text structure:")
    print(f"   Length: {len(sample_text):,} characters")
    print(f"   Paragraphs: {sample_text.count('\\n\\n')}")
    print(f"   Sentences: ~{sample_text.count('. ') + sample_text.count('? ') + sample_text.count('! ') + 1}")
    
    # Test recursive splitter (handles structured text well)
    print("\n🔧 Testing Recursive Splitter on structured text...")
    
    try:
        splitter = RecursiveSplitter(chunk_size=400, chunk_overlap=100)
        chunks = splitter.split_text(sample_text)
        
        print(f"   ✓ Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n   Chunk {i} ({chunk.char_count} chars):")
            print(f"   {chunk.text[:100]}...")
            print(f"   Metadata: {chunk.metadata.get('separator_used', 'N/A')}")
    
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # Test sliding window for code-like text
    print("\n🔧 Testing Sliding Window Splitter...")
    
    try:
        splitter = SlidingWindowSplitter(chunk_size=300, chunk_overlap=75)
        chunks = splitter.split_text(sample_text)
        
        print(f"   ✓ Created {len(chunks)} chunks with sliding window")
        
        # Show window positions
        for i, chunk in enumerate(chunks[:2], 1):
            print(f"   Chunk {i}: Window [{chunk.metadata['window_start']}:{chunk.metadata['window_end']}]")
    
    except Exception as e:
        print(f"   ✗ Failed: {e}")

def benchmark_splitters():
    """Benchmark different text splitters"""
    print("\n⏱️ TEXT SPLITTER BENCHMARK")
    print("=" * 60)
    
    import time
    
    # Generate longer text for benchmarking
    paragraphs = [
        "Artificial Intelligence is transforming industries. " * 10,
        "Machine learning algorithms analyze vast datasets. " * 10,
        "Natural Language Processing enables human-computer interaction. " * 10,
        "Computer vision systems interpret visual information. " * 10,
        "Deep learning models require substantial computational resources. " * 10,
    ]
    test_text = '\n\n'.join(paragraphs)
    
    print(f"Test text: {len(test_text):,} characters")
    
    # Test configurations
    test_configs = [
        ("Fixed Size", FixedSizeSplitter, {"chunk_size": 500, "chunk_overlap": 100}),
        ("Sentence", SentenceSplitter, {"chunk_size": 500, "chunk_overlap": 100}),
        ("Paragraph", ParagraphSplitter, {"chunk_size": 500, "chunk_overlap": 100}),
        ("Recursive", RecursiveSplitter, {"chunk_size": 500, "chunk_overlap": 100}),
    ]
    
    results = []
    
    for name, splitter_class, kwargs in test_configs:
        print(f"\n⏱️ Benchmarking {name} Splitter...")
        
        try:
            # Warm up
            splitter = splitter_class(**kwargs)
            _ = splitter.split_text(test_text[:1000])
            
            # Actual benchmark
            start_time = time.perf_counter()
            chunks = splitter.split_text(test_text)
            elapsed = time.perf_counter() - start_time
            
            stats = splitter.get_stats(chunks)
            
            results.append({
                'name': name,
                'time': elapsed,
                'chunks': len(chunks),
                'chunks_per_sec': len(chunks) / elapsed if elapsed > 0 else 0,
                'chars_per_sec': len(test_text) / elapsed if elapsed > 0 else 0
            })
            
            print(f"   ✓ Time: {elapsed:.3f}s")
            print(f"     Chunks: {len(chunks)}")
            print(f"     Chunks/sec: {len(chunks)/elapsed:.1f}")
            print(f"     Chars/sec: {len(test_text)/elapsed:,.0f}")
        
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results.append({
                'name': name,
                'time': 0,
                'chunks': 0,
                'chunks_per_sec': 0,
                'chars_per_sec': 0
            })
    
    # Display results
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\n{'Splitter':<15} {'Time (s)':<10} {'Chunks':<10} {'Chunks/s':<12} {'Chars/s':<15}")
    print("-" * 62)
    
    for result in results:
        print(f"{result['name']:<15} "
              f"{result['time']:<10.3f} "
              f"{result['chunks']:<10} "
              f"{result['chunks_per_sec']:<12.1f} "
              f"{result['chars_per_sec']:<15,.0f}")

def interactive_splitter_demo():
    """Interactive text splitter demonstration"""
    print("\n🎮 Interactive Text Splitter")
    print("=" * 60)
    
    print("\n✂️ Available chunking methods:")
    for method in ChunkingMethod:
        print(f"  - {method.value.replace('_', ' ').title()}")
    
    print("\n⚙️ Available commands:")
    print("  /method [name] - Set chunking method")
    print("  /size [number] - Set chunk size")
    print("  /overlap [number] - Set chunk overlap")
    print("  /text [your text] - Load text (use quotes for multi-line)")
    print("  /split - Split current text")
    print("  /stats - Show splitting statistics")
    print("  /preview [n] - Preview first n chunks")
    print("  /clear - Clear text and chunks")
    print("  /quit - Exit")
    
    current_text = ""
    current_chunks = []
    current_method = ChunkingMethod.FIXED_SIZE
    chunk_size = 500
    chunk_overlap = 100
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if command.lower() == '/quit':
                print("Goodbye!")
                break
            
            elif command.lower() == '/clear':
                current_text = ""
                current_chunks = []
                print("✅ Cleared text and chunks")
            
            elif command.lower() == '/stats':
                if not current_chunks:
                    print("❌ No chunks available")
                else:
                    total_chars = sum(chunk.char_count for chunk in current_chunks)
                    total_tokens = sum(chunk.token_count for chunk in current_chunks)
                    
                    print(f"\n📊 Statistics:")
                    print(f"  Method: {current_method.value}")
                    print(f"  Chunk size: {chunk_size}")
                    print(f"  Chunk overlap: {chunk_overlap}")
                    print(f"  Total chunks: {len(current_chunks)}")
                    print(f"  Total characters: {total_chars:,}")
                    print(f"  Total tokens: {total_tokens:,}")
                    print(f"  Avg chars per chunk: {total_chars/len(current_chunks):.0f}")
                    print(f"  Min chars: {min(chunk.char_count for chunk in current_chunks)}")
                    print(f"  Max chars: {max(chunk.char_count for chunk in current_chunks)}")
            
            elif command.lower().startswith('/preview '):
                try:
                    n = int(command.split()[1])
                    if not current_chunks:
                        print("❌ No chunks available")
                    else:
                        print(f"\n📄 Preview (first {min(n, len(current_chunks))} chunks):")
                        for i, chunk in enumerate(current_chunks[:n], 1):
                            print(f"\n{i}. Chunk #{chunk.chunk_number} ({chunk.char_count} chars):")
                            print(f"   {chunk.text[:100]}...")
                except (IndexError, ValueError):
                    print("❌ Usage: /preview [number_of_chunks]")
            
            elif command.lower().startswith('/method '):
                try:
                    method_name = command[8:].strip().upper()
                    method = ChunkingMethod[method_name]
                    current_method = method
                    print(f"✅ Method set to: {method.value.replace('_', ' ').title()}")
                except KeyError:
                    print(f"❌ Unknown method. Available: {', '.join(m.name for m in ChunkingMethod)}")
            
            elif command.lower().startswith('/size '):
                try:
                    size = int(command.split()[1])
                    if size <= 0:
                        print("❌ Chunk size must be positive")
                    else:
                        chunk_size = size
                        print(f"✅ Chunk size set to: {chunk_size}")
                except (IndexError, ValueError):
                    print("❌ Usage: /size [number]")
            
            elif command.lower().startswith('/overlap '):
                try:
                    overlap = int(command.split()[1])
                    if overlap < 0:
                        print("❌ Chunk overlap cannot be negative")
                    elif overlap >= chunk_size:
                        print("❌ Chunk overlap must be smaller than chunk size")
                    else:
                        chunk_overlap = overlap
                        print(f"✅ Chunk overlap set to: {chunk_overlap}")
                except (IndexError, ValueError):
                    print("❌ Usage: /overlap [number]")
            
            elif command.lower().startswith('/text '):
                try:
                    text = command[6:].strip()
                    if text.startswith('"') and text.endswith('"'):
                        text = text[1:-1]
                    
                    current_text = text
                    print(f"✅ Text loaded: {len(current_text):,} characters")
                except Exception as e:
                    print(f"❌ Error loading text: {e}")
            
            elif command.lower() == '/split':
                if not current_text:
                    print("❌ No text to split")
                    continue
                
                print(f"\n✂️ Splitting text with {current_method.value}...")
                print(f"   Chunk size: {chunk_size}, Overlap: {chunk_overlap}")
                
                try:
                    # Create splitter
                    splitter = TextSplitterFactory.create_splitter(
                        current_method,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Split text
                    current_chunks = splitter.split_text(current_text)
                    
                    print(f"✅ Created {len(current_chunks)} chunks")
                    
                    # Show first chunk as preview
                    if current_chunks:
                        print(f"\n📄 First chunk preview:")
                        print(f"   {current_chunks[0].text[:150]}...")
                
                except Exception as e:
                    print(f"❌ Error splitting text: {e}")
            
            else:
                print("❌ Unknown command. Type /help for commands.")
        
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run text splitting demonstrations"""
    print("🚀 Advanced Text Splitting Strategies")
    print("Day 26: Intelligent Text Chunking for RAG Systems")
    
    # Demo 1: Compare chunking methods
    compare_chunking_methods()
    
    # Demo 2: Text splitting demonstration
    demo_text_splitting()
    
    # Demo 3: Benchmarking
    benchmark_splitters()
    
    # Demo 4: Interactive demo
    try:
        run_interactive = input("\n🎮 Run interactive splitter demo? (y/n): ").lower()
        if run_interactive == 'y':
            interactive_splitter_demo()
    except:
        print("Skipping interactive demo...")
    
    print("\n" + "=" * 70)
    print("✅ Day 26 Complete!")
    print("📚 You've built a comprehensive text splitting system!")
    print("   Next: Vector Store Setup (Day 27)")

if __name__ == "__main__":
    main()