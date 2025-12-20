"""
✂️ Text Splitting & Chunking Strategies - COMPLETE FIXED VERSION
Day 26: Advanced text segmentation for RAG systems
"""

import re
import time
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# ==================== DEPENDENCY CHECKS ====================
try:
    import nltk
    HAS_NLTK = True
    
    # Auto-download required NLTK data
    def setup_nltk():
        """Ensure NLTK data is downloaded"""
        required_packages = ['punkt', 'punkt_tab']
        for package in required_packages:
            try:
                nltk.data.find(f'tokenizers/{package}')
            except LookupError:
                print(f"📥 Downloading NLTK {package}...")
                nltk.download(package, quiet=True)
    
    setup_nltk()
    from nltk.tokenize import sent_tokenize
    
except ImportError:
    HAS_NLTK = False
    print("⚠️ NLTK not installed. Install with: pip install nltk")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ==================== CORE CLASSES ====================

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
        self.token_count = len(self.text.split())
        if not self.chunk_id:
            import hashlib
            self.chunk_id = hashlib.md5(self.text.encode()).hexdigest()[:16]
        
        self.metadata.setdefault('chunk_number', self.chunk_number)
        self.metadata.setdefault('char_count', self.char_count)
        self.metadata.setdefault('token_count', self.token_count)

class TextSplitter:
    """Base class for text splitters"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"Chunk overlap ({self.chunk_overlap}) must be smaller than chunk size ({self.chunk_size})")
    
    def split_text(self, text: str) -> List[TextChunk]:
        """Split text into chunks"""
        raise NotImplementedError
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(text.split())
    
    def get_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about chunks"""
        if not chunks:
            return {
                'total_chunks': 0, 'total_characters': 0, 'total_tokens': 0,
                'avg_chars_per_chunk': 0, 'avg_tokens_per_chunk': 0,
                'min_chars': 0, 'max_chars': 0,
                'chunk_size_target': self.chunk_size, 'chunk_overlap': self.chunk_overlap
            }
        
        total_chars = sum(c.char_count for c in chunks)
        total_tokens = sum(c.token_count for c in chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'total_tokens': total_tokens,
            'avg_chars_per_chunk': total_chars / len(chunks),
            'avg_tokens_per_chunk': total_tokens / len(chunks),
            'min_chars': min(c.char_count for c in chunks),
            'max_chars': max(c.char_count for c in chunks),
            'chunk_size_target': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }

class FixedSizeSplitter(TextSplitter):
    """Split text by fixed character count"""
    
    def split_text(self, text: str) -> List[TextChunk]:
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_number = 0
        
        while start < len(text):
            end = start + self.chunk_size
            if start > 0:
                start = start - self.chunk_overlap
            end = min(end, len(text))
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata={'start_position': start, 'end_position': end, 
                             'chunking_method': ChunkingMethod.FIXED_SIZE.value},
                    chunk_number=chunk_number,
                    token_count=self.count_tokens(chunk_text)
                ))
                chunk_number += 1
            start = end
        
        return chunks

class SentenceSplitter(TextSplitter):
    """Split text by sentences"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, language: str = 'english'):
        super().__init__(chunk_size, chunk_overlap)
        self.language = language
    
    def split_text(self, text: str) -> List[TextChunk]:
        if not text:
            return []
        
        if HAS_NLTK:
            try:
                sentences = sent_tokenize(text, language=self.language)
            except:
                sentences = re.split(r'(?<=[.!?])\s+', text)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_number = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        metadata={'sentence_count': len(current_chunk),
                                 'chunking_method': ChunkingMethod.SENTENCE.value,
                                 'language': self.language},
                        chunk_number=chunk_number,
                        token_count=self.count_tokens(chunk_text)
                    ))
                    chunk_number += 1
                
                if self.chunk_overlap > 0:
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
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata={'sentence_count': len(current_chunk),
                             'chunking_method': ChunkingMethod.SENTENCE.value,
                             'language': self.language},
                    chunk_number=chunk_number,
                    token_count=self.count_tokens(chunk_text)
                ))
        
        return chunks

class ParagraphSplitter(TextSplitter):
    """Split text by paragraphs"""
    
    def split_text(self, text: str) -> List[TextChunk]:
        if not text:
            return []
        
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_number = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            if current_size + paragraph_size > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        metadata={'paragraph_count': len(current_chunk),
                                 'chunking_method': ChunkingMethod.PARAGRAPH.value},
                        chunk_number=chunk_number,
                        token_count=self.count_tokens(chunk_text)
                    ))
                    chunk_number += 1
                
                if self.chunk_overlap > 0 and current_chunk:
                    current_chunk = [current_chunk[-1]]
                    current_size = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(paragraph)
            current_size += paragraph_size
        
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata={'paragraph_count': len(current_chunk),
                             'chunking_method': ChunkingMethod.PARAGRAPH.value},
                    chunk_number=chunk_number,
                    token_count=self.count_tokens(chunk_text)
                ))
        
        return chunks

class RecursiveSplitter(TextSplitter):
    """Recursively split text using multiple separators"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__(chunk_size, chunk_overlap)
        self.separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    
    def split_text(self, text: str) -> List[TextChunk]:
        chunks = self._split_recursive(text, 0)
        for i, chunk in enumerate(chunks):
            chunk.chunk_number = i
        return chunks
    
    def _split_recursive(self, text: str, separator_index: int) -> List[TextChunk]:
        if not text:
            return []
        
        separator = self.separators[separator_index]
        splits = text.split(separator) if separator else list(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split_text = split + (separator if separator else "")
            split_size = len(split_text)
            
            if split_size > self.chunk_size:
                if separator_index < len(self.separators) - 1:
                    chunks.extend(self._split_recursive(split, separator_index + 1))
                else:
                    chunks.extend(self._split_by_fixed_size(split))
                continue
            
            if current_size + split_size > self.chunk_size and current_chunk:
                chunk_text = ''.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        metadata={'chunking_method': ChunkingMethod.RECURSIVE.value,
                                 'separator_used': separator or "character"},
                        token_count=self.count_tokens(chunk_text)
                    ))
                
                if self.chunk_overlap > 0 and current_chunk:
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
            
            current_chunk.append(split_text)
            current_size += split_size
        
        if current_chunk:
            chunk_text = ''.join(current_chunk).strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata={'chunking_method': ChunkingMethod.RECURSIVE.value,
                             'separator_used': separator or "character"},
                    token_count=self.count_tokens(chunk_text)
                ))
        
        return chunks
    
    def _split_by_fixed_size(self, text: str) -> List[TextChunk]:
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata={'chunking_method': 'fixed_size_fallback',
                             'start_position': i, 'end_position': i + len(chunk_text)},
                    token_count=self.count_tokens(chunk_text)
                ))
        return chunks

class SlidingWindowSplitter(TextSplitter):
    """Split text using a sliding window approach"""
    
    def split_text(self, text: str) -> List[TextChunk]:
        if not text:
            return []
        
        chunks = []
        step_size = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(text), step_size):
            start = max(0, i - self.chunk_overlap if i > 0 else 0)
            end = min(len(text), start + self.chunk_size)
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata={'chunking_method': ChunkingMethod.SLIDING_WINDOW.value,
                             'window_start': start, 'window_end': end, 'step_size': step_size},
                    chunk_number=len(chunks),
                    token_count=self.count_tokens(chunk_text)
                ))
        
        return chunks

class SemanticSplitter(TextSplitter):
    """Split text semantically using sentence embeddings"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, similarity_threshold: float = 0.5):
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.has_embeddings = False
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_embeddings = True
        except ImportError:
            pass
    
    def split_text(self, text: str) -> List[TextChunk]:
        if not text:
            return []
        
        if not self.has_embeddings:
            return SentenceSplitter(self.chunk_size, self.chunk_overlap).split_text(text)
        
        sentences = sent_tokenize(text) if HAS_NLTK else re.split(r'(?<=[.!?])\s+', text)
        if not sentences:
            return []
        
        sentence_embeddings = self.model.encode(sentences)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_number = 0
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            sentence_size = len(sentence)
            should_start_new = False
            
            if current_size + sentence_size > self.chunk_size:
                should_start_new = True
            elif i > 0 and current_chunk:
                prev_embedding = sentence_embeddings[i-1]
                similarity = self.cosine_similarity(embedding, prev_embedding)
                if similarity < self.similarity_threshold:
                    should_start_new = True
            
            if should_start_new and current_chunk:
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunks.append(TextChunk(
                        text=chunk_text,
                        metadata={'sentence_count': len(current_chunk),
                                 'chunking_method': ChunkingMethod.SEMANTIC.value,
                                 'similarity_threshold': self.similarity_threshold},
                        chunk_number=chunk_number,
                        token_count=self.count_tokens(chunk_text)
                    ))
                    chunk_number += 1
                
                current_chunk = [current_chunk[-1]] if current_chunk and self.chunk_overlap > 0 else []
                current_size = len(current_chunk[0]) if current_chunk else 0
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    metadata={'sentence_count': len(current_chunk),
                             'chunking_method': ChunkingMethod.SEMANTIC.value,
                             'similarity_threshold': self.similarity_threshold},
                    chunk_number=chunk_number,
                    token_count=self.count_tokens(chunk_text)
                ))
        
        return chunks
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        import numpy as np
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        return dot / (norm1 * norm2) if norm1 and norm2 else 0

class TextSplitterFactory:
    """Factory for creating text splitters"""
    
    @staticmethod
    def create_splitter(method: ChunkingMethod, **kwargs) -> TextSplitter:
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

# ==================== DEMO FUNCTIONS ====================

def compare_chunking_methods():
    """Compare different chunking methods"""
    print("=" * 70)
    print("✂️ CHUNKING METHODS COMPARISON (Day 26)")
    print("=" * 70)
    
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
    
    methods_to_test = [
        (ChunkingMethod.FIXED_SIZE, {"chunk_size": 300, "chunk_overlap": 50}),
        (ChunkingMethod.SENTENCE, {"chunk_size": 300, "chunk_overlap": 50}),
        (ChunkingMethod.PARAGRAPH, {"chunk_size": 300, "chunk_overlap": 50}),
        (ChunkingMethod.RECURSIVE, {"chunk_size": 300, "chunk_overlap": 50}),
        (ChunkingMethod.SLIDING_WINDOW, {"chunk_size": 300, "chunk_overlap": 50}),
        (ChunkingMethod.SEMANTIC, {"chunk_size": 300, "chunk_overlap": 50}),
    ]
    
    results = []
    
    for method, kwargs in methods_to_test:
        print(f"\n🔧 Testing {method.value.replace('_', ' ').title()}...")
        
        try:
            splitter = TextSplitterFactory.create_splitter(method, **kwargs)
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
            
            if chunks and len(chunks) >= 2:
                print(f"     Sample 1: {chunks[0].text[:80]}...")
                print(f"     Sample 2: {chunks[1].text[:80]}...")
        
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results.append({
                'method': method.value, 'chunks': 0, 'avg_chars': 0,
                'avg_tokens': 0, 'min_chars': 0, 'max_chars': 0
            })
    
    print("\n" + "=" * 70)
    print("📊 CHUNKING METHODS COMPARISON")
    print("=" * 70)
    
    print(f"\n{'Method':<20} {'Chunks':<10} {'Avg Chars':<12} {'Avg Tokens':<12} {'Min':<10} {'Max':<10}")
    print("-" * 74)
    
    for result in results:
        print(f"{result['method'].replace('_', ' '):<20} "
              f"{result['chunks']:<10} {result['avg_chars']:<12.0f} "
              f"{result['avg_tokens']:<12.0f} {result['min_chars']:<10} {result['max_chars']:<10}")

def demo_text_splitting():
    """Demonstrate text splitting capabilities"""
    print("\n🎯 TEXT SPLITTING DEMONSTRATION")
    print("=" * 60)
    
    sample_text = """
# Artificial Intelligence Overview

Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.

## Machine Learning

Machine learning is a subset of AI that enables computers to learn and improve from experience.

### Types of Machine Learning

1. Supervised Learning: The algorithm learns from labeled training data.
2. Unsupervised Learning: The algorithm finds patterns in unlabeled data.
3. Reinforcement Learning: The algorithm learns through trial and error.

## Deep Learning

Deep learning uses neural networks with multiple layers to analyze various factors of data.
"""
    
    print(f"\n📄 Sample text: {len(sample_text):,} characters")
    
    print("\n🔧 Testing Recursive Splitter...")
    try:
        splitter = RecursiveSplitter(chunk_size=400, chunk_overlap=100)
        chunks = splitter.split_text(sample_text)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n   Chunk {i} ({chunk.char_count} chars): {chunk.text[:100]}...")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n🔧 Testing Sliding Window Splitter...")
    try:
        splitter = SlidingWindowSplitter(chunk_size=300, chunk_overlap=75)
        chunks = splitter.split_text(sample_text)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks[:2], 1):
            print(f"   Chunk {i}: Window [{chunk.metadata['window_start']}:{chunk.metadata['window_end']}]")
    except Exception as e:
        print(f"   ✗ Failed: {e}")

def benchmark_splitters():
    """Benchmark different text splitters"""
    print("\n⏱️ TEXT SPLITTER BENCHMARK")
    print("=" * 60)
    
    test_text = '\n\n'.join([
        "Artificial Intelligence is transforming industries. " * 10,
        "Machine learning algorithms analyze vast datasets. " * 10,
        "Natural Language Processing enables interaction. " * 10,
    ])
    
    print(f"Test text: {len(test_text):,} characters")
    
    test_configs = [
        ("Fixed Size", ChunkingMethod.FIXED_SIZE),
        ("Sentence", ChunkingMethod.SENTENCE),
        ("Paragraph", ChunkingMethod.PARAGRAPH),
        ("Recursive", ChunkingMethod.RECURSIVE),
    ]
    
    results = []
    
    for name, method in test_configs:
        print(f"\n⏱️ Benchmarking {name}...")
        
        try:
            splitter = TextSplitterFactory.create_splitter(method, chunk_size=500, chunk_overlap=100)
            
            start_time = time.perf_counter()
            chunks = splitter.split_text(test_text)
            elapsed = time.perf_counter() - start_time
            
            results.append({
                'name': name, 'time': elapsed, 'chunks': len(chunks),
                'chunks_per_sec': len(chunks) / elapsed if elapsed > 0 else 0,
                'chars_per_sec': len(test_text) / elapsed if elapsed > 0 else 0
            })
            
            print(f"   ✓ Time: {elapsed:.3f}s, Chunks: {len(chunks)}")
        
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            results.append({'name': name, 'time': 0, 'chunks': 0, 'chunks_per_sec': 0, 'chars_per_sec': 0})
    
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\n{'Splitter':<15} {'Time (s)':<10} {'Chunks':<10} {'Chunks/s':<12} {'Chars/s'}")
    print("-" * 62)
    
    for r in results:
        print(f"{r['name']:<15} {r['time']:<10.3f} {r['chunks']:<10} "
              f"{r['chunks_per_sec']:<12.1f} {r['chars_per_sec']:,.0f}")

def main():
    """Run text splitting demonstrations"""
    print("🚀 Advanced Text Splitting Strategies")
    print("Day 26: Intelligent Text Chunking for RAG Systems")
    
    compare_chunking_methods()
    demo_text_splitting()
    benchmark_splitters()
    
    print("\n" + "=" * 70)
    print("✅ Day 26 Complete!")
    print("📚 You've built a comprehensive text splitting system!")
    print("   Next: Vector Store Setup (Day 27)")

if __name__ == "__main__":
    main()