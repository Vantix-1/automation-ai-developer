"""
🔤 Embeddings Demonstration
Day 22: Understanding and creating text embeddings
"""

import os
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

load_dotenv()

@dataclass
class EmbeddingResult:
    """Stores embedding results with metadata"""
    text: str
    embedding: np.ndarray
    model: str
    dimensions: int
    tokens: int = 0

class EmbeddingDemo:
    """Comprehensive embedding demonstration"""
    
    def __init__(self):
        self.results = []
    
    def demo_openai_embeddings(self) -> List[EmbeddingResult]:
        """Demo OpenAI embeddings (requires API key)"""
        try:
            from openai import OpenAI
            
            client = OpenAI()
            texts = [
                "The cat sat on the mat",
                "A feline rested on a rug",
                "Artificial intelligence is transforming technology",
                "Machine learning enables computers to learn from data",
                "Python is a popular programming language",
                "JavaScript is used for web development",
                "I enjoy reading science fiction novels",
                "Fantasy books with magic and dragons are exciting"
            ]
            
            print("🔤 Creating OpenAI embeddings...")
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-ada-002"
            )
            
            for i, data in enumerate(response.data):
                result = EmbeddingResult(
                    text=texts[i],
                    embedding=np.array(data.embedding),
                    model="text-embedding-ada-002",
                    dimensions=len(data.embedding),
                    tokens=data.usage.total_tokens
                )
                self.results.append(result)
                print(f"  ✓ '{texts[i][:30]}...' → {result.dimensions} dimensions")
            
            return self.results
            
        except ImportError:
            print("⚠️ OpenAI package not installed. Install with: pip install openai")
            return []
        except Exception as e:
            print(f"⚠️ OpenAI error: {e}")
            return []
    
    def demo_sentence_transformers(self) -> List[EmbeddingResult]:
        """Demo local Sentence Transformers embeddings"""
        try:
            from sentence_transformers import SentenceTransformer
            
            print("\n🔤 Creating Sentence Transformers embeddings...")
            
            # Try different models
            models = [
                ('all-MiniLM-L6-v2', 384),
                ('all-mpnet-base-v2', 768),
                ('paraphrase-MiniLM-L3-v2', 384)
            ]
            
            texts = [
                "The cat sat on the mat",
                "A feline rested on a rug",
                "Artificial intelligence is transforming technology",
                "Machine learning enables computers to learn from data"
            ]
            
            for model_name, dimensions in models:
                print(f"\n  Model: {model_name} ({dimensions} dimensions)")
                model = SentenceTransformer(model_name)
                
                # Create embeddings
                embeddings = model.encode(texts)
                
                for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                    result = EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        model=model_name,
                        dimensions=dimensions
                    )
                    self.results.append(result)
                    print(f"    ✓ '{text[:30]}...' → embedding created")
            
            return self.results
            
        except ImportError:
            print("⚠️ Sentence Transformers not installed. Install with: pip install sentence-transformers")
            return []
    
    def demo_huggingface_embeddings(self) -> List[EmbeddingResult]:
        """Demo Hugging Face embeddings"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            print("\n🔤 Creating Hugging Face embeddings...")
            
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            texts = [
                "Natural language processing is fascinating",
                "NLP allows computers to understand human language"
            ]
            
            for text in texts:
                # Tokenize
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                
                # Get embeddings
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Use mean pooling to get sentence embedding
                    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    model=model_name,
                    dimensions=embedding.shape[0],
                    tokens=len(inputs['input_ids'][0])
                )
                self.results.append(result)
                print(f"  ✓ '{text[:30]}...' → {result.dimensions} dimensions")
            
            return self.results
            
        except ImportError:
            print("⚠️ Transformers not installed. Install with: pip install transformers")
            return []
    
    def calculate_similarities(self):
        """Calculate and display similarities between embeddings"""
        if not self.results:
            print("⚠️ No embeddings to compare")
            return
        
        print("\n📊 Calculating Similarities")
        print("=" * 50)
        
        # Group by model
        by_model = {}
        for result in self.results:
            if result.model not in by_model:
                by_model[result.model] = []
            by_model[result.model].append(result)
        
        # Calculate similarities within each model
        for model_name, results in by_model.items():
            print(f"\nModel: {model_name}")
            print("-" * 30)
            
            if len(results) < 2:
                print("  Need at least 2 embeddings for comparison")
                continue
            
            # Calculate cosine similarities
            embeddings = np.array([r.embedding for r in results])
            
            # Normalize for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normalized = embeddings / norms
            
            # Cosine similarity matrix
            similarity_matrix = np.dot(normalized, normalized.T)
            
            # Print top similar pairs
            print("  Most similar pairs:")
            for i in range(len(results)):
                for j in range(i + 1, len(results)):
                    sim = similarity_matrix[i, j]
                    text1 = results[i].text[:20] + "..." if len(results[i].text) > 20 else results[i].text
                    text2 = results[j].text[:20] + "..." if len(results[j].text) > 20 else results[j].text
                    print(f"    '{text1}' ↔ '{text2}': {sim:.3f}")
    
    def visualize_embeddings(self):
        """Visualize embeddings in 2D space"""
        if len(self.results) < 3:
            print("⚠️ Need at least 3 embeddings for visualization")
            return
        
        print("\n🎨 Visualizing Embeddings")
        print("=" * 50)
        
        # Group by model for color coding
        models = list(set([r.model for r in self.results]))
        colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
        model_to_color = {model: colors[i] for i, model in enumerate(models)}
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Prepare data
        embeddings = np.array([r.embedding for r in self.results])
        texts = [r.text[:30] + "..." if len(r.text) > 30 else r.text for r in self.results]
        model_labels = [r.model for r in self.results]
        colors_list = [model_to_color[m] for m in model_labels]
        
        # Method 1: PCA
        print("  Applying PCA...")
        pca = PCA(n_components=2)
        embeddings_pca = pca.fit_transform(embeddings)
        
        ax1 = axes[0]
        scatter1 = ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                              c=colors_list, alpha=0.7, s=100)
        ax1.set_title('PCA Visualization of Embeddings')
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        
        # Add text labels
        for i, (x, y) in enumerate(embeddings_pca):
            ax1.annotate(f"{i+1}", (x, y), fontsize=8, ha='center')
        
        # Method 2: t-SNE
        print("  Applying t-SNE (this may take a moment)...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(embeddings)-1))
        embeddings_tsne = tsne.fit_transform(embeddings)
        
        ax2 = axes[1]
        scatter2 = ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                              c=colors_list, alpha=0.7, s=100)
        ax2.set_title('t-SNE Visualization of Embeddings')
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        
        # Add text labels
        for i, (x, y) in enumerate(embeddings_tsne):
            ax2.annotate(f"{i+1}", (x, y), fontsize=8, ha='center')
        
        # Create legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=model_to_color[model], 
                                 label=model, markersize=10) 
                          for model in models]
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(models))
        
        # Add text index
        print("\n  Text Index:")
        for i, text in enumerate(texts):
            print(f"    {i+1}. {text}")
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        plt.savefig('embeddings_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n  💾 Visualization saved as 'embeddings_visualization.png'")
        plt.show()
    
    def analyze_embedding_properties(self):
        """Analyze properties of embeddings"""
        if not self.results:
            print("⚠️ No embeddings to analyze")
            return
        
        print("\n🔍 Analyzing Embedding Properties")
        print("=" * 50)
        
        for result in self.results[:5]:  # Limit to first 5
            embedding = result.embedding
            
            print(f"\n📝 Text: '{result.text[:50]}...'")
            print(f"  Model: {result.model}")
            print(f"  Dimensions: {result.dimensions}")
            print(f"  Shape: {embedding.shape}")
            print(f"  Data Type: {embedding.dtype}")
            print(f"  Mean: {np.mean(embedding):.6f}")
            print(f"  Std Dev: {np.std(embedding):.6f}")
            print(f"  Min: {np.min(embedding):.6f}")
            print(f"  Max: {np.max(embedding):.6f}")
            print(f"  Norm (length): {np.linalg.norm(embedding):.6f}")
            
            # Check if normalized (typical for cosine similarity)
            norm = np.linalg.norm(embedding)
            if 0.95 < norm < 1.05:
                print(f"  ✅ Approximately normalized (norm ≈ {norm:.3f})")
            else:
                print(f"  ⚠️ Not normalized (norm = {norm:.3f})")
    
    def benchmark_embedding_models(self):
        """Benchmark different embedding models"""
        print("\n⏱️ Benchmarking Embedding Models")
        print("=" * 50)
        
        test_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Artificial intelligence will transform every industry",
            "Python programming language is versatile and powerful",
            "Machine learning models require large amounts of data"
        ]
        
        models_to_test = [
            ('Sentence Transformers - MiniLM', 'all-MiniLM-L6-v2'),
            ('Sentence Transformers - MPNet', 'all-mpnet-base-v2'),
        ]
        
        results = []
        
        for model_name, model_id in models_to_test:
            try:
                from sentence_transformers import SentenceTransformer
                import time
                
                print(f"\n  Testing: {model_name}")
                print(f"    Model ID: {model_id}")
                
                # Load model (first time may be slow)
                start_load = time.time()
                model = SentenceTransformer(model_id)
                load_time = time.time() - start_load
                print(f"    Load time: {load_time:.2f}s")
                
                # Warmup
                model.encode(["warmup"])
                
                # Benchmark encoding
                times = []
                for _ in range(5):  # Run 5 times
                    start = time.time()
                    embeddings = model.encode(test_texts)
                    times.append(time.time() - start)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                # Calculate tokens per second
                total_tokens = sum(len(text.split()) for text in test_texts)
                tokens_per_second = total_tokens / avg_time
                
                print(f"    Average encode time: {avg_time:.3f}s (±{std_time:.3f})")
                print(f"    Tokens per second: {tokens_per_second:.1f}")
                print(f"    Embedding shape: {embeddings.shape}")
                
                results.append({
                    'model': model_name,
                    'load_time': load_time,
                    'encode_time': avg_time,
                    'tokens_per_second': tokens_per_second,
                    'dimensions': embeddings.shape[1]
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        # Display summary
        if results:
            print("\n📊 Benchmark Summary:")
            print("-" * 40)
            print(f"{'Model':<30} {'Dim':<6} {'Load (s)':<8} {'Encode (s)':<10} {'Tokens/s':<10}")
            print("-" * 40)
            for r in results:
                print(f"{r['model']:<30} {r['dimensions']:<6} {r['load_time']:<8.2f} "
                      f"{r['encode_time']:<10.3f} {r['tokens_per_second']:<10.1f}")

def main():
    """Run all embedding demonstrations"""
    print("=" * 60)
    print("🔤 EMBEDDINGS DEMONSTRATION (Day 22)")
    print("=" * 60)
    
    demo = EmbeddingDemo()
    
    # Run different embedding demos
    print("\n1️⃣ OpenAI Embeddings")
    demo.demo_openai_embeddings()
    
    print("\n2️⃣ Sentence Transformers (Local)")
    demo.demo_sentence_transformers()
    
    print("\n3️⃣ Hugging Face Transformers")
    demo.demo_huggingface_embeddings()
    
    # Analyze results
    if demo.results:
        print("\n" + "=" * 60)
        demo.analyze_embedding_properties()
        
        print("\n" + "=" * 60)
        demo.calculate_similarities()
        
        print("\n" + "=" * 60)
        demo.benchmark_embedding_models()
        
        # Ask if user wants visualization
        visualize = input("\n🎨 Generate visualization? (y/n): ").lower()
        if visualize == 'y':
            demo.visualize_embeddings()
    
    print("\n" + "=" * 60)
    print("✅ Embeddings Demonstration Complete!")
    print(f"📊 Generated {len(demo.results)} embeddings from multiple models")
    print("📚 Next: Vector Similarity Calculations (Day 23)")

if __name__ == "__main__":
    main()