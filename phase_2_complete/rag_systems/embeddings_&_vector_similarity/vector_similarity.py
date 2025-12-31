"""
📐 Vector Similarity & Distance Metrics
Day 23: Implementing and comparing vector similarity measures
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import time
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import seaborn as sns

class DistanceMetric(Enum):
    """Supported distance/similarity metrics"""
    COSINE_SIMILARITY = "cosine_similarity"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    DOT_PRODUCT = "dot_product"
    JACCARD = "jaccard"  # For binary vectors
    PEARSON = "pearson"

@dataclass
class SimilarityResult:
    """Stores similarity calculation results"""
    vector1: np.ndarray
    vector2: np.ndarray
    metric: DistanceMetric
    value: float
    computation_time: float
    normalized: bool = False

class VectorSimilarityCalculator:
    """Comprehensive vector similarity calculations"""
    
    def __init__(self):
        self.results = []
        self.setup_plotting()
    
    def setup_plotting(self):
        """Setup matplotlib for better visualizations"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def cosine_similarity_custom(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity manually"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def euclidean_distance_custom(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Euclidean distance manually"""
        return np.sqrt(np.sum((vec1 - vec2) ** 2))
    
    def manhattan_distance_custom(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Manhattan distance manually"""
        return np.sum(np.abs(vec1 - vec2))
    
    def jaccard_similarity_custom(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Jaccard similarity for binary vectors"""
        # Convert to binary if not already
        vec1_bin = (vec1 > 0).astype(int)
        vec2_bin = (vec2 > 0).astype(int)
        
        intersection = np.sum(np.minimum(vec1_bin, vec2_bin))
        union = np.sum(np.maximum(vec1_bin, vec2_bin))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def pearson_correlation_custom(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient"""
        # Center the vectors
        vec1_centered = vec1 - np.mean(vec1)
        vec2_centered = vec2 - np.mean(vec2)
        
        numerator = np.sum(vec1_centered * vec2_centered)
        denominator = np.sqrt(np.sum(vec1_centered ** 2) * np.sum(vec2_centered ** 2))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def calculate_similarity(self, vec1: np.ndarray, vec2: np.ndarray, 
                           metric: DistanceMetric) -> SimilarityResult:
        """Calculate similarity using specified metric"""
        start_time = time.perf_counter()
        
        if metric == DistanceMetric.COSINE_SIMILARITY:
            value = self.cosine_similarity_custom(vec1, vec2)
            normalized = True
        elif metric == DistanceMetric.EUCLIDEAN:
            value = self.euclidean_distance_custom(vec1, vec2)
            normalized = False
        elif metric == DistanceMetric.MANHATTAN:
            value = self.manhattan_distance_custom(vec1, vec2)
            normalized = False
        elif metric == DistanceMetric.DOT_PRODUCT:
            value = np.dot(vec1, vec2)
            normalized = False
        elif metric == DistanceMetric.JACCARD:
            value = self.jaccard_similarity_custom(vec1, vec2)
            normalized = True
        elif metric == DistanceMetric.PEARSON:
            value = self.pearson_correlation_custom(vec1, vec2)
            normalized = True
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        computation_time = time.perf_counter() - start_time
        
        return SimilarityResult(
            vector1=vec1,
            vector2=vec2,
            metric=metric,
            value=value,
            computation_time=computation_time,
            normalized=normalized
        )
    
    def compare_all_metrics(self, vec1: np.ndarray, vec2: np.ndarray) -> List[SimilarityResult]:
        """Compare all similarity metrics for given vectors"""
        print(f"🔍 Comparing {len(vec1)}-dimensional vectors")
        print(f"  Vector 1 mean: {np.mean(vec1):.4f}, std: {np.std(vec1):.4f}")
        print(f"  Vector 2 mean: {np.mean(vec2):.4f}, std: {np.std(vec2):.4f}")
        
        results = []
        for metric in DistanceMetric:
            try:
                result = self.calculate_similarity(vec1, vec2, metric)
                results.append(result)
                print(f"  {metric.value:20s}: {result.value:10.6f} ({result.computation_time*1000:6.3f} ms)")
            except Exception as e:
                print(f"  {metric.value:20s}: ERROR - {e}")
        
        self.results.extend(results)
        return results
    
    def generate_test_vectors(self, n: int = 100, dim: int = 300) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test vectors with varying similarity"""
        print(f"\n📊 Generating test vectors (dim={dim}, n={n})")
        
        # Generate random base vector
        np.random.seed(42)
        base_vector = np.random.randn(dim)
        
        # Normalize base vector
        base_vector = base_vector / np.linalg.norm(base_vector)
        
        # Generate similar vectors with controlled similarity
        vectors = []
        similarities = np.linspace(0.1, 0.9, n)  # Range of similarities
        
        for sim in similarities:
            # Create orthogonal component
            orthogonal = np.random.randn(dim)
            # Make it orthogonal to base vector
            orthogonal = orthogonal - np.dot(orthogonal, base_vector) * base_vector
            orthogonal = orthogonal / np.linalg.norm(orthogonal)
            
            # Create new vector with controlled similarity
            theta = np.arccos(sim)  # Angle for desired cosine similarity
            new_vector = np.cos(theta) * base_vector + np.sin(theta) * orthogonal
            
            vectors.append(new_vector)
        
        return np.array(vectors), similarities
    
    def benchmark_performance(self, dimensions: List[int] = [50, 100, 300, 768, 1536], 
                            n_pairs: int = 1000):
        """Benchmark similarity calculations across dimensions"""
        print("\n⏱️ Performance Benchmarking")
        print("=" * 60)
        
        results = []
        
        for dim in dimensions:
            print(f"\nDimension: {dim}")
            print("-" * 40)
            
            # Generate random vectors
            np.random.seed(42)
            vectors = np.random.randn(n_pairs * 2, dim).astype(np.float32)
            
            # Benchmark cosine similarity
            times = []
            for i in range(n_pairs):
                vec1 = vectors[i*2]
                vec2 = vectors[i*2 + 1]
                
                start = time.perf_counter()
                _ = self.cosine_similarity_custom(vec1, vec2)
                times.append(time.perf_counter() - start)
            
            avg_time = np.mean(times) * 1000  # Convert to ms
            std_time = np.std(times) * 1000
            
            results.append({
                'dimension': dim,
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'pairs_per_second': 1000 / avg_time * 1000 if avg_time > 0 else 0
            })
            
            print(f"  Average time: {avg_time:.3f} ± {std_time:.3f} ms")
            print(f"  Pairs per second: {results[-1]['pairs_per_second']:.0f}")
        
        # Plot results
        self.plot_benchmark_results(results)
        return results
    
    def plot_benchmark_results(self, results: List[Dict]):
        """Plot benchmarking results"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Time vs Dimension
        dimensions = [r['dimension'] for r in results]
        times = [r['avg_time_ms'] for r in results]
        stds = [r['std_time_ms'] for r in results]
        
        axes[0].errorbar(dimensions, times, yerr=stds, fmt='o-', capsize=5)
        axes[0].set_xlabel('Vector Dimension')
        axes[0].set_ylabel('Time per pair (ms)')
        axes[0].set_title('Computation Time vs Vector Dimension')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        
        # Plot 2: Throughput vs Dimension
        throughput = [r['pairs_per_second'] for r in results]
        
        axes[1].plot(dimensions, throughput, 's-', color='orange')
        axes[1].set_xlabel('Vector Dimension')
        axes[1].set_ylabel('Pairs per second')
        axes[1].set_title('Throughput vs Vector Dimension')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('similarity_benchmark.png', dpi=150, bbox_inches='tight')
        print(f"\n📈 Benchmark plot saved as 'similarity_benchmark.png'")
        plt.show()
    
    def demo_similarity_relationships(self):
        """Demonstrate relationships between different metrics"""
        print("\n📐 Similarity Metric Relationships")
        print("=" * 60)
        
        # Generate test data
        n_samples = 1000
        dim = 100
        
        np.random.seed(42)
        base = np.random.randn(dim)
        base = base / np.linalg.norm(base)
        
        # Generate vectors with varying similarity
        cos_sims = []
        euclid_dists = []
        manhattan_dists = []
        dot_products = []
        
        for _ in range(n_samples):
            # Generate random vector
            vec = np.random.randn(dim)
            vec = vec / np.linalg.norm(vec)
            
            # Calculate all metrics
            cos_sim = self.cosine_similarity_custom(base, vec)
            euclid_dist = self.euclidean_distance_custom(base, vec)
            manhattan_dist = self.manhattan_distance_custom(base, vec)
            dot_product = np.dot(base, vec)
            
            cos_sims.append(cos_sim)
            euclid_dists.append(euclid_dist)
            manhattan_dists.append(manhattan_dist)
            dot_products.append(dot_product)
        
        # Create relationship plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Cosine vs Euclidean
        axes[0, 0].scatter(cos_sims, euclid_dists, alpha=0.6, s=10)
        axes[0, 0].set_xlabel('Cosine Similarity')
        axes[0, 0].set_ylabel('Euclidean Distance')
        axes[0, 0].set_title('Cosine Similarity vs Euclidean Distance')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Cosine vs Manhattan
        axes[0, 1].scatter(cos_sims, manhattan_dists, alpha=0.6, s=10, color='green')
        axes[0, 1].set_xlabel('Cosine Similarity')
        axes[0, 1].set_ylabel('Manhattan Distance')
        axes[0, 1].set_title('Cosine Similarity vs Manhattan Distance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Cosine vs Dot Product
        axes[1, 0].scatter(cos_sims, dot_products, alpha=0.6, s=10, color='red')
        axes[1, 0].set_xlabel('Cosine Similarity')
        axes[1, 0].set_ylabel('Dot Product')
        axes[1, 0].set_title('Cosine Similarity vs Dot Product')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Euclidean vs Manhattan
        axes[1, 1].scatter(euclid_dists, manhattan_dists, alpha=0.6, s=10, color='purple')
        axes[1, 1].set_xlabel('Euclidean Distance')
        axes[1, 1].set_ylabel('Manhattan Distance')
        axes[1, 1].set_title('Euclidean vs Manhattan Distance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('metric_relationships.png', dpi=150, bbox_inches='tight')
        print(f"📊 Relationship plots saved as 'metric_relationships.png'")
        plt.show()
        
        # Print correlation coefficients
        print("\n📊 Correlation between metrics:")
        print(f"  Cosine vs Euclidean: {np.corrcoef(cos_sims, euclid_dists)[0,1]:.3f}")
        print(f"  Cosine vs Manhattan: {np.corrcoef(cos_sims, manhattan_dists)[0,1]:.3f}")
        print(f"  Cosine vs Dot Product: {np.corrcoef(cos_sims, dot_products)[0,1]:.3f}")
        print(f"  Euclidean vs Manhattan: {np.corrcoef(euclid_dists, manhattan_dists)[0,1]:.3f}")
    
    def demo_normalization_effects(self):
        """Demonstrate effects of vector normalization"""
        print("\n⚖️ Vector Normalization Effects")
        print("=" * 60)
        
        # Create example vectors
        np.random.seed(42)
        
        # Unnormalized vectors
        vec1 = np.array([1, 2, 3, 4, 5])
        vec2 = np.array([2, 3, 4, 5, 6])
        
        # Normalized vectors
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        print("Original vectors:")
        print(f"  Vector 1: {vec1}")
        print(f"  Vector 2: {vec2}")
        print(f"  Norms: {np.linalg.norm(vec1):.3f}, {np.linalg.norm(vec2):.3f}")
        
        print("\nNormalized vectors:")
        print(f"  Vector 1: {vec1_norm}")
        print(f"  Vector 2: {vec2_norm}")
        print(f"  Norms: {np.linalg.norm(vec1_norm):.3f}, {np.linalg.norm(vec2_norm):.3f}")
        
        # Compare metrics
        metrics_to_compare = [DistanceMetric.COSINE_SIMILARITY, 
                            DistanceMetric.EUCLIDEAN, 
                            DistanceMetric.DOT_PRODUCT]
        
        print("\n📊 Metric comparison (original vs normalized):")
        print("-" * 60)
        
        for metric in metrics_to_compare:
            orig_result = self.calculate_similarity(vec1, vec2, metric)
            norm_result = self.calculate_similarity(vec1_norm, vec2_norm, metric)
            
            print(f"\n{metric.value}:")
            print(f"  Original: {orig_result.value:.6f}")
            print(f"  Normalized: {norm_result.value:.6f}")
            
            if metric == DistanceMetric.COSINE_SIMILARITY:
                print(f"  ✅ Cosine similarity is invariant to normalization")
            elif metric == DistanceMetric.EUCLIDEAN:
                print(f"  ⚠️ Euclidean distance changes with normalization")
            elif metric == DistanceMetric.DOT_PRODUCT:
                print(f"  ⚠️ Dot product changes with normalization")
    
    def interactive_similarity_explorer(self):
        """Interactive exploration of similarity metrics"""
        print("\n🎮 Interactive Similarity Explorer")
        print("=" * 60)
        
        try:
            import ipywidgets as widgets
            from IPython.display import display, clear_output
        except ImportError:
            print("⚠️ ipywidgets not installed. Install with: pip install ipywidgets")
            return
        
        # Create widgets
        dim_slider = widgets.IntSlider(value=3, min=2, max=10, step=1, description='Dimensions:')
        vector1_text = widgets.Text(value='1,2,3', description='Vector 1:')
        vector2_text = widgets.Text(value='4,5,6', description='Vector 2:')
        metric_dropdown = widgets.Dropdown(
            options=[(m.value, m) for m in DistanceMetric],
            value=DistanceMetric.COSINE_SIMILARITY,
            description='Metric:'
        )
        calculate_btn = widgets.Button(description='Calculate Similarity')
        output = widgets.Output()
        
        def on_calculate_click(b):
            with output:
                clear_output()
                
                try:
                    # Parse vectors
                    vec1 = np.array([float(x.strip()) for x in vector1_text.value.split(',')])
                    vec2 = np.array([float(x.strip()) for x in vector2_text.value.split(',')])
                    
                    if len(vec1) != len(vec2):
                        print("❌ Vectors must have same length!")
                        return
                    
                    # Calculate similarity
                    result = self.calculate_similarity(vec1, vec2, metric_dropdown.value)
                    
                    print(f"✅ {metric_dropdown.value.value}: {result.value:.6f}")
                    print(f"   Computation time: {result.computation_time*1000:.3f} ms")
                    print(f"   Vector 1: {vec1}")
                    print(f"   Vector 2: {vec2}")
                    
                    # Add to results
                    self.results.append(result)
                    
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        calculate_btn.on_click(on_calculate_click)
        
        # Display widgets
        display(widgets.VBox([
            dim_slider,
            vector1_text,
            vector2_text,
            metric_dropdown,
            calculate_btn,
            output
        ]))

def main():
    """Run all vector similarity demonstrations"""
    print("=" * 70)
    print("📐 VECTOR SIMILARITY & DISTANCE METRICS (Day 23)")
    print("=" * 70)
    
    calculator = VectorSimilarityCalculator()
    
    # Demo 1: Basic similarity calculations
    print("\n1️⃣ Basic Similarity Calculations")
    print("-" * 40)
    
    # Create test vectors
    np.random.seed(42)
    vec1 = np.random.randn(10)
    vec2 = np.random.randn(10)
    
    calculator.compare_all_metrics(vec1, vec2)
    
    # Demo 2: Normalization effects
    calculator.demo_normalization_effects()
    
    # Demo 3: Performance benchmarking
    dimensions = [50, 100, 300, 768, 1536]
    calculator.benchmark_performance(dimensions)
    
    # Demo 4: Metric relationships
    calculator.demo_similarity_relationships()
    
    # Demo 5: Interactive explorer (optional)
    try:
        run_interactive = input("\n🎮 Run interactive explorer? (y/n): ").lower()
        if run_interactive == 'y':
            calculator.interactive_similarity_explorer()
    except:
        print("Skipping interactive explorer...")
    
    print("\n" + "=" * 70)
    print("✅ Vector Similarity Demonstration Complete!")
    print(f"📊 Calculated {len(calculator.results)} similarity results")
    print("📚 Next: Semantic Search Engine (Day 24)")

if __name__ == "__main__":
    main()