"""
Simple Chains - Days 11-13
Building and customizing LangChain chains
"""

import os
from typing import List, Dict, Any, Optional
from colorama import Fore, Style, init
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_community.callbacks import get_openai_callback

from dotenv import load_dotenv

# Initialize colorama
init(autoreset=True)

load_dotenv()

class SimpleChainBuilder:
    """Builder for creating and testing LangChain chains"""
    
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.7):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key
        )
        
        print(Fore.CYAN + "🔗 Simple Chain Builder Initialized")
        print(Fore.YELLOW + f"🤖 Model: {model}, Temperature: {temperature}")
    
    def create_basic_chain(self):
        """Create and test a basic LCEL chain"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "🔗 Basic LCEL Chain")
        print(Fore.CYAN + "="*50)
        
        # Create prompt template
        template = """You are a {expert_type}. 
        
        Question: {question}
        
        Provide a detailed answer suitable for a {audience} audience.
        
        Answer:"""
        
        prompt = PromptTemplate(
            input_variables=["expert_type", "question", "audience"],
            template=template
        )
        
        # Create chain using LCEL
        chain = prompt | self.llm | StrOutputParser()
        
        # Test the chain
        inputs = {
            "expert_type": "data scientist",
            "question": "What is overfitting in machine learning?",
            "audience": "beginner"
        }
        
        with get_openai_callback() as cb:
            result = chain.invoke(inputs)
            
            print(Fore.GREEN + "\n✅ Chain executed successfully!")
            print(Fore.WHITE + f"\n📥 Inputs: {inputs}")
            print(Fore.WHITE + f"\n📤 Output:")
            print(Fore.WHITE + result)
            print(Fore.CYAN + f"\n📊 Cost: ${cb.total_cost:.6f}")
        
        return chain
    
    def create_transformation_chain(self):
        """Create a chain with transformation steps"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "🔄 Transformation Chain")
        print(Fore.CYAN + "="*50)
        
        # Define transformation functions
        def clean_text(inputs: Dict) -> Dict:
            """Clean and preprocess text"""
            text = inputs["text"]
            # Simple cleaning
            cleaned = text.strip().replace('\n', ' ').replace('  ', ' ')
            return {"cleaned_text": cleaned, "text": text}
        
        def count_words(inputs: Dict) -> Dict:
            """Count words in text"""
            text = inputs["cleaned_text"]
            word_count = len(text.split())
            return {"word_count": word_count, "cleaned_text": text, "text": inputs["text"]}
        
        def analyze_complexity(inputs: Dict) -> Dict:
            """Analyze text complexity based on word count"""
            count = inputs["word_count"]
            
            if count < 50:
                complexity = "Simple"
            elif count < 150:
                complexity = "Moderate"
            else:
                complexity = "Complex"
            
            return {
                "complexity": complexity, 
                "word_count": count,
                "original_text": inputs["text"]
            }
        
        # Create transformation chain using LCEL
        chain = (
            RunnableLambda(clean_text) 
            | RunnableLambda(count_words) 
            | RunnableLambda(analyze_complexity)
        )
        
        # Test the chain
        test_text = """
        Machine learning is a branch of artificial intelligence that focuses on 
        building systems that learn from data. These systems improve their 
        performance as they are exposed to more data over time.
        
        Deep learning is a subset of machine learning that uses neural networks 
        with many layers. It has revolutionized fields like computer vision and 
        natural language processing.
        """
        
        print(Fore.YELLOW + "\n📝 Testing with sample text...")
        result = chain.invoke({"text": test_text})
        
        print(Fore.GREEN + "\n✅ Transformation Complete!")
        print(Fore.CYAN + f"\n📊 Results:")
        print(Fore.WHITE + f"Original text length: {len(test_text)} characters")
        print(Fore.WHITE + f"Word count: {result.get('word_count', 'N/A')}")
        print(Fore.WHITE + f"Complexity: {result['complexity']}")
        
        return chain
    
    def create_conditional_chain(self):
        """Create a chain with conditional logic"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "🎯 Conditional Chain")
        print(Fore.CYAN + "="*50)
        
        # Chain 1: Classify query type
        classify_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Classify this query into one of these categories: 
            [code, explanation, creative, fact]. 
            
            Query: {query}
            
            Category (one word only):"""
        )
        
        classify_chain = classify_prompt | self.llm | StrOutputParser()
        
        # Chain 2: Generate appropriate response based on category
        response_prompt = PromptTemplate(
            input_variables=["query", "category"],
            template="""Based on the category '{category}', respond to this query:
            
            Query: {query}
            
            Response:"""
        )
        
        # Create sequential chain using LCEL
        def add_category(inputs):
            """Helper to pass through query and add category"""
            query = inputs["query"]
            category = inputs["category"]
            return {"query": query, "category": category}
        
        sequential_chain = (
            {"query": RunnablePassthrough(), "category": classify_chain}
            | RunnableLambda(add_category)
            | response_prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Test with different queries
        test_queries = [
            "Write a Python function to calculate factorial",
            "Explain the theory of relativity",
            "Write a short poem about artificial intelligence",
            "What's the capital of France?"
        ]
        
        print(Fore.YELLOW + "\n🧪 Testing conditional chain with different queries...")
        
        total_cost = 0
        
        for i, query in enumerate(test_queries, 1):
            print(Fore.CYAN + f"\n📝 Query {i}: {query}")
            
            with get_openai_callback() as cb:
                # First get category
                category = classify_chain.invoke({"query": query})
                # Then get response
                response = sequential_chain.invoke({"query": query})
                
                total_cost += cb.total_cost
                
                print(Fore.WHITE + f"Category: {category.strip()}")
                print(Fore.WHITE + f"Response: {response[:100]}...")
                print(Fore.YELLOW + f"Cost: ${cb.total_cost:.6f}")
        
        print(Fore.CYAN + f"\n💰 Total cost for all queries: ${total_cost:.6f}")
        
        return sequential_chain
    
    def create_chat_prompt_chain(self):
        """Create chain with chat prompt templates"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "💬 Chat Prompt Chain")
        print(Fore.CYAN + "="*50)
        
        # Create chat prompt template
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant specialized in {domain}."),
            ("human", "Hello! I need help with {topic}."),
            ("ai", "I'd be happy to help you with {topic}. What specifically would you like to know?"),
            ("human", "{question}")
        ])
        
        # Create chain using LCEL
        chain = chat_prompt | self.llm | StrOutputParser()
        
        # Test the chain
        test_cases = [
            {
                "domain": "programming",
                "topic": "Python",
                "question": "What are decorators and how do I use them?"
            },
            {
                "domain": "cooking",
                "topic": "Italian cuisine",
                "question": "How do I make authentic pasta carbonara?"
            }
        ]
        
        print(Fore.YELLOW + "\n🧪 Testing chat prompt chain...")
        
        for i, test_case in enumerate(test_cases, 1):
            print(Fore.CYAN + f"\n🧪 Test {i}: {test_case['domain']} - {test_case['topic']}")
            
            with get_openai_callback() as cb:
                result = chain.invoke(test_case)
                
                print(Fore.WHITE + f"\nQuestion: {test_case['question']}")
                print(Fore.WHITE + f"\nAnswer: {result[:150]}...")
                print(Fore.YELLOW + f"Cost: ${cb.total_cost:.6f}")
        
        return chain
    
    def create_parallel_chains(self):
        """Create and run multiple chains in parallel"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "⚡ Parallel Chains")
        print(Fore.CYAN + "="*50)
        
        # Define different chains for different purposes
        
        # Chain 1: Summarizer
        summarize_prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize this text in 2-3 sentences:\n\n{text}\n\nSummary:"
        )
        summary_chain = summarize_prompt | self.llm | StrOutputParser()
        
        # Chain 2: Sentiment analyzer
        sentiment_prompt = PromptTemplate(
            input_variables=["text"],
            template="Analyze the sentiment of this text (positive, negative, neutral):\n\n{text}\n\nSentiment:"
        )
        sentiment_chain = sentiment_prompt | self.llm | StrOutputParser()
        
        # Chain 3: Key points extractor
        keypoints_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract 3-5 key points from this text:\n\n{text}\n\nKey points:"
        )
        keypoints_chain = keypoints_prompt | self.llm | StrOutputParser()
        
        # Create parallel chain using RunnableParallel
        parallel_chain = RunnableParallel(
            summary=summary_chain,
            sentiment=sentiment_chain,
            keypoints=keypoints_chain
        )
        
        # Test text
        test_text = """
        Artificial Intelligence has made significant strides in recent years. 
        From language models that can write human-like text to computer vision 
        systems that can identify objects with remarkable accuracy, AI is 
        transforming industries. However, there are concerns about job 
        displacement and ethical implications that need to be addressed.
        """
        
        print(Fore.YELLOW + f"\n📝 Processing text: '{test_text[:80]}...'")
        print(Fore.YELLOW + "\n🔗 Running parallel chains...")
        
        # Run chains in parallel
        with get_openai_callback() as cb:
            results = parallel_chain.invoke({"text": test_text})
            total_cost = cb.total_cost
        
        # Display all results
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "📊 Parallel Chain Results")
        print(Fore.CYAN + "="*50)
        
        for name, result in results.items():
            print(Fore.YELLOW + f"\n{name.upper()}:")
            print(Fore.WHITE + f"{result}")
        
        print(Fore.CYAN + f"\n💰 Total parallel processing cost: ${total_cost:.6f}")
        
        return parallel_chain
    
    def chain_comparison(self):
        """Compare different chain configurations"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "⚖️ Chain Comparison")
        print(Fore.CYAN + "="*50)
        
        # Same task, different chain configurations
        task = "Explain machine learning to a beginner"
        
        configurations = [
            {
                "name": "Simple Chain",
                "temperature": 0.3,
                "template": "Explain {topic} simply."
            },
            {
                "name": "Detailed Chain",
                "temperature": 0.7,
                "template": """You are an expert teacher. Explain {topic} to a complete beginner.
                
                Include:
                1. A simple definition
                2. A real-world analogy
                3. One example application
                
                Explanation:"""
            },
            {
                "name": "Interactive Chain",
                "temperature": 0.9,
                "template": """Imagine you're having a conversation with a curious student about {topic}.
                
                Student: "What is {topic}?"
                
                Teacher:"""
            }
        ]
        
        print(Fore.YELLOW + f"\n📋 Task: {task}")
        print(Fore.YELLOW + "\n🔬 Comparing chain configurations...")
        
        comparison_results = []
        
        for config in configurations:
            print(Fore.CYAN + f"\n🔧 Testing: {config['name']}")
            
            # Create chain with this configuration
            prompt = PromptTemplate(
                input_variables=["topic"],
                template=config['template']
            )
            
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=config['temperature'],
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
            chain = prompt | llm | StrOutputParser()
            
            with get_openai_callback() as cb:
                result = chain.invoke({"topic": "machine learning"})
                
                comparison_results.append({
                    "name": config['name'],
                    "temperature": config['temperature'],
                    "tokens": cb.total_tokens,
                    "cost": cb.total_cost,
                    "result_preview": result[:100] + "..."
                })
                
                print(Fore.WHITE + f"Temperature: {config['temperature']}")
                print(Fore.WHITE + f"Tokens: {cb.total_tokens}")
                print(Fore.WHITE + f"Cost: ${cb.total_cost:.6f}")
                print(Fore.WHITE + f"Preview: {result[:80]}...")
        
        # Display comparison table
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "📈 Comparison Summary")
        print(Fore.CYAN + "="*50)
        
        print(Fore.YELLOW + "\n{:<15} {:<10} {:<10} {:<12} {}".format(
            "Chain Name", "Temp", "Tokens", "Cost", "Preview"
        ))
        print(Fore.YELLOW + "-" * 70)
        
        for result in comparison_results:
            print(Fore.WHITE + "{:<15} {:<10} {:<10} ${:<11.6f} {}".format(
                result["name"],
                result["temperature"],
                result["tokens"],
                result["cost"],
                result["result_preview"][:30] + "..."
            ))
    
    def interactive_chain_builder(self):
        """Interactive chain building demo"""
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + "🔧 Interactive Chain Builder")
        print(Fore.CYAN + "="*70)
        
        while True:
            print(Fore.YELLOW + "\n" + "━" * 50)
            print(Fore.YELLOW + "🔗 Available Chain Types:")
            print(Fore.YELLOW + "1. Basic LCEL Chain")
            print(Fore.YELLOW + "2. Transformation Chain")
            print(Fore.YELLOW + "3. Conditional Chain")
            print(Fore.YELLOW + "4. Chat Prompt Chain")
            print(Fore.YELLOW + "5. Parallel Chains")
            print(Fore.YELLOW + "6. Chain Comparison")
            print(Fore.YELLOW + "7. Exit")
            print(Fore.YELLOW + "━" * 50)
            
            choice = input(Fore.WHITE + "\nSelect chain type (1-7): ").strip()
            
            if choice == "7":
                print(Fore.YELLOW + "\n👋 Goodbye!")
                break
            
            try:
                if choice == "1":
                    self.create_basic_chain()
                elif choice == "2":
                    self.create_transformation_chain()
                elif choice == "3":
                    self.create_conditional_chain()
                elif choice == "4":
                    self.create_chat_prompt_chain()
                elif choice == "5":
                    self.create_parallel_chains()
                elif choice == "6":
                    self.chain_comparison()
                else:
                    print(Fore.RED + "❌ Invalid choice")
            except Exception as e:
                print(Fore.RED + f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
            
            input(Fore.YELLOW + "\nPress Enter to continue...")

def main():
    """Main function"""
    try:
        builder = SimpleChainBuilder()
        builder.interactive_chain_builder()
        
    except Exception as e:
        print(Fore.RED + f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()