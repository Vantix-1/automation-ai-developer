"""
🎯 Chain Routing & Orchestration (Modern LCEL)
Day 16: Smart workflow routing and orchestration
"""

from typing import Dict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()

class ChainRouter:
    """Intelligent chain routing system"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7
        )
        self.parser = StrOutputParser()
        self.setup_chains()
    
    def setup_chains(self):
        """Setup specialized chains"""
        
        # QA Chain - For factual questions
        self.qa_prompt = ChatPromptTemplate.from_template(
            """You are a knowledgeable assistant. Answer this question accurately and concisely:

Question: {query}

Provide a clear, factual answer."""
        )
        self.qa_chain = self.qa_prompt | self.llm | self.parser
        
        # Analysis Chain - For analytical tasks
        self.analysis_prompt = ChatPromptTemplate.from_template(
            """You are an analytical expert. Provide deep analysis of:

Topic: {query}

Include:
1. Key insights
2. Trends
3. Implications"""
        )
        self.analysis_chain = self.analysis_prompt | self.llm | self.parser
        
        # Creative Chain - For creative tasks
        self.creative_prompt = ChatPromptTemplate.from_template(
            """You are a creative assistant. Generate creative content for:

Request: {query}

Be imaginative, original, and engaging."""
        )
        self.creative_chain = self.creative_prompt | self.llm | self.parser
        
        # Code Chain - For programming questions
        self.code_prompt = ChatPromptTemplate.from_template(
            """You are a programming expert. Help with this coding task:

Task: {query}

Provide code examples and explanations."""
        )
        self.code_chain = self.code_prompt | self.llm | self.parser
        
        # Router prompt
        self.router_prompt = ChatPromptTemplate.from_template(
            """Analyze this query and classify it into ONE category:

Query: {query}

Categories:
- qa: Factual questions, definitions, explanations
- analysis: Analytical tasks, comparisons, evaluations
- creative: Creative writing, brainstorming, storytelling
- code: Programming, technical coding questions

Respond with ONLY the category name (qa, analysis, creative, or code)."""
        )
        self.router_chain = self.router_prompt | self.llm | self.parser
    
    def route_query(self, query: str) -> Dict:
        """Route query to appropriate chain"""
        
        print(f"\n🔍 Analyzing query: {query}")
        
        # Determine route
        route = self.router_chain.invoke({"query": query}).strip().lower()
        print(f"📍 Routing to: {route} chain")
        
        # Execute appropriate chain
        if "qa" in route:
            response = self.qa_chain.invoke({"query": query})
            chain_type = "QA"
        elif "analysis" in route or "analys" in route:
            response = self.analysis_chain.invoke({"query": query})
            chain_type = "Analysis"
        elif "creative" in route:
            response = self.creative_chain.invoke({"query": query})
            chain_type = "Creative"
        elif "code" in route:
            response = self.code_chain.invoke({"query": query})
            chain_type = "Code"
        else:
            # Fallback to QA
            response = self.qa_chain.invoke({"query": query})
            chain_type = "QA (Fallback)"
        
        return {
            "query": query,
            "route": route,
            "chain_type": chain_type,
            "response": response
        }
    
    def orchestrate_workflow(self, task: str) -> Dict:
        """Complex multi-chain orchestration"""
        
        print(f"\n🎭 Orchestrating workflow for: {task}")
        
        # Step 1: Break down task
        breakdown_prompt = ChatPromptTemplate.from_template(
            """Break down this complex task into steps:

Task: {task}

List the steps needed."""
        )
        breakdown_chain = breakdown_prompt | self.llm | self.parser
        
        steps = breakdown_chain.invoke({"task": task})
        print(f"\n📋 Steps identified:\n{steps}")
        
        # Step 2: Execute each step through routing
        execution_prompt = ChatPromptTemplate.from_template(
            """Execute this step of the task:

Overall Task: {task}
Current Step: {step}

Complete this step."""
        )
        execution_chain = execution_prompt | self.llm | self.parser
        
        result = execution_chain.invoke({
            "task": task,
            "step": steps
        })
        
        # Step 3: Synthesize
        synthesis_prompt = ChatPromptTemplate.from_template(
            """Synthesize these results into a final deliverable:

Task: {task}
Steps: {steps}
Results: {results}

Create final output."""
        )
        synthesis_chain = synthesis_prompt | self.llm | self.parser
        
        final_output = synthesis_chain.invoke({
            "task": task,
            "steps": steps,
            "results": result
        })
        
        return {
            "task": task,
            "steps": steps,
            "execution": result,
            "final_output": final_output
        }

class BranchingChainDemo:
    """Demonstrate RunnableBranch for conditional routing"""
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        self.parser = StrOutputParser()
    
    def create_branching_chain(self):
        """Create a branching chain based on input length"""
        
        # Short answer chain
        short_prompt = ChatPromptTemplate.from_template(
            "Give a brief one-sentence answer: {question}"
        )
        short_chain = short_prompt | self.llm | self.parser
        
        # Detailed answer chain
        detailed_prompt = ChatPromptTemplate.from_template(
            "Provide a detailed, comprehensive answer: {question}"
        )
        detailed_chain = detailed_prompt | self.llm | self.parser
        
        # Create branch based on question length
        branch = RunnableBranch(
            (
                lambda x: len(x["question"]) < 50,
                short_chain
            ),
            detailed_chain  # Default branch
        )
        
        return branch
    
    def demo(self):
        """Run branching demo"""
        print("\n" + "="*50)
        print("🌳 BRANCHING CHAIN DEMO")
        print("="*50)
        
        chain = self.create_branching_chain()
        
        questions = [
            "What is AI?",  # Short - brief answer
            "Explain the complete history of artificial intelligence, including major milestones and breakthroughs"  # Long - detailed answer
        ]
        
        for q in questions:
            print(f"\n❓ Question ({len(q)} chars): {q}")
            result = chain.invoke({"question": q})
            print(f"💡 Answer: {result[:150]}...")

def run_examples():
    """Run all routing and orchestration examples"""
    print("=" * 60)
    print("🎯 CHAIN ROUTING & ORCHESTRATION (Day 16)")
    print("=" * 60)
    
    router = ChainRouter()
    
    # Example 1: Basic routing
    print("\n" + "="*60)
    print("1️⃣ BASIC QUERY ROUTING")
    print("="*60)
    
    queries = [
        "What is machine learning?",  # Should route to QA
        "Compare supervised vs unsupervised learning",  # Should route to Analysis
        "Write a short story about a robot",  # Should route to Creative
        "How do I implement a binary search in Python?"  # Should route to Code
    ]
    
    for query in queries:
        result = router.route_query(query)
        print(f"\n🎯 Chain: {result['chain_type']}")
        print(f"📝 Response: {result['response'][:150]}...")
        print("-" * 60)
    
    # Example 2: Complex orchestration
    print("\n" + "="*60)
    print("2️⃣ WORKFLOW ORCHESTRATION")
    print("="*60)
    
    complex_task = "Create a simple web scraper in Python with error handling and documentation"
    result = router.orchestrate_workflow(complex_task)
    
    print(f"\n📊 Final Output:")
    print(result['final_output'][:300])
    print("...")
    
    # Example 3: Branching
    branching_demo = BranchingChainDemo()
    branching_demo.demo()
    
    print("\n" + "="*60)
    print("✅ Day 16 Complete!")
    print("="*60)
    print("\n📚 Key Concepts:")
    print("  • Query Classification: Route to specialized chains")
    print("  • Workflow Orchestration: Combine multiple chains")
    print("  • Conditional Branching: Different paths based on input")
    print("  • Fallback Mechanisms: Handle unknown query types")

if __name__ == "__main__":
    run_examples()