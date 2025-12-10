"""
🚀 Sequential Chains Implementation
Day 14: Multi-step AI workflows with LangChain
"""

import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# Load environment
load_dotenv()

class SequentialChainWorkflows:
    """Complete sequential chain implementations"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
    
    def basic_chain(self, product_type: str):
        """Simple 2-step chain: Name → Slogan"""
        print(f"📦 Creating marketing for: {product_type}")
        
        # Step 1: Name generation
        name_template = PromptTemplate(
            input_variables=["product"],
            template="Generate 3 creative names for a {product}"
        )
        name_chain = LLMChain(llm=self.llm, prompt=name_template, output_key="names")
        
        # Step 2: Slogan generation
        slogan_template = PromptTemplate(
            input_variables=["names"],
            template="Create marketing slogans for these product names: {names}"
        )
        slogan_chain = LLMChain(llm=self.llm, prompt=slogan_template, output_key="slogans")
        
        # Combine chains
        chain = SimpleSequentialChain(
            chains=[name_chain, slogan_chain],
            verbose=True
        )
        
        return chain.run(product_type)
    
    def content_creation_chain(self, topic: str, tone: str = "professional"):
        """Complete content creation pipeline"""
        print(f"📝 Creating {tone} content about: {topic}")
        
        # 1. Outline generation
        outline_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["topic", "tone"],
                template="Create a detailed outline about {topic} in a {tone} tone"
            ),
            output_key="outline"
        )
        
        # 2. Content writing
        content_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["outline", "tone"],
                template="Write full content based on this outline: {outline}\nTone: {tone}"
            ),
            output_key="content"
        )
        
        # 3. SEO optimization
        seo_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["content", "topic"],
                template="Add SEO keywords to this content: {content}\nPrimary topic: {topic}"
            ),
            output_key="optimized_content"
        )
        
        # Full workflow
        workflow = SequentialChain(
            chains=[outline_chain, content_chain, seo_chain],
            input_variables=["topic", "tone"],
            output_variables=["outline", "content", "optimized_content"],
            verbose=True
        )
        
        return workflow({"topic": topic, "tone": tone})
    
    def ecommerce_workflow(self, product: str, target: str):
        """E-commerce product launch workflow"""
        print(f"🛍️ Launching {product} for {target}")
        
        chains = []
        
        # 1. Product description
        desc_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["product", "target"],
                template="Write compelling product description for {product} targeting {target}"
            ),
            output_key="description"
        )
        chains.append(desc_chain)
        
        # 2. Target audience analysis
        audience_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["product", "target"],
                template="Analyze target audience for {product}: {target}"
            ),
            output_key="audience_analysis"
        )
        chains.append(audience_chain)
        
        # 3. Marketing strategy
        strategy_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["description", "audience_analysis"],
                template="Create marketing strategy based on:\nProduct: {description}\nAudience: {audience_analysis}"
            ),
            output_key="strategy"
        )
        chains.append(strategy_chain)
        
        # Execute
        final_chain = SequentialChain(
            chains=chains,
            input_variables=["product", "target"],
            output_variables=["description", "audience_analysis", "strategy"],
            verbose=True
        )
        
        return final_chain({"product": product, "target": target})

def run_examples():
    """Run all sequential chain examples"""
    print("=" * 50)
    print("🔗 SEQUENTIAL CHAINS DEMO (Day 14)")
    print("=" * 50)
    
    workflows = SequentialChainWorkflows()
    
    # Example 1: Basic chain
    print("\n1️⃣ Basic 2-Step Chain:")
    result1 = workflows.basic_chain("smart water bottle")
    print(f"Result: {result1}")
    
    # Example 2: Content creation
    print("\n2️⃣ Content Creation Workflow:")
    result2 = workflows.content_creation_chain(
        topic="Artificial Intelligence Ethics",
        tone="educational"
    )
    print(f"Outline: {result2['outline'][:100]}...")
    print(f"Content: {result2['content'][:100]}...")
    
    # Example 3: E-commerce
    print("\n3️⃣ E-commerce Product Launch:")
    result3 = workflows.ecommerce_workflow(
        product="wireless headphones",
        target="music producers"
    )
    print(f"Description: {result3['description'][:100]}...")

if __name__ == "__main__":
    run_examples()