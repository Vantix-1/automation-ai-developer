"""
🚀 Sequential Chains Implementation (Modern LCEL)
Day 14: Multi-step AI workflows with LangChain 1.x
"""

import os
from typing import Dict, List
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment
load_dotenv()

class SequentialChainWorkflows:
    """Complete sequential chain implementations using modern LCEL"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
        self.parser = StrOutputParser()
    
    def basic_chain(self, product_type: str):
        """Simple 2-step chain: Name → Slogan"""
        print(f"📦 Creating marketing for: {product_type}")
        
        # Step 1: Name generation
        name_prompt = PromptTemplate.from_template(
            "Generate 3 creative names for a {product}. List them clearly."
        )
        
        # Step 2: Slogan generation
        slogan_prompt = PromptTemplate.from_template(
            "Create marketing slogans for these product names:\n{names}"
        )
        
        # Chain them together using LCEL
        chain = (
            {"product": RunnablePassthrough()} 
            | name_prompt 
            | self.llm 
            | self.parser
            | (lambda names: {"names": names})
            | slogan_prompt
            | self.llm
            | self.parser
        )
        
        return chain.invoke(product_type)
    
    def content_creation_chain(self, topic: str, tone: str = "professional"):
        """Complete content creation pipeline"""
        print(f"📝 Creating {tone} content about: {topic}")
        
        # Step 1: Outline
        outline_prompt = ChatPromptTemplate.from_template(
            "Create a detailed outline about {topic} in a {tone} tone."
        )
        
        # Step 2: Content writing
        content_prompt = ChatPromptTemplate.from_template(
            "Write full content based on this outline:\n{outline}\n\nTone: {tone}"
        )
        
        # Step 3: SEO optimization
        seo_prompt = ChatPromptTemplate.from_template(
            "Add SEO keywords and optimize this content:\n{content}\n\nPrimary topic: {topic}"
        )
        
        # Build the chain
        chain = (
            {
                "topic": lambda x: x["topic"],
                "tone": lambda x: x["tone"]
            }
            | RunnablePassthrough.assign(
                outline=lambda x: (
                    outline_prompt 
                    | self.llm 
                    | self.parser
                ).invoke(x)
            )
            | RunnablePassthrough.assign(
                content=lambda x: (
                    content_prompt 
                    | self.llm 
                    | self.parser
                ).invoke(x)
            )
            | RunnablePassthrough.assign(
                optimized_content=lambda x: (
                    seo_prompt 
                    | self.llm 
                    | self.parser
                ).invoke(x)
            )
        )
        
        result = chain.invoke({"topic": topic, "tone": tone})
        return result
    
    def ecommerce_workflow(self, product: str, target: str):
        """E-commerce product launch workflow"""
        print(f"🛍️ Launching {product} for {target}")
        
        # Step 1: Product description
        desc_prompt = ChatPromptTemplate.from_template(
            "Write a compelling product description for {product} targeting {target}."
        )
        
        # Step 2: Audience analysis
        audience_prompt = ChatPromptTemplate.from_template(
            "Analyze the target audience for {product}: {target}. Include demographics and pain points."
        )
        
        # Step 3: Marketing strategy
        strategy_prompt = ChatPromptTemplate.from_template(
            "Create a marketing strategy based on:\n\nProduct: {description}\n\nAudience: {audience_analysis}"
        )
        
        # Build the complete workflow
        chain = (
            {
                "product": lambda x: x["product"],
                "target": lambda x: x["target"]
            }
            | RunnablePassthrough.assign(
                description=lambda x: (
                    desc_prompt 
                    | self.llm 
                    | self.parser
                ).invoke(x)
            )
            | RunnablePassthrough.assign(
                audience_analysis=lambda x: (
                    audience_prompt 
                    | self.llm 
                    | self.parser
                ).invoke(x)
            )
            | RunnablePassthrough.assign(
                strategy=lambda x: (
                    strategy_prompt 
                    | self.llm 
                    | self.parser
                ).invoke(x)
            )
        )
        
        result = chain.invoke({"product": product, "target": target})
        return result
    
    def simple_sequential_example(self, product: str):
        """Simplest possible sequential chain example"""
        print(f"\n🔗 Simple Sequential Chain for: {product}")
        
        # Create two simple prompts
        step1 = PromptTemplate.from_template("Suggest a creative name for: {product}")
        step2 = PromptTemplate.from_template("Write a tagline for this product name: {name}")
        
        # Chain them
        chain = (
            step1 
            | self.llm 
            | self.parser 
            | (lambda name: {"name": name}) 
            | step2 
            | self.llm 
            | self.parser
        )
        
        result = chain.invoke({"product": product})
        print(f"✅ Final tagline: {result}")
        return result

def run_examples():
    """Run all sequential chain examples"""
    print("=" * 50)
    print("🔗 SEQUENTIAL CHAINS DEMO (Day 14)")
    print("=" * 50)
    
    workflows = SequentialChainWorkflows()
    
    # Example 1: Basic chain
    print("\n1️⃣ Basic 2-Step Chain:")
    try:
        result1 = workflows.basic_chain("smart water bottle")
        print(f"✅ Result:\n{result1}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Example 1.5: Simple sequential (easiest to understand)
    print("\n1.5️⃣ Simplest Sequential Chain:")
    try:
        workflows.simple_sequential_example("eco-friendly smartphone case")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Example 2: Content creation
    print("\n2️⃣ Content Creation Workflow:")
    try:
        result2 = workflows.content_creation_chain(
            topic="Artificial Intelligence Ethics",
            tone="educational"
        )
        print(f"📋 Outline: {result2['outline'][:150]}...")
        print(f"📄 Content: {result2['content'][:150]}...")
        print(f"🎯 Optimized: {result2['optimized_content'][:150]}...\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Example 3: E-commerce
    print("\n3️⃣ E-commerce Product Launch:")
    try:
        result3 = workflows.ecommerce_workflow(
            product="wireless headphones",
            target="music producers"
        )
        print(f"📦 Description: {result3['description'][:150]}...")
        print(f"👥 Audience: {result3['audience_analysis'][:150]}...")
        print(f"📈 Strategy: {result3['strategy'][:150]}...\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    print("=" * 50)
    print("✅ Day 14 Complete!")
    print("=" * 50)

if __name__ == "__main__":
    run_examples()