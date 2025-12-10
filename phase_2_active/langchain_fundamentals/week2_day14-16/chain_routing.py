"""
🔄 Chain Routing & Workflow Orchestration
Day 16: Dynamic routing and smart workflow selection
"""

import json
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

load_dotenv()

class ChainRouter:
    """Intelligent chain routing system"""
    
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)  # Lower temp for routing
        self.chains = self._setup_chains()
    
    def _setup_chains(self) -> Dict[str, Any]:
        """Setup specialized chains"""
        
        chains = {}
        
        # QA Chain
        chains["qa"] = LLMChain(
            llm=ChatOpenAI(temperature=0.7),
            prompt=PromptTemplate(
                input_variables=["question"],
                template="Answer this question clearly: {question}"
            )
        )
        
        # Analysis Chain
        chains["analysis"] = LLMChain(
            llm=ChatOpenAI(temperature=0.5),
            prompt=PromptTemplate(
                input_variables=["text", "type"],
                template="Analyze this text for {type}:\n{text}"
            )
        )
        
        # Creative Chain
        chains["creative"] = LLMChain(
            llm=ChatOpenAI(temperature=0.9),
            prompt=PromptTemplate(
                input_variables=["prompt", "style"],
                template="Create in {style} style:\n{prompt}"
            )
        )
        
        # Code Chain
        chains["code"] = LLMChain(
            llm=ChatOpenAI(temperature=0.2),
            prompt=PromptTemplate(
                input_variables=["task", "language"],
                template="Write {language} code for: {task}"
            )
        )
        
        return chains
    
    def route_query(self, query: str) -> Dict:
        """Determine which chain to use"""
        
        routing_prompt = f"""Analyze this query and select the best processing chain:

Query: {query}

Available chains:
1. qa - Question answering
2. analysis - Text analysis
3. creative - Creative writing
4. code - Code generation

Return JSON: {{"chain": "chain_name", "reason": "explanation"}}"""
        
        response = self.llm.predict(routing_prompt)
        
        try:
            # Extract JSON
            json_str = response.split("```json")[-1].split("```")[0].strip()
            return json.loads(json_str)
        except:
            # Default to QA
            return {"chain": "qa", "reason": "Default fallback"}
    
    def execute_chain(self, query: str) -> str:
        """Route and execute query"""
        
        # Get routing decision
        routing = self.route_query(query)
        chain_name = routing["chain"]
        
        print(f"🔄 Routing to: {chain_name} ({routing['reason']})")
        
        # Execute appropriate chain
        if chain_name == "qa":
            return self.chains["qa"].run(question=query)
        elif chain_name == "analysis":
            return self.chains["analysis"].run(text=query, type="general analysis")
        elif chain_name == "creative":
            return self.chains["creative"].run(prompt=query, style="creative")
        elif chain_name == "code":
            return self.chains["code"].run(task=query, language="Python")
        else:
            return f"Unknown chain: {chain_name}"

class WorkflowOrchestrator:
    """Orchestrate complex workflows"""
    
    def __init__(self):
        self.router = ChainRouter()
        self.workflows = self._setup_workflows()
    
    def _setup_workflows(self) -> Dict[str, Any]:
        """Define workflows"""
        
        workflows = {}
        
        # Document Processing Workflow
        workflows["document"] = {
            "steps": ["analysis", "qa"],
            "description": "Process documents: analyze then answer questions"
        }
        
        # Content Creation Workflow
        workflows["content"] = {
            "steps": ["analysis", "creative"],
            "description": "Create content: research then write"
        }
        
        # Coding Workflow
        workflows["coding"] = {
            "steps": ["analysis", "code"],
            "description": "Code generation: analyze requirements then write code"
        }
        
        return workflows
    
    def execute_workflow(self, workflow_name: str, input_data: str) -> Dict:
        """Execute complete workflow"""
        
        if workflow_name not in self.workflows:
            return {"error": f"Unknown workflow: {workflow_name}"}
        
        workflow = self.workflows[workflow_name]
        results = {}
        
        print(f"🚀 Executing {workflow_name} workflow...")
        print(f"Description: {workflow['description']}")
        
        # Execute each step
        for i, step in enumerate(workflow["steps"], 1):
            print(f"\nStep {i}: {step}")
            
            # For first step, use input data
            # For subsequent steps, use previous result
            if i == 1:
                result = self.router.chains[step].run(input_data)
            else:
                result = self.router.chains[step].run(previous_step)
            
            results[f"step_{i}_{step}"] = result
            previous_step = result
        
        return results

def demo_routing():
    """Demonstrate chain routing"""
    print("=" * 50)
    print("🔄 CHAIN ROUTING DEMO (Day 16)")
    print("=" * 50)
    
    router = ChainRouter()
    orchestrator = WorkflowOrchestrator()
    
    # Test queries for routing
    test_queries = [
        "What is machine learning?",
        "Analyze the sentiment of this text: I love this product!",
        "Write a poem about artificial intelligence",
        "Create a function to sort a list in Python"
    ]
    
    print("\n1️⃣ Smart Chain Routing:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = router.execute_chain(query)
        print(f"Result: {result[:100]}...")
    
    print("\n2️⃣ Workflow Orchestration:")
    
    # Document workflow
    print("\n📄 Document Processing Workflow:")
    doc_result = orchestrator.execute_workflow("document", 
        "Machine learning algorithms learn from data without explicit programming.")
    for key, value in doc_result.items():
        print(f"  {key}: {value[:80]}...")
    
    # Content workflow
    print("\n✍️ Content Creation Workflow:")
    content_result = orchestrator.execute_workflow("content",
        "Create content about climate change solutions")
    for key, value in content_result.items():
        print(f"  {key}: {value[:80]}...")

if __name__ == "__main__":
    demo_routing()