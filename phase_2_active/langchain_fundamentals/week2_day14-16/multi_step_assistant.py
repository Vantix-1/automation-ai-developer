"""
🤖 Multi-Step AI Assistant
Day 15-16: Production assistant with memory and reasoning
"""

import sys
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

class MultiStepAssistant:
    """Production-ready multi-step reasoning assistant"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.setup_chains()
    
    def setup_chains(self):
        """Setup reasoning chains"""
        
        # Understanding chain
        self.understand_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["query", "history"],
                template="""Analyze this query and extract key information:

Query: {query}
History: {history}

Extract:
1. Main topic
2. User intent
3. Required information
4. Any constraints"""
            ),
            output_key="analysis"
        )
        
        # Research chain
        self.research_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["analysis"],
                template="""Based on this analysis: {analysis}

Gather relevant information and key points."""
            ),
            output_key="research"
        )
        
        # Synthesis chain
        self.synthesize_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["analysis", "research"],
                template="""Combine analysis and research into coherent response:

Analysis: {analysis}
Research: {research}

Create structured, helpful response."""
            ),
            output_key="response"
        )
        
        # Validation chain
        self.validate_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["response", "query"],
                template="""Validate this response for the original query:

Original query: {query}
Response: {response}

Check for accuracy, completeness, and relevance."""
            ),
            output_key="validation"
        )
    
    def process_query(self, query: str) -> Dict:
        """Process query through multi-step reasoning"""
        
        print(f"\n🔍 Processing: {query}")
        
        # Get conversation history
        history = self.memory.load_memory_variables({})
        history_str = str(history.get("chat_history", ""))
        
        print("Step 1: Understanding...")
        analysis = self.understand_chain.run(query=query, history=history_str)
        
        print("Step 2: Researching...")
        research = self.research_chain.run(analysis=analysis)
        
        print("Step 3: Synthesizing...")
        response = self.synthesize_chain.run(analysis=analysis, research=research)
        
        print("Step 4: Validating...")
        validation = self.validate_chain.run(response=response, query=query)
        
        # Update memory
        self.memory.save_context(
            {"input": query},
            {"output": response}
        )
        
        return {
            "response": response,
            "analysis": analysis,
            "research": research,
            "validation": validation
        }
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("=" * 50)
        print("🤖 MULTI-STEP ASSISTANT")
        print("=" * 50)
        print("Type 'quit' to exit, 'clear' to clear memory\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    self.memory.clear()
                    print("Memory cleared!")
                    continue
                elif not user_input:
                    continue
                
                result = self.process_query(user_input)
                print(f"\nAssistant: {result['response']}")
                
                # Show reasoning (optional)
                if "--debug" in sys.argv:
                    print(f"\n[Debug] Validation: {result['validation'][:100]}...")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

def example_workflow():
    """Run example workflow"""
    assistant = MultiStepAssistant()
    
    print("=" * 50)
    print("📋 EXAMPLE WORKFLOW: Planning a Tech Talk")
    print("=" * 50)
    
    queries = [
        "I need to give a 30-minute talk about AI ethics",
        "What are the key topics I should cover?",
        "How should I structure the presentation?",
        "What are some engaging examples I could use?"
    ]
    
    for query in queries:
        print(f"\nYou: {query}")
        result = assistant.process_query(query)
        print(f"Assistant: {result['response'][:150]}...")
    
    print("\n✅ Workflow complete!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "example":
        example_workflow()
    else:
        assistant = MultiStepAssistant()
        assistant.chat_loop()