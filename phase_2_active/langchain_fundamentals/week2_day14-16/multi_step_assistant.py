"""
🤖 Multi-Step AI Assistant (Modern LCEL)
Day 15-16: Production assistant with memory and reasoning
"""

import sys
from typing import Dict, List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

class MultiStepAssistant:
    """Production-ready multi-step reasoning assistant"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=500
        )
        self.parser = StrOutputParser()
        
        # Simple in-memory chat history
        self.chat_history = ChatMessageHistory()
        
        self.setup_chains()
    
    def setup_chains(self):
        """Setup reasoning chains using modern LCEL"""
        
        # Step 1: Understanding chain
        self.understand_prompt = ChatPromptTemplate.from_template(
            """Analyze this query and extract key information:

Query: {query}
Previous conversation: {history}

Extract:
1. Main topic
2. User intent
3. Required information
4. Any constraints

Provide a clear analysis."""
        )
        
        self.understand_chain = (
            self.understand_prompt 
            | self.llm 
            | self.parser
        )
        
        # Step 2: Research chain
        self.research_prompt = ChatPromptTemplate.from_template(
            """Based on this analysis: {analysis}

Gather relevant information and key points needed to answer the query."""
        )
        
        self.research_chain = (
            self.research_prompt 
            | self.llm 
            | self.parser
        )
        
        # Step 3: Synthesis chain
        self.synthesize_prompt = ChatPromptTemplate.from_template(
            """Combine analysis and research into a coherent response:

Analysis: {analysis}
Research: {research}
Original Query: {query}

Create a structured, helpful response that directly answers the user's question."""
        )
        
        self.synthesize_chain = (
            self.synthesize_prompt 
            | self.llm 
            | self.parser
        )
        
        # Step 4: Validation chain
        self.validate_prompt = ChatPromptTemplate.from_template(
            """Validate this response for the original query:

Original query: {query}
Response: {response}

Check for accuracy, completeness, and relevance. Is this a good answer?"""
        )
        
        self.validate_chain = (
            self.validate_prompt 
            | self.llm 
            | self.parser
        )
    
    def get_history_string(self) -> str:
        """Get conversation history as string"""
        messages = self.chat_history.messages
        if not messages:
            return "No previous conversation"
        
        history_str = ""
        for msg in messages[-6:]:  # Last 6 messages (3 exchanges)
            role = "User" if msg.type == "human" else "Assistant"
            history_str += f"{role}: {msg.content}\n"
        return history_str
    
    def process_query(self, query: str) -> Dict:
        """Process query through multi-step reasoning"""
        
        print(f"\n🔍 Processing: {query}")
        
        # Get conversation history
        history_str = self.get_history_string()
        
        try:
            # Step 1: Understanding
            print("📊 Step 1: Understanding...")
            analysis = self.understand_chain.invoke({
                "query": query,
                "history": history_str
            })
            
            # Step 2: Research
            print("🔬 Step 2: Researching...")
            research = self.research_chain.invoke({
                "analysis": analysis
            })
            
            # Step 3: Synthesis
            print("⚙️ Step 3: Synthesizing...")
            response = self.synthesize_chain.invoke({
                "analysis": analysis,
                "research": research,
                "query": query
            })
            
            # Step 4: Validation
            print("✅ Step 4: Validating...")
            validation = self.validate_chain.invoke({
                "response": response,
                "query": query
            })
            
            # Update memory
            self.chat_history.add_user_message(query)
            self.chat_history.add_ai_message(response)
            
            return {
                "response": response,
                "analysis": analysis,
                "research": research,
                "validation": validation
            }
            
        except Exception as e:
            print(f"❌ Error in processing: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "analysis": "",
                "research": "",
                "validation": ""
            }
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("=" * 50)
        print("🤖 MULTI-STEP ASSISTANT")
        print("=" * 50)
        print("Commands:")
        print("  'quit' - Exit")
        print("  'clear' - Clear memory")
        print("  'history' - Show conversation history")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                    
                elif user_input.lower() == 'clear':
                    self.chat_history.clear()
                    print("🗑️ Memory cleared!")
                    continue
                    
                elif user_input.lower() == 'history':
                    print("\n📜 Conversation History:")
                    print(self.get_history_string())
                    continue
                    
                elif not user_input:
                    continue
                
                result = self.process_query(user_input)
                print(f"\n🤖 Assistant: {result['response']}")
                
                # Show reasoning in debug mode
                if "--debug" in sys.argv:
                    print(f"\n[Debug Analysis]: {result['analysis'][:100]}...")
                    print(f"[Debug Research]: {result['research'][:100]}...")
                    print(f"[Debug Validation]: {result['validation'][:100]}...")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def example_workflow():
    """Run example workflow"""
    assistant = MultiStepAssistant()
    
    print("=" * 60)
    print("📋 EXAMPLE WORKFLOW: Planning a Tech Talk on AI Ethics")
    print("=" * 60)
    
    queries = [
        "I need to give a 30-minute talk about AI ethics",
        "What are the key topics I should cover?",
        "How should I structure the presentation?",
        "What are some engaging examples I could use?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*60}")
        print(f"Query {i}: {query}")
        print('='*60)
        
        result = assistant.process_query(query)
        
        print(f"\n🤖 Response:")
        print(result['response'])
        print(f"\n{'─'*60}")
    
    print("\n" + "="*60)
    print("✅ Workflow complete!")
    print("="*60)

def quick_test():
    """Quick test of the assistant"""
    print("🧪 Running Quick Test...\n")
    
    assistant = MultiStepAssistant()
    
    # Single test query
    test_query = "Explain the concept of chain-of-thought reasoning in AI"
    result = assistant.process_query(test_query)
    
    print(f"\n📊 Test Results:")
    print(f"✅ Response generated: {len(result['response'])} chars")
    print(f"✅ Analysis completed: {len(result['analysis'])} chars")
    print(f"✅ Research completed: {len(result['research'])} chars")
    print(f"✅ Validation completed: {len(result['validation'])} chars")
    
    print(f"\n🤖 Assistant Response:")
    print(result['response'])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "example":
            example_workflow()
        elif sys.argv[1] == "test":
            quick_test()
        else:
            print("Usage:")
            print("  python multi_step_assistant.py           # Interactive chat")
            print("  python multi_step_assistant.py example   # Run example workflow")
            print("  python multi_step_assistant.py test      # Quick test")
            print("  python multi_step_assistant.py --debug   # Interactive with debug info")
    else:
        assistant = MultiStepAssistant()
        assistant.chat_loop()