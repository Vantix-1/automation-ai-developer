"""
LangChain Setup & Fundamentals - Days 11-13
Introduction to LangChain framework (Updated for LangChain 0.1+)
"""

import os
import sys
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from colorama import Fore, Style, init
import tiktoken

# LangChain imports - UPDATED FOR NEWER VERSIONS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# For memory - use langchain_community
try:
    from langchain_community.callbacks import get_openai_callback
except ImportError:
    from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv

# Initialize colorama
init(autoreset=True)

load_dotenv()

class LangChainIntroduction:
    """Introduction to LangChain framework"""
    
    def __init__(self):
        """Initialize LangChain with OpenAI"""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=api_key
        )
        
        # Token counter
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + "🔗 LangChain Fundamentals - Days 11-13")
        print(Fore.CYAN + "="*70)
        print(Fore.GREEN + "✅ LangChain initialized successfully!")
        print(Fore.YELLOW + f"🤖 Model: {self.llm.model_name}")
        print(Fore.YELLOW + f"🌡️ Temperature: {self.llm.temperature}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoder.encode(text))
    
    def basic_chat_example(self):
        """Basic chat example using LangChain"""
        print(Fore.CYAN + "\n" + "━" * 50)
        print(Fore.CYAN + "💬 Example 1: Basic Chat")
        print(Fore.CYAN + "━" * 50)
        
        # Create messages
        messages = [
            SystemMessage(content="You are a helpful coding assistant."),
            HumanMessage(content="Explain recursion in programming with a simple example.")
        ]
        
        print(Fore.YELLOW + "\n📤 Sending messages to LLM...")
        
        # Get response with token tracking
        with get_openai_callback() as cb:
            response = self.llm.invoke(messages)
            
            print(Fore.GREEN + f"✅ Response received!")
            print(Fore.WHITE + f"\n🤖 Assistant: {response.content}")
            print(Fore.CYAN + f"\n📊 Token Usage:")
            print(Fore.CYAN + f"   Prompt tokens: {cb.prompt_tokens}")
            print(Fore.CYAN + f"   Completion tokens: {cb.completion_tokens}")
            print(Fore.CYAN + f"   Total tokens: {cb.total_tokens}")
            print(Fore.CYAN + f"   Total cost: ${cb.total_cost:.6f}")
    
    def prompt_template_example(self):
        """Example using prompt templates"""
        print(Fore.CYAN + "\n" + "━" * 50)
        print(Fore.CYAN + "📝 Example 2: Prompt Templates")
        print(Fore.CYAN + "━" * 50)
        
        # Create a prompt template
        template = """You are a {role}. Answer the following question in the style of {style}.

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Format the prompt
        formatted_prompt = prompt.format_messages(
            role="philosopher",
            style="Socrates",
            question="What is the meaning of life?"
        )
        
        print(Fore.YELLOW + "\n📝 Formatted Prompt:")
        for msg in formatted_prompt:
            print(Fore.WHITE + f"  {msg.type}: {msg.content}")
        
        # Get response
        with get_openai_callback() as cb:
            response = self.llm.invoke(formatted_prompt)
            
            print(Fore.GREEN + f"\n✅ Response:")
            print(Fore.WHITE + f"🤖 {response.content}")
            print(Fore.CYAN + f"\n📊 Cost: ${cb.total_cost:.6f}")
    
    def chain_example(self):
        """Example using modern LCEL (LangChain Expression Language) chains"""
        print(Fore.CYAN + "\n" + "━" * 50)
        print(Fore.CYAN + "⛓️ Example 3: LCEL Chains")
        print(Fore.CYAN + "━" * 50)
        
        # Create a prompt template
        template = """You are a helpful assistant that explains complex concepts.
        
        Concept: {concept}
        
        Explain this concept in simple terms suitable for a {audience}.
        
        Explanation:"""
        
        prompt = PromptTemplate(
            input_variables=["concept", "audience"],
            template=template
        )
        
        # Create chain using LCEL (modern approach)
        chain = prompt | self.llm | StrOutputParser()
        
        print(Fore.YELLOW + "\n🔗 Running LCEL Chain...")
        
        # Run the chain
        with get_openai_callback() as cb:
            result = chain.invoke({
                "concept": "quantum entanglement",
                "audience": "10-year-old child"
            })
            
            print(Fore.GREEN + f"\n✅ Chain Result:")
            print(Fore.WHITE + result)
            print(Fore.CYAN + f"\n📊 Total cost: ${cb.total_cost:.6f}")
    
    def sequential_chain_example(self):
        """Example using sequential chain with LCEL"""
        print(Fore.CYAN + "\n" + "━" * 50)
        print(Fore.CYAN + "🔗 Example 4: Sequential Chains (LCEL)")
        print(Fore.CYAN + "━" * 50)
        
        # First chain: Generate a story idea
        story_prompt = PromptTemplate(
            input_variables=["genre"],
            template="Generate a creative story idea in the {genre} genre. Just give the idea, nothing else."
        )
        story_chain = story_prompt | self.llm | StrOutputParser()
        
        # Second chain: Expand the idea into a summary
        summary_prompt = PromptTemplate(
            input_variables=["story_idea"],
            template="Expand this story idea into a one-paragraph summary: {story_idea}"
        )
        summary_chain = summary_prompt | self.llm | StrOutputParser()
        
        # Third chain: Create character descriptions
        character_prompt = PromptTemplate(
            input_variables=["story_summary"],
            template="Based on this story summary, describe the main character in 2-3 sentences: {story_summary}"
        )
        character_chain = character_prompt | self.llm | StrOutputParser()
        
        print(Fore.YELLOW + "\n🔗 Running Sequential Chain...")
        
        # Run the chains sequentially
        with get_openai_callback() as cb:
            print(Fore.CYAN + "\n1️⃣ Generating story idea...")
            story_idea = story_chain.invoke({"genre": "science fiction"})
            print(Fore.WHITE + f"Story Idea: {story_idea}")
            
            print(Fore.CYAN + "\n2️⃣ Creating summary...")
            summary = summary_chain.invoke({"story_idea": story_idea})
            print(Fore.WHITE + f"Summary: {summary}")
            
            print(Fore.CYAN + "\n3️⃣ Describing character...")
            character = character_chain.invoke({"story_summary": summary})
            print(Fore.WHITE + f"Character: {character}")
            
            print(Fore.GREEN + f"\n✅ Sequential chain complete!")
            print(Fore.CYAN + f"\n📊 Total cost: ${cb.total_cost:.6f}")
    
    def memory_example(self):
        """Example using conversation memory with message history"""
        print(Fore.CYAN + "\n" + "━" * 50)
        print(Fore.CYAN + "🧠 Example 5: Conversation Memory")
        print(Fore.CYAN + "━" * 50)
        
        # Create a chat history manually
        chat_history = []
        
        # Create a prompt with message history
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that remembers the conversation."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{human_input}")
        ])
        
        # Create chain
        chain = prompt | self.llm | StrOutputParser()
        
        print(Fore.YELLOW + "\n💬 Starting conversation with memory...")
        
        # Conversation loop
        conversation = [
            "My name is Alex.",
            "What's my name?",
            "What's 2 + 2?",
            "What did I tell you my name was?"
        ]
        
        total_cost = 0
        
        for i, message in enumerate(conversation, 1):
            print(Fore.CYAN + f"\n💭 Turn {i}: {message}")
            
            with get_openai_callback() as cb:
                response = chain.invoke({
                    "chat_history": chat_history,
                    "human_input": message
                })
                total_cost += cb.total_cost
                
                # Add to history
                chat_history.append(HumanMessage(content=message))
                chat_history.append(AIMessage(content=response))
                
                print(Fore.WHITE + f"🤖 Assistant: {response}")
                print(Fore.YELLOW + f"   Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.6f}")
        
        # Show memory contents
        print(Fore.CYAN + "\n" + "━" * 50)
        print(Fore.CYAN + "📝 Memory Contents:")
        print(Fore.CYAN + "━" * 50)
        
        for msg in chat_history:
            role = msg.type.capitalize()
            print(Fore.WHITE + f"{role}: {msg.content}")
        
        print(Fore.CYAN + f"\n💰 Total conversation cost: ${total_cost:.6f}")
    
    def embeddings_example(self):
        """Example using embeddings"""
        print(Fore.CYAN + "\n" + "━" * 50)
        print(Fore.CYAN + "🔤 Example 6: Text Embeddings")
        print(Fore.CYAN + "━" * 50)
        
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with many layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information."
        ]
        
        print(Fore.YELLOW + "\n📚 Generating embeddings for texts...")
        
        # Generate embeddings
        with get_openai_callback() as cb:
            embeddings = self.embeddings.embed_documents(texts)
            
            print(Fore.GREEN + f"✅ Generated {len(embeddings)} embeddings")
            print(Fore.CYAN + f"\n📊 Embedding details:")
            print(Fore.CYAN + f"   First embedding dimension: {len(embeddings[0])}")
            print(Fore.CYAN + f"   Total tokens: {cb.total_tokens}")
            print(Fore.CYAN + f"   Cost: ${cb.total_cost:.6f}")
        
        # Show similarity between texts
        print(Fore.YELLOW + "\n🔍 Calculating similarities...")
        
        # Get query embedding
        query = "AI and neural networks"
        query_embedding = self.embeddings.embed_query(query)
        
        # Calculate cosine similarities
        from numpy import dot
        from numpy.linalg import norm
        
        print(Fore.WHITE + f"\nQuery: '{query}'")
        print(Fore.CYAN + "\nSimilarities to texts:")
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings), 1):
            # Cosine similarity
            similarity = dot(query_embedding, embedding) / (norm(query_embedding) * norm(embedding))
            print(Fore.WHITE + f"{i}. Similarity: {similarity:.3f} - {text[:60]}...")
    
    def cost_tracking_example(self):
        """Example of detailed cost tracking"""
        print(Fore.CYAN + "\n" + "━" * 50)
        print(Fore.CYAN + "💰 Example 7: Cost Tracking")
        print(Fore.CYAN + "━" * 50)
        
        print(Fore.YELLOW + "\n📊 Running multiple operations with cost tracking...")
        
        total_tokens = 0
        total_cost = 0
        
        # Operation 1: Simple chat
        print(Fore.CYAN + "\n1. Simple Chat Operation:")
        messages = [HumanMessage(content="What is Python?")]
        with get_openai_callback() as cb:
            self.llm.invoke(messages)
            print(Fore.WHITE + f"   Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.6f}")
            total_tokens += cb.total_tokens
            total_cost += cb.total_cost
        
        # Operation 2: Chain execution
        print(Fore.CYAN + "\n2. Chain Execution:")
        template = "Explain {topic} in simple terms."
        prompt = PromptTemplate(input_variables=["topic"], template=template)
        chain = prompt | self.llm | StrOutputParser()
        
        with get_openai_callback() as cb:
            chain.invoke({"topic": "blockchain technology"})
            print(Fore.WHITE + f"   Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.6f}")
            total_tokens += cb.total_tokens
            total_cost += cb.total_cost
        
        # Operation 3: Embeddings
        print(Fore.CYAN + "\n3. Embeddings Generation:")
        with get_openai_callback() as cb:
            self.embeddings.embed_documents(["Sample text for embedding"])
            print(Fore.WHITE + f"   Tokens: {cb.total_tokens}, Cost: ${cb.total_cost:.6f}")
            total_tokens += cb.total_tokens
            total_cost += cb.total_cost
        
        print(Fore.CYAN + "\n" + "━" * 50)
        print(Fore.GREEN + f"📈 TOTAL for all operations:")
        print(Fore.GREEN + f"   Total tokens: {total_tokens}")
        print(Fore.GREEN + f"   Total cost: ${total_cost:.6f}")
        print(Fore.CYAN + "━" * 50)
    
    def interactive_demo(self):
        """Interactive LangChain demonstration"""
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + "🎮 Interactive LangChain Demo")
        print(Fore.CYAN + "="*70)
        
        examples = {
            "1": ("Basic Chat", self.basic_chat_example),
            "2": ("Prompt Templates", self.prompt_template_example),
            "3": ("LCEL Chains", self.chain_example),
            "4": ("Sequential Chains", self.sequential_chain_example),
            "5": ("Conversation Memory", self.memory_example),
            "6": ("Text Embeddings", self.embeddings_example),
            "7": ("Cost Tracking", self.cost_tracking_example)
        }
        
        while True:
            print(Fore.YELLOW + "\n" + "━" * 50)
            print(Fore.YELLOW + "📚 Available Examples:")
            for key, (name, _) in examples.items():
                print(Fore.YELLOW + f"  {key}. {name}")
            print(Fore.YELLOW + "  Q. Quit")
            print(Fore.YELLOW + "━" * 50)
            
            choice = input(Fore.WHITE + "\nSelect example (1-7 or Q): ").strip().upper()
            
            if choice == "Q":
                print(Fore.YELLOW + "\n👋 Goodbye!")
                break
            
            if choice in examples:
                name, function = examples[choice]
                print(Fore.CYAN + f"\n🚀 Running: {name}")
                try:
                    function()
                except Exception as e:
                    print(Fore.RED + f"❌ Error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(Fore.RED + "❌ Invalid choice. Please select 1-7 or Q.")
            
            input(Fore.YELLOW + "\nPress Enter to continue...")

def main():
    """Main function"""
    try:
        # Initialize LangChain
        langchain_intro = LangChainIntroduction()
        
        # Run interactive demo
        langchain_intro.interactive_demo()
        
    except ValueError as e:
        print(Fore.RED + f"\n❌ Configuration Error: {e}")
        print(Fore.YELLOW + "Please ensure you have:")
        print(Fore.YELLOW + "1. A .env file with OPENAI_API_KEY")
        print(Fore.YELLOW + "2. Installed required packages: pip install -r requirements.txt")
    except ImportError as e:
        print(Fore.RED + f"\n❌ Import Error: {e}")
        print(Fore.YELLOW + "Try: pip install langchain langchain-openai langchain-community langchain-core")
    except Exception as e:
        print(Fore.RED + f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()