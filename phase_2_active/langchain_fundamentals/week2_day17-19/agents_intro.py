"""
🤖 Agents Introduction
Day 17: LangChain Agents and the ReAct Pattern
Compatible with langchain 1.1.3
"""

import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain import hub

load_dotenv()

class BasicAgent:
    """Introduction to LangChain agents"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=500
        )
        
    def create_calculator_tool(self) -> Tool:
        """Create a simple calculator tool"""
        def calculator(expression: str) -> str:
            """Evaluate mathematical expressions"""
            try:
                # Safe evaluation
                allowed_names = {
                    'abs': abs, 'min': min, 'max': max, 
                    'sum': sum, 'len': len, 'round': round
                }
                result = eval(expression, {"__builtins__": {}}, allowed_names)
                return str(result)
            except Exception as e:
                return f"Error: {e}"
        
        return Tool(
            name="Calculator",
            func=calculator,
            description="Useful for mathematical calculations. Input should be a valid mathematical expression."
        )
    
    def create_text_tool(self) -> Tool:
        """Create text processing tools"""
        def text_processor(input_str: str) -> str:
            """Process text with various operations"""
            try:
                if '|' in input_str:
                    operation, text = input_str.split('|', 1)
                else:
                    return "Error: Use format 'operation|text'"
                
                if operation == "count":
                    words = len(text.split())
                    return f"Word count: {words}"
                elif operation == "summarize":
                    sentences = text.split('.')
                    summary = '. '.join(sentences[:2]) + '.'
                    return f"Summary: {summary}"
                else:
                    return f"Unknown operation: {operation}"
            except Exception as e:
                return f"Error: {e}"
        
        return Tool(
            name="TextProcessor",
            func=text_processor,
            description="Process text. Input format: 'operation|text'. Operations: count, summarize"
        )
    
    def create_weather_tool(self) -> Tool:
        """Mock weather tool (replace with real API)"""
        def get_weather(city: str) -> str:
            """Get weather information for a city"""
            # Mock data - in real implementation, call a weather API
            weather_data = {
                "new york": "Sunny, 72°F",
                "london": "Cloudy, 55°F", 
                "tokyo": "Rainy, 65°F",
                "sydney": "Clear, 80°F"
            }
            
            city_lower = city.lower()
            if city_lower in weather_data:
                return f"Weather in {city.title()}: {weather_data[city_lower]}"
            else:
                return f"Weather data not available for {city}"
        
        return Tool(
            name="Weather",
            func=get_weather,
            description="Get weather information for a city. Input: city name"
        )
    
    def create_basic_agent(self) -> AgentExecutor:
        """Create a basic agent with tools"""
        tools = [
            self.create_calculator_tool(),
            self.create_text_tool(),
            self.create_weather_tool()
        ]
        
        # Get the ReAct prompt from hub
        prompt = hub.pull("hwchase17/react")
        
        # Create the agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        # Create the executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        
        return agent_executor
    
    def demo_basic_agent(self):
        """Demonstrate basic agent capabilities"""
        print("=" * 50)
        print("🤖 BASIC AGENT DEMO (Day 17)")
        print("=" * 50)
        
        agent = self.create_basic_agent()
        
        test_queries = [
            "What is 25 * 4 + 100?",
            "What's the weather like in London?",
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            try:
                result = agent.invoke({"input": query})
                print(f"🤖 Response: {result['output']}")
            except Exception as e:
                print(f"❌ Error: {e}")

class ReActAgent:
    """Implementation of the ReAct (Reasoning + Acting) pattern"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1000
        )
        
    def create_react_tools(self) -> List[Tool]:
        """Create tools for ReAct agent"""
        
        def search_knowledge_base(query: str) -> str:
            """Search a mock knowledge base"""
            knowledge = {
                "python": "Python is a high-level programming language known for its readability.",
                "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
                "langchain": "LangChain is a framework for developing applications powered by language models.",
                "openai": "OpenAI is an AI research company that created ChatGPT and GPT models."
            }
            
            query_lower = query.lower()
            for key in knowledge:
                if key in query_lower:
                    return knowledge[key]
            
            return f"No information found for: {query}"
        
        def analyze_sentiment(text: str) -> str:
            """Simple sentiment analysis"""
            positive_words = ["good", "great", "excellent", "happy", "love"]
            negative_words = ["bad", "terrible", "awful", "sad", "hate"]
            
            text_lower = text.lower()
            positive = sum(1 for word in positive_words if word in text_lower)
            negative = sum(1 for word in negative_words if word in text_lower)
            
            if positive > negative:
                return "Positive sentiment"
            elif negative > positive:
                return "Negative sentiment"
            else:
                return "Neutral sentiment"
        
        def format_data(input_str: str) -> str:
            """Format data in different ways"""
            try:
                if '|' in input_str:
                    data_type, content = input_str.split('|', 1)
                else:
                    return "Error: Use format 'format_type|content'"
                
                if data_type == "list":
                    items = content.split(',')
                    return "List format:\n" + "\n".join([f"- {item.strip()}" for item in items])
                elif data_type == "table":
                    items = content.split(',')
                    return "Table format:\n| Item |\n|------|\n" + "\n".join([f"| {item.strip()} |" for item in items])
                else:
                    return f"Format: {data_type}\n{content}"
            except Exception as e:
                return f"Error: {e}"
        
        return [
            Tool(
                name="KnowledgeSearch",
                func=search_knowledge_base,
                description="Search for information in the knowledge base. Input: search query"
            ),
            Tool(
                name="SentimentAnalyzer",
                func=analyze_sentiment,
                description="Analyze sentiment of text. Input: text to analyze"
            ),
            Tool(
                name="DataFormatter",
                func=format_data,
                description="Format data. Input: 'format_type|content'. Formats: list, table"
            )
        ]
    
    def demo_react_pattern(self):
        """Demonstrate ReAct pattern"""
        print("\n" + "=" * 50)
        print("🧠 REACT PATTERN DEMO")
        print("=" * 50)
        
        tools = self.create_react_tools()
        prompt = hub.pull("hwchase17/react")
        
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
        
        # Complex query that requires reasoning
        complex_query = """
        I need to understand what LangChain is. Also, analyze the sentiment 
        of this text: "I love using Python for AI projects!"
        """
        
        print(f"\n🔍 Complex Query: {complex_query}")
        
        try:
            result = agent_executor.invoke({"input": complex_query})
            print(f"\n🤖 Final Response:\n{result['output']}")
                
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Run all agent demonstrations"""
    print("🚀 LangChain Agents Introduction")
    print("Day 17: Understanding Agents and the ReAct Pattern")
    print("(Compatible with langchain 1.1.3)")
    
    # Demo basic agent
    print("\nMake sure you have OPENAI_API_KEY in your .env file!")
    basic_agent = BasicAgent()
    basic_agent.demo_basic_agent()
    
    # Demo ReAct pattern
    react_agent = ReActAgent()
    react_agent.demo_react_pattern()
    
    print("\n" + "=" * 50)
    print("✅ Day 17 Complete!")
    print("Next: Custom Tools (Day 18)")

if __name__ == "__main__":
    main()