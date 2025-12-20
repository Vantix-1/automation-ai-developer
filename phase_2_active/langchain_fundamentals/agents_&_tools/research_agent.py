"""
🔍 Research Agent with Web Search
Day 19: Building a research assistant with internet access
"""

import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import BaseTool, Tool
from langchain import hub
from langchain.callbacks import get_openai_callback

# Optional: Install duckduckgo-search for web search
# pip install duckduckgo-search

load_dotenv()

class WebSearchTool(BaseTool):
    """Tool for searching the web"""
    name: str = "WebSearch"
    description: str = "Search the internet for current information. Input: search query as a string"
    
    def _run(self, query: str) -> str:
        """Search the web using DuckDuckGo"""
        try:
            from duckduckgo_search import DDGS
            
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=5))
                
                if not results:
                    return "No results found"
                
                # Format results
                formatted = []
                for i, result in enumerate(results, 1):
                    formatted.append(f"""
                    Result {i}:
                    Title: {result.get('title', 'No title')}
                    Snippet: {result.get('body', 'No snippet')}
                    URL: {result.get('href', 'No URL')}
                    """)
                
                return "\n".join(formatted)
                
        except ImportError:
            # Fallback to mock data if duckduckgo-search not installed
            return f"""Mock search results for '{query}':
            
            Result 1:
            Title: Understanding {query}
            Snippet: {query} is an important concept in modern technology...
            URL: https://example.com/{query.replace(' ', '_')}
            
            Result 2:
            Title: Latest developments in {query}
            Snippet: Recent advances in {query} have shown promising results...
            URL: https://example.com/news/{query.replace(' ', '-')}
            """
        except Exception as e:
            return f"Search error: {e}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class WikipediaTool(BaseTool):
    """Tool for Wikipedia searches"""
    name: str = "Wikipedia"
    description: str = "Search Wikipedia for information. Input: search query as a string"
    
    def _run(self, query: str) -> str:
        """Search Wikipedia"""
        try:
            import wikipedia
            
            # Search for pages
            search_results = wikipedia.search(query, results=3)
            
            if not search_results:
                return f"No Wikipedia pages found for '{query}'"
            
            # Get content from first result
            try:
                page = wikipedia.page(search_results[0])
                summary = wikipedia.summary(search_results[0], sentences=3)
                
                return f"""
                Wikipedia: {page.title}
                Summary: {summary}
                
                Full content available at: {page.url}
                """
            except wikipedia.exceptions.DisambiguationError as e:
                return f"Disambiguation needed for '{query}'. Options: {e.options[:5]}"
            except wikipedia.exceptions.PageError:
                return f"Wikipedia page not found for '{query}'"
                
        except ImportError:
            return "Wikipedia package not installed. Install with: pip install wikipedia"
        except Exception as e:
            return f"Wikipedia error: {e}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class CitationTool(BaseTool):
    """Tool for managing citations and sources"""
    name: str = "CitationManager"
    description: str = "Manage citations and track sources. Input format: 'action' where action is 'list' or 'format'"
    
    # Pydantic v2 requires mutable default to be defined properly
    sources: List[Dict] = []
    
    def __init__(self, **data):
        super().__init__(**data)
        self.sources = []
    
    def _run(self, input_str: str) -> str:
        """Manage citations"""
        try:
            action = input_str.strip().lower()
            
            if action == "add":
                # For simplicity, add a generic source
                source = {"title": "Research Source", "url": "https://example.com", "date": "2024"}
                self.sources.append(source)
                return f"Added source: {source.get('title', 'Unknown')}"
                
            elif action == "list":
                if not self.sources:
                    return "No sources added yet"
                
                formatted = []
                for i, source in enumerate(self.sources, 1):
                    formatted.append(f"""
                    Source {i}:
                    Title: {source.get('title', 'Unknown')}
                    URL: {source.get('url', 'No URL')}
                    Date: {source.get('date', 'Unknown')}
                    """)
                
                return "\n".join(formatted)
                
            elif action.startswith("format"):
                parts = input_str.split('|')
                citation_style = parts[1].strip().upper() if len(parts) > 1 else 'APA'
                formatted_citations = []
                
                for source in self.sources:
                    if citation_style == 'APA':
                        formatted = f"Author. ({source.get('date', 'n.d.')}). {source.get('title', 'Unknown')}. Retrieved from {source.get('url', '')}"
                    elif citation_style == 'MLA':
                        formatted = f"Author. \"{source.get('title', 'Unknown')}\". {source.get('date', 'n.d.')}. Web."
                    else:
                        formatted = str(source)
                    
                    formatted_citations.append(formatted)
                
                return f"{citation_style} Citations:\n" + "\n".join(formatted_citations)
                
            else:
                return f"Unknown action: {action}. Use: add, list, format|STYLE"
                
        except Exception as e:
            return f"Citation error: {e}"
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)

class ContentAnalyzerTool(BaseTool):
    """Tool for analyzing content quality"""
    name: str = "ContentAnalyzer"
    description: str = "Analyze content for quality, bias, and relevance. Input: content text as a string"
    
    def _run(self, content: str) -> str:
        """Analyze content"""
        # Simple analysis - can be enhanced with NLP models
        word_count = len(content.split())
        sentence_count = len(re.findall(r'[.!?]+', content))
        
        # Check for common bias indicators
        bias_indicators = ['always', 'never', 'everyone', 'nobody', 'best', 'worst']
        bias_score = sum(1 for word in bias_indicators if word.lower() in content.lower())
        
        # Check for citations/references
        has_citations = bool(re.search(r'\[\d+\]|\([^)]+\)|http[s]?://', content))
        
        analysis = f"""
        Content Analysis:
        - Word count: {word_count}
        - Sentence count: {sentence_count}
        - Bias indicators found: {bias_score}
        - Citations/references: {'Yes' if has_citations else 'No'}
        - Average sentence length: {word_count/max(sentence_count, 1):.1f} words
        
        Recommendations:
        {self._get_recommendations(word_count, bias_score, has_citations)}
        """
        
        return analysis
    
    def _get_recommendations(self, word_count: int, bias_score: int, has_citations: bool) -> str:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if word_count < 100:
            recommendations.append("Consider adding more detail and examples.")
        elif word_count > 1000:
            recommendations.append("Consider breaking content into sections.")
        
        if bias_score > 3:
            recommendations.append("Consider using more neutral language.")
        
        if not has_citations:
            recommendations.append("Add citations to support your claims.")
        
        return "\n".join(recommendations) if recommendations else "Content looks good!"
    
    async def _arun(self, content: str) -> str:
        return self._run(content)

class ResearchAgent:
    """Complete research agent with web capabilities"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo-16k",  # Larger context for research
            temperature=0.3,
            max_tokens=2000
        )
        self.citation_tool = CitationTool()
    
    def create_research_tools(self) -> List[BaseTool]:
        """Create tools for research agent"""
        
        # Core research tools
        tools = [
            WebSearchTool(),
            WikipediaTool(),
            self.citation_tool,
            ContentAnalyzerTool()
        ]
        
        return tools
    
    def create_research_agent(self) -> AgentExecutor:
        """Create the research agent"""
        tools = self.create_research_tools()
        
        # Use standard ReAct prompt
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=hub.pull("hwchase17/react")
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True
        )
    
    def conduct_research(self, topic: str, depth: str = "overview") -> Dict[str, Any]:
        """Conduct research on a topic"""
        print(f"\n🔍 Researching: {topic}")
        print(f"📊 Depth: {depth}")
        
        agent = self.create_research_agent()
        
        # Track cost
        with get_openai_callback() as cb:
            result = agent.invoke({
                "input": f"Research {topic}. Provide a comprehensive {depth} with citations."
            })
            
            print(f"\n💰 Research Cost:")
            print(f"  Tokens: {cb.total_tokens}")
            print(f"  Cost: ${cb.total_cost:.4f}")
        
        return result
    
    def demo_research_capabilities(self):
        """Demonstrate research agent capabilities"""
        print("=" * 50)
        print("🔬 RESEARCH AGENT DEMO (Day 19)")
        print("=" * 50)
        
        research_topics = [
            {
                "topic": "artificial intelligence ethics",
                "depth": "comprehensive overview",
                "description": "Current state of AI ethics and key debates"
            },
            {
                "topic": "Python programming best practices",
                "depth": "practical overview",
                "description": "Best practices for Python development"
            }
        ]
        
        for i, research in enumerate(research_topics[:1], 1):  # Just do first one for demo
            print(f"\n{'='*40}")
            print(f"Research {i}: {research['topic']}")
            print(f"{'='*40}")
            
            try:
                result = self.conduct_research(
                    topic=research['topic'],
                    depth=research['depth']
                )
                
                print(f"\n📄 Research Summary:")
                print(result['output'][:1000] + "...")
                
                # Show citations if available
                print(f"\n📚 Sources:")
                citation_list = self.citation_tool._run("list")
                print(citation_list[:500])
                
            except Exception as e:
                print(f"❌ Research error: {e}")
        
        print("\n" + "=" * 50)
        print("✅ Research Agent Demo Complete!")

class ResearchWorkflow:
    """Complete research workflow with multiple stages"""
    
    def __init__(self):
        self.agent = ResearchAgent()
    
    def execute_workflow(self, research_topic: str) -> Dict[str, Any]:
        """Execute complete research workflow"""
        print(f"\n📋 Research Workflow: {research_topic}")
        print("-" * 40)
        
        workflow_steps = [
            ("🔍 Initial Exploration", f"Find basic information about {research_topic}"),
            ("📚 Deep Dive", f"Research current developments in {research_topic}"),
        ]
        
        results = {}
        
        for step_name, step_query in workflow_steps[:1]:  # Just first step for demo
            print(f"\n{step_name}")
            print(f"Query: {step_query}")
            
            try:
                result = self.agent.conduct_research(
                    topic=step_query,
                    depth="detailed"
                )
                results[step_name] = result['output']
                
                print(f"✅ Step completed")
                
            except Exception as e:
                print(f"❌ Step failed: {e}")
                results[step_name] = f"Error: {e}"
        
        return results

def interactive_research_session():
    """Interactive research session"""
    print("\n" + "=" * 50)
    print("💬 INTERACTIVE RESEARCH SESSION")
    print("=" * 50)
    
    agent = ResearchAgent()
    
    print("\n🤖 Hello! I'm your research assistant.")
    print("I can help you research any topic with web search, Wikipedia, and analysis.")
    print("Type 'quit' to exit, 'citations' to see sources.\n")
    
    session_count = 0
    max_sessions = 2  # Limit for demo
    
    while session_count < max_sessions:
        try:
            user_input = input("\n🎯 What would you like to research? ").strip()
            
            if user_input.lower() == 'quit' or not user_input:
                print("Goodbye! Happy researching!")
                break
            elif user_input.lower() == 'citations':
                citations = agent.citation_tool._run("list")
                print(f"\n📚 Current Citations:\n{citations}")
                continue
            
            print(f"\n🔍 Researching: {user_input}")
            
            # Conduct research
            result = agent.conduct_research(user_input, "comprehensive")
            
            print(f"\n📄 Research Results:")
            print(result['output'][:500] + "...")
            
            session_count += 1
            
        except KeyboardInterrupt:
            print("\n\nResearch session ended.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            break

def main():
    """Run research agent demonstrations"""
    print("🚀 Research Agent with Web Search")
    print("Day 19: Building an AI Research Assistant")
    
    # Demo 1: Research capabilities
    research_agent = ResearchAgent()
    research_agent.demo_research_capabilities()
    
    # Demo 2: Complete workflow
    print("\n" + "=" * 50)
    print("📋 COMPLETE RESEARCH WORKFLOW")
    print("=" * 50)
    
    workflow = ResearchWorkflow()
    workflow_results = workflow.execute_workflow("renewable energy")
    
    print(f"\n✅ Workflow completed with {len(workflow_results)} steps")
    
    print("\n" + "=" * 50)
    print("✅ Day 19 Complete!")
    print("Next: Persistent Memory (Day 20)")

if __name__ == "__main__":
    main()