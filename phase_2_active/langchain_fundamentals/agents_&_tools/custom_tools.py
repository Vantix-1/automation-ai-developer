"""
🔧 Custom Tools for LangChain Agents
Day 18: Building specialized tools for AI agents
"""

import os
import json
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

load_dotenv()

class DatabaseTool(BaseTool):
    """Custom tool for database operations"""
    name: str = "DatabaseQuery"
    description: str = "Execute SQL queries on a database"
    
    def _run(self, query: str) -> str:
        """Execute SQL query (mock implementation)"""
        # In production, connect to actual database
        mock_data = {
            "SELECT * FROM users LIMIT 3": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
                {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
            ],
            "SELECT COUNT(*) FROM users": [{"count": 100}],
            "SELECT * FROM products WHERE price > 50": [
                {"id": 101, "name": "Laptop", "price": 999},
                {"id": 102, "name": "Monitor", "price": 299}
            ]
        }
        
        if query in mock_data:
            return json.dumps(mock_data[query], indent=2)
        else:
            return f"Query executed (mock): {query}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class FileSystemTool(BaseTool):
    """Custom tool for file system operations"""
    name: str = "FileSystem"
    description: str = "Read, write, and manage files. Input format: 'operation|path|content' where operation is read/write/list/delete"
    
    def _run(self, input_str: str) -> str:
        """Perform file system operations"""
        try:
            # Parse input: operation|path|content (content is optional)
            parts = input_str.split('|')
            if len(parts) < 2:
                return "Error: Use format 'operation|path|content'"
            
            operation = parts[0].strip()
            path = parts[1].strip()
            content = parts[2].strip() if len(parts) > 2 else None
            
            if operation == "read":
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        return f.read()
                else:
                    return f"File not found: {path}"
                    
            elif operation == "write":
                with open(path, 'w') as f:
                    f.write(content or "")
                return f"File written: {path}"
                
            elif operation == "list":
                if os.path.exists(path):
                    items = os.listdir(path)
                    return f"Contents of {path}:\n" + "\n".join(items)
                else:
                    return f"Directory not found: {path}"
                    
            elif operation == "delete":
                if os.path.exists(path):
                    os.remove(path)
                    return f"File deleted: {path}"
                else:
                    return f"File not found: {path}"
                    
            else:
                return f"Unknown operation: {operation}. Use: read, write, list, delete"
                
        except Exception as e:
            return f"Error: {e}"
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)

class WebAPITool(BaseTool):
    """Custom tool for calling web APIs"""
    name: str = "WebAPI"
    description: str = "Call RESTful APIs and process responses. Input format: 'method|url' where method is GET/POST/PUT/DELETE"
    
    def _run(self, input_str: str) -> str:
        """Make HTTP requests"""
        try:
            # Parse input: method|url
            parts = input_str.split('|')
            if len(parts) < 2:
                return "Error: Use format 'method|url'"
            
            method = parts[0].strip()
            url = parts[1].strip()
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "LangChain Agent"
            }
            
            if method.upper() == "GET":
                response = requests.get(url, headers=headers, timeout=10)
            elif method.upper() == "POST":
                response = requests.post(url, headers=headers, timeout=10)
            elif method.upper() == "PUT":
                response = requests.put(url, headers=headers, timeout=10)
            elif method.upper() == "DELETE":
                response = requests.delete(url, headers=headers, timeout=10)
            else:
                return f"Unsupported method: {method}"
            
            if response.status_code == 200:
                return f"API Response ({response.status_code}):\n{response.text[:500]}..."
            else:
                return f"API Error ({response.status_code}): {response.text}"
                
        except Exception as e:
            return f"API Call Error: {e}"
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)

# Using the @tool decorator for simpler tools
@tool
def time_converter(input_str: str) -> str:
    """Convert time between timezones. Input format: 'time|from_tz|to_tz' e.g. '09:00|EST|PST'"""
    try:
        parts = input_str.split('|')
        if len(parts) < 3:
            return "Error: Use format 'time|from_tz|to_tz'"
        
        time_str = parts[0].strip()
        from_tz = parts[1].strip()
        to_tz = parts[2].strip()
        
        # Mock conversions - in production, use pytz or similar library
        mock_conversions = {
            ("09:00", "EST", "PST"): "06:00",
            ("15:00", "UTC", "EST"): "10:00",
            ("18:00", "GMT", "CET"): "19:00"
        }
        
        key = (time_str, from_tz.upper(), to_tz.upper())
        if key in mock_conversions:
            return f"{time_str} {from_tz} = {mock_conversions[key]} {to_tz}"
        else:
            return f"Converted {time_str} from {from_tz} to {to_tz} (mock)"
    except Exception as e:
        return f"Error: {e}"

@tool
def data_validator(input_str: str) -> str:
    """Validate different types of data. Input format: 'data_type|value' e.g. 'email|test@example.com'"""
    try:
        parts = input_str.split('|')
        if len(parts) < 2:
            return "Error: Use format 'data_type|value'"
        
        data_type = parts[0].strip()
        value = parts[1].strip()
        
        if data_type == "email":
            import re
            pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if re.match(pattern, value):
                return f"Valid email: {value}"
            else:
                return f"Invalid email: {value}"
                
        elif data_type == "phone":
            # Simple phone validation
            digits = ''.join(filter(str.isdigit, value))
            if len(digits) >= 10:
                return f"Valid phone number: {value}"
            else:
                return f"Invalid phone number: {value}"
                
        elif data_type == "date":
            from datetime import datetime
            datetime.strptime(value, "%Y-%m-%d")
            return f"Valid date: {value}"
            
        elif data_type == "number":
            float(value)
            return f"Valid number: {value}"
            
        else:
            return f"Unknown data type: {data_type}"
            
    except Exception as e:
        return f"Validation error: {e}"

class CustomToolAgent:
    """Agent with custom tools"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=1000
        )
    
    def create_custom_tools(self) -> List[BaseTool]:
        """Create a collection of custom tools"""
        
        # Class-based tools
        db_tool = DatabaseTool()
        fs_tool = FileSystemTool()
        api_tool = WebAPITool()
        
        # Decorator-based tools
        time_tool = time_converter
        validation_tool = data_validator
        
        return [db_tool, fs_tool, api_tool, time_tool, validation_tool]
    
    def demo_custom_tools(self):
        """Demonstrate custom tools in action"""
        print("=" * 50)
        print("🔧 CUSTOM TOOLS DEMO (Day 18)")
        print("=" * 50)
        
        tools = self.create_custom_tools()
        
        # Create agent with custom tools
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
            max_iterations=8
        )
        
        # Test scenarios
        test_scenarios = [
            {
                "name": "Database Query",
                "query": "Query the database for all users using 'SELECT * FROM users LIMIT 3'"
            },
            {
                "name": "API Integration",
                "query": "Call the JSONPlaceholder API to get sample posts. Use GET method with https://jsonplaceholder.typicode.com/posts/1"
            },
            {
                "name": "Data Validation",
                "query": "Validate this email address: test@example.com"
            }
        ]
        
        for scenario in test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"🔍 Query: {scenario['query']}")
            
            try:
                result = agent_executor.invoke({"input": scenario['query']})
                print(f"\n🤖 Response:\n{result['output'][:500]}...")
                
                # Show tool usage
                steps = result.get('intermediate_steps', [])
                if steps:
                    print(f"\n🛠️ Tools Used: {len(steps)} steps")
                    for i, step in enumerate(steps, 1):
                        action, _ = step
                        print(f"  Step {i}: {action.tool} - {str(action.tool_input)[:50]}...")
                        
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n" + "=" * 50)
        print("✅ Custom Tools Demo Complete!")
        print("Each tool can be extended with real implementations.")

class ToolBuildingWorkshop:
    """Interactive workshop for building custom tools"""
    
    def __init__(self):
        self.tool_templates = {
            "calculator": self._create_calculator_template,
            "validator": self._create_validator_template,
            "api_client": self._create_api_client_template,
        }
    
    def _create_calculator_template(self, name: str) -> str:
        """Generate calculator tool template"""
        return f'''
class {name}Tool(BaseTool):
    """Custom calculator tool"""
    name: str = "{name}"
    description: str = "Perform calculations: add, multiply, divide, subtract"
    
    def _run(self, input_str: str) -> str:
        """Execute calculation. Format: 'operation|num1|num2'"""
        try:
            parts = input_str.split('|')
            operation = parts[0].strip()
            args = [float(x) for x in parts[1:]]
            
            if operation == "add":
                return str(sum(args))
            elif operation == "multiply":
                result = 1
                for arg in args:
                    result *= arg
                return str(result)
            # Add more operations...
        except Exception as e:
            return f"Error: {{e}}"
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)
'''
    
    def _create_validator_template(self, name: str) -> str:
        """Generate validation tool template"""
        return f'''
class {name}Validator(BaseTool):
    """Validate data"""
    name: str = "{name}Validator"
    description: str = "Validate {name} data. Format: 'rule|data'"
    
    def _run(self, input_str: str) -> str:
        """Validate data"""
        try:
            parts = input_str.split('|')
            rule = parts[0].strip()
            data = parts[1].strip()
            
            if rule == "required":
                return "Valid" if data.strip() else "Invalid: Required field"
            # Add more validation rules...
        except Exception as e:
            return f"Validation error: {{e}}"
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)
'''
    
    def _create_api_client_template(self, name: str) -> str:
        """Generate API client template"""
        return f'''
class {name}APITool(BaseTool):
    """API client tool"""
    name: str = "{name}API"
    description: str = "Call {name} API. Format: 'endpoint|method'"
    
    def _run(self, input_str: str) -> str:
        """Call API"""
        try:
            parts = input_str.split('|')
            endpoint = parts[0].strip()
            method = parts[1].strip() if len(parts) > 1 else "GET"
            
            # Implement your API logic here
            return f"Called {{endpoint}} with {{method}}"
        except Exception as e:
            return f"API Error: {{e}}"
    
    async def _arun(self, input_str: str) -> str:
        return self._run(input_str)
'''
    
    def workshop(self):
        """Interactive tool building workshop"""
        print("\n" + "=" * 50)
        print("🏗️ TOOL BUILDING WORKSHOP")
        print("=" * 50)
        
        print("\n📚 Available Tool Templates:")
        for i, template in enumerate(self.tool_templates.keys(), 1):
            print(f"  {i}. {template}")
        
        print("\n🎯 Tool Template Examples:")
        print("\nCalculator Tool:")
        print(self._create_calculator_template("MyCalculator"))
        
        print("\n💡 Remember to add type annotations (: str, : int, etc.) to all class attributes!")

def main():
    """Run custom tools demonstrations"""
    print("🚀 Custom Tools for LangChain Agents")
    print("Day 18: Building Specialized Tools")
    
    # Demo: Custom tools in action
    custom_agent = CustomToolAgent()
    custom_agent.demo_custom_tools()
    
    # Demo: Tool building workshop
    workshop = ToolBuildingWorkshop()
    workshop.workshop()
    
    print("\n" + "=" * 50)
    print("✅ Day 18 Complete!")
    print("Next: Research Agent (Day 19)")

if __name__ == "__main__":
    main()