"""
Prompt Pattern Library - Days 6-8
Collection of advanced prompt patterns and techniques
"""

import os
import json
from typing import Dict, List, Any
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class PromptPatternLibrary:
    """Library of advanced prompt patterns"""
    
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self.patterns = self._load_patterns()
    
    def _load_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load prompt patterns"""
        return {
            "chain_of_thought": {
                "name": "Chain of Thought",
                "description": "Break down complex problems step by step",
                "template": """Let's think through this step by step.

Problem: {problem}

First, let's understand what's being asked...
Then, we can break it down into smaller parts...
Finally, we'll put it all together.

Step-by-step solution:""",
                "example_problem": "If a train leaves Station A at 60 mph and another leaves Station B at 80 mph, 200 miles apart, when will they meet?",
                "best_for": ["Math problems", "Logical reasoning", "Complex analysis"]
            },
            
            "few_shot": {
                "name": "Few-Shot Learning",
                "description": "Provide examples to guide the model",
                "template": """Here are some examples:

Example 1:
Input: {example1_input}
Output: {example1_output}

Example 2:
Input: {example2_input}
Output: {example2_output}

Now for the new input:
Input: {new_input}
Output:""",
                "example_data": {
                    "example1_input": "Translate to French: Hello",
                    "example1_output": "Bonjour",
                    "example2_input": "Translate to French: Thank you",
                    "example2_output": "Merci",
                    "new_input": "Translate to French: Good morning"
                },
                "best_for": ["Translation", "Classification", "Format learning"]
            },
            
            "role_playing": {
                "name": "Role Playing",
                "description": "Assume a specific role or persona",
                "template": """You are {role}. You have the following characteristics:
- {characteristic1}
- {characteristic2}
- {characteristic3}

As {role}, respond to: {query}""",
                "example_data": {
                    "role": "Shakespeare",
                    "characteristic1": "You speak in Elizabethan English",
                    "characteristic2": "You use poetic language",
                    "characteristic3": "You reference classical themes",
                    "query": "What thinkest thou of modern technology?"
                },
                "best_for": ["Creative writing", "Character dialogue", "Perspective taking"]
            },
            
            "structured_output": {
                "name": "Structured Output",
                "description": "Request specific output format",
                "template": """Please provide your response in the following format:

{format_description}

For the following query: {query}

Response:""",
                "example_data": {
                    "format_description": """JSON format with these keys:
- "summary": string (brief summary)
- "key_points": array of strings (3-5 key points)
- "action_items": array of strings (recommended actions)
- "confidence": number (0-100)""",
                    "query": "Analyze this business proposal for expanding to European markets"
                },
                "best_for": ["Data extraction", "API responses", "Structured analysis"]
            },
            
            "socratic": {
                "name": "Socratic Method",
                "description": "Guide through questioning",
                "template": """Instead of giving a direct answer, I'll help you think through this by asking questions:

1. What do you understand about {topic} so far?
2. What have you tried already?
3. What assumptions are you making?
4. What would be a simpler version of this problem?
5. How would you know if you've found a good solution?

Based on your answers to these questions, what approach seems most promising?""",
                "example_topic": "optimizing a database query",
                "best_for": ["Learning", "Problem solving", "Critical thinking"]
            },
            
            "reverse": {
                "name": "Reverse Prompting",
                "description": "Work backwards from desired outcome",
                "template": """Let's work backwards from the desired outcome.

Desired outcome: {desired_outcome}

To achieve this, what would need to be true?
What steps would lead to that?
What resources would be needed?

Working backwards, the first step would be:""",
                "example_outcome": "Launch a successful mobile app with 10,000 users in 6 months",
                "best_for": ["Planning", "Goal setting", "Project management"]
            }
        }
    
    def list_patterns(self) -> List[Dict[str, Any]]:
        """List all available patterns"""
        return [
            {
                "id": pattern_id,
                "name": pattern["name"],
                "description": pattern["description"],
                "best_for": pattern["best_for"]
            }
            for pattern_id, pattern in self.patterns.items()
        ]
    
    def get_pattern(self, pattern_id: str) -> Dict[str, Any]:
        """Get a specific pattern"""
        return self.patterns.get(pattern_id, {})
    
    def test_pattern(self, pattern_id: str, **kwargs) -> str:
        """Test a pattern with custom inputs"""
        pattern = self.get_pattern(pattern_id)
        if not pattern:
            return f"Pattern '{pattern_id}' not found"
        
        # Fill template with provided values or examples
        template = pattern["template"]
        example_data = pattern.get("example_data", {})
        
        # Merge example data with provided kwargs
        all_data = {**example_data, **kwargs}
        
        # Fill template
        try:
            filled_template = template.format(**all_data)
        except KeyError as e:
            return f"Missing template variable: {e}. Provided: {list(all_data.keys())}"
        
        # Get response from OpenAI
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an AI that follows instructions precisely."},
                    {"role": "user", "content": filled_template}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error getting response: {e}"
    
    def create_custom_pattern(self, pattern_id: str, name: str, description: str, 
                             template: str, example_data: Dict[str, Any], best_for: List[str]):
        """Create a custom prompt pattern"""
        self.patterns[pattern_id] = {
            "name": name,
            "description": description,
            "template": template,
            "example_data": example_data,
            "best_for": best_for
        }
        return self.patterns[pattern_id]
    
    def save_patterns(self, filepath: str = "prompt_patterns.json"):
        """Save patterns to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.patterns, f, indent=2, ensure_ascii=False)
        print(f"✅ Patterns saved to {filepath}")
    
    def load_patterns(self, filepath: str = "prompt_patterns.json"):
        """Load patterns from JSON file"""
        if Path(filepath).exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                self.patterns = json.load(f)
            print(f"✅ Patterns loaded from {filepath}")
        else:
            print(f"❌ File not found: {filepath}")

def interactive_pattern_lab():
    """Interactive prompt pattern laboratory"""
    print("\n" + "="*70)
    print("🔬 Prompt Pattern Laboratory - Days 6-8")
    print("="*70)
    
    try:
        library = PromptPatternLibrary()
        
        while True:
            print("\n" + "="*50)
            print("📚 Pattern Library Options:")
            print("1. Browse Patterns")
            print("2. Test a Pattern")
            print("3. Create Custom Pattern")
            print("4. Save Patterns")
            print("5. Load Patterns")
            print("6. Exit")
            print("="*50)
            
            choice = input("\nSelect option (1-6): ").strip()
            
            if choice == "6":
                print("👋 Goodbye!")
                break
            
            if choice == "1":
                print("\n📖 Available Prompt Patterns:")
                print("-" * 50)
                
                patterns = library.list_patterns()
                for i, pattern in enumerate(patterns, 1):
                    print(f"\n{i}. {pattern['name']}")
                    print(f"   📝 {pattern['description']}")
                    print(f"   🎯 Best for: {', '.join(pattern['best_for'])}")
            
            elif choice == "2":
                patterns = library.list_patterns()
                print("\n🎯 Select a pattern to test:")
                
                for i, pattern in enumerate(patterns, 1):
                    print(f"{i}. {pattern['name']}")
                
                pattern_choice = input("\nEnter pattern number or name: ").strip()
                
                if pattern_choice.isdigit() and 1 <= int(pattern_choice) <= len(patterns):
                    pattern_id = list(library.patterns.keys())[int(pattern_choice) - 1]
                elif pattern_choice in library.patterns:
                    pattern_id = pattern_choice
                else:
                    print("❌ Invalid pattern selection")
                    continue
                
                pattern = library.get_pattern(pattern_id)
                print(f"\n🧪 Testing: {pattern['name']}")
                print(f"📝 Description: {pattern['description']}")
                
                # Show template variables
                print("\n📋 Template variables available:")
                if pattern.get('example_data'):
                    for key, value in pattern['example_data'].items():
                        print(f"  {key}: {value}")
                
                # Allow custom inputs
                print("\n💡 Enter custom values (press Enter to use defaults):")
                custom_data = {}
                for key in pattern.get('example_data', {}).keys():
                    value = input(f"{key}: ").strip()
                    if value:
                        custom_data[key] = value
                
                # Test the pattern
                print(f"\n🧪 Testing pattern...")
                result = library.test_pattern(pattern_id, **custom_data)
                
                print("\n" + "="*50)
                print("✅ TEST RESULTS")
                print("="*50)
                print(result)
                print("="*50)
            
            elif choice == "3":
                print("\n🛠️ Create Custom Pattern")
                
                pattern_id = input("Pattern ID (no spaces): ").strip()
                name = input("Pattern name: ").strip()
                description = input("Description: ").strip()
                
                print("\nEnter template (use {variable_name} for placeholders):")
                print("Press Enter twice when finished:")
                template_lines = []
                while True:
                    line = input()
                    if line == "" and template_lines and template_lines[-1] == "":
                        template_lines.pop()
                        break
                    template_lines.append(line)
                template = "\n".join(template_lines)
                
                # Collect example data
                print("\nEnter example data (key:value pairs, 'done' when finished):")
                example_data = {}
                while True:
                    entry = input("> ").strip()
                    if entry.lower() == 'done':
                        break
                    if ':' in entry:
                        key, value = entry.split(':', 1)
                        example_data[key.strip()] = value.strip()
                
                best_for_input = input("Best for (comma-separated): ").strip()
                best_for = [item.strip() for item in best_for_input.split(',') if item.strip()]
                
                pattern = library.create_custom_pattern(pattern_id, name, description, 
                                                       template, example_data, best_for)
                print(f"\n✅ Custom pattern created: {pattern['name']}")
            
            elif choice == "4":
                filename = input("Save filename (default: prompt_patterns.json): ").strip() or "prompt_patterns.json"
                library.save_patterns(filename)
            
            elif choice == "5":
                filename = input("Load filename (default: prompt_patterns.json): ").strip() or "prompt_patterns.json"
                library.load_patterns(filename)
            
            else:
                print("❌ Invalid choice")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_pattern_lab()