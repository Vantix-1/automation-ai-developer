"""
Prompt Templates - Days 11-13
Advanced prompt template usage in LangChain
"""

import os
import json
from typing import Dict, List, Any, Optional
from colorama import Fore, Style, init
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate, 
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate
)
from langchain.prompts.example_selector import (
    LengthBasedExampleSelector,
    SemanticSimilarityExampleSelector
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseOutputParser
from langchain.callbacks import get_openai_callback

from dotenv import load_dotenv

# Initialize colorama
init(autoreset=True)

load_dotenv()

class CustomOutputParser(BaseOutputParser):
    """Custom output parser for structured responses"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse LLM output into structured format"""
        try:
            # Try to parse as JSON
            if "```json" in text:
                # Extract JSON from code block
                start = text.find("```json") + 7
                end = text.find("```", start)
                json_str = text[start:end].strip()
            elif "{" in text and "}" in text:
                # Try to extract JSON object
                start = text.find("{")
                end = text.rfind("}") + 1
                json_str = text[start:end]
            else:
                # Fallback to simple parsing
                return {"raw_output": text.strip()}
            
            return json.loads(json_str)
        except json.JSONDecodeError:
            return {"raw_output": text.strip(), "error": "Failed to parse JSON"}
    
    def get_format_instructions(self) -> str:
        return """Return your response as a valid JSON object. 
        If you need to include a JSON object in your response, use ```json code blocks."""

class AdvancedPromptTemplates:
    """Advanced prompt template examples and utilities"""
    
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key
        )
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            api_key=api_key
        )
        
        print(Fore.CYAN + "📝 Advanced Prompt Templates Initialized")
    
    def basic_templates(self):
        """Demonstrate basic prompt templates"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "📋 Basic Prompt Templates")
        print(Fore.CYAN + "="*50)
        
        # Example 1: Simple template
        print(Fore.YELLOW + "\n1. Simple String Template:")
        template1 = "Tell me a {adjective} story about {topic}."
        prompt1 = PromptTemplate.from_template(template1)
        formatted1 = prompt1.format(adjective="funny", topic="robots")
        print(Fore.WHITE + f"Template: {template1}")
        print(Fore.WHITE + f"Formatted: {formatted1}")
        
        # Example 2: Template with multiple variables
        print(Fore.YELLOW + "\n2. Multi-variable Template:")
        template2 = """As a {role}, help me with this task:
        
        Task: {task}
        Constraints: {constraints}
        
        Solution:"""
        prompt2 = PromptTemplate(
            input_variables=["role", "task", "constraints"],
            template=template2
        )
        formatted2 = prompt2.format(
            role="software architect",
            task="design a scalable web application",
            constraints="must handle 1 million users, be cost-effective"
        )
        print(Fore.WHITE + f"\nFormatted:\n{formatted2}")
        
        # Example 3: Using the template with LLM
        print(Fore.YELLOW + "\n3. Using Template with LLM:")
        with get_openai_callback() as cb:
            chain = prompt2 | self.llm
            result = chain.invoke({
                "role": "fitness trainer",
                "task": "create a workout plan",
                "constraints": "for beginners, 30 minutes daily, no equipment"
            })
            print(Fore.WHITE + f"LLM Response: {result.content[:150]}...")
            print(Fore.CYAN + f"Cost: ${cb.total_cost:.6f}")
    
    def chat_prompt_templates(self):
        """Demonstrate chat prompt templates"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "💬 Chat Prompt Templates")
        print(Fore.CYAN + "="*50)
        
        # Example 1: Simple chat template
        print(Fore.YELLOW + "\n1. Simple Chat Template:")
        chat_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful {assistant_type}."),
            ("human", "Hello! I need help with {topic}."),
            ("ai", "I'd be happy to help you with {topic}. What specifically do you need?"),
            ("human", "{question}")
        ])
        
        messages = chat_template.format_messages(
            assistant_type="math tutor",
            topic="algebra",
            question="How do I solve quadratic equations?"
        )
        
        print(Fore.WHITE + "\nFormatted Messages:")
        for msg in messages:
            print(Fore.CYAN + f"  {msg.type}: {msg.content}")
        
        # Example 2: Using message prompt templates
        print(Fore.YELLOW + "\n2. Message Prompt Templates:")
        
        system_template = SystemMessagePromptTemplate.from_template(
            "You are a {style} writer. Write in the style of {author}."
        )
        
        human_template = HumanMessagePromptTemplate.from_template(
            "Write a {genre} story about {topic}."
        )
        
        chat_prompt = ChatPromptTemplate.from_messages([
            system_template,
            human_template
        ])
        
        messages = chat_prompt.format_messages(
            style="creative",
            author="Edgar Allan Poe",
            genre="mystery",
            topic="a haunted library"
        )
        
        print(Fore.WHITE + "\nFormatted Messages:")
        for msg in messages:
            print(Fore.CYAN + f"  {msg.type}: {msg.content}")
        
        # Example 3: With placeholder for conversation history
        print(Fore.YELLOW + "\n3. Template with Conversation History:")
        
        prompt_with_history = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a helpful assistant. Here's our conversation history:"
            ),
            MessagesPlaceholder(variable_name="conversation_history"),
            HumanMessagePromptTemplate.from_template("{user_input}")
        ])
        
        # Simulate conversation history
        conversation_history = [
            ("human", "My name is Alex."),
            ("ai", "Nice to meet you, Alex!"),
            ("human", "I'm learning Python.")
        ]
        
        messages = prompt_with_history.format_messages(
            conversation_history=conversation_history,
            user_input="What should I learn first?"
        )
        
        print(Fore.WHITE + "\nWith Conversation History:")
        for msg in messages:
            print(Fore.CYAN + f"  {msg.type}: {msg.content[:80]}...")
    
    def few_shot_templates(self):
        """Demonstrate few-shot prompting"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "🎯 Few-Shot Prompt Templates")
        print(Fore.CYAN + "="*50)
        
        # Define examples
        examples = [
            {
                "input": "The food was delicious and the service was excellent!",
                "output": "positive"
            },
            {
                "input": "The product broke after one day of use.",
                "output": "negative"
            },
            {
                "input": "The package arrived on time as expected.",
                "output": "neutral"
            }
        ]
        
        # Example 1: Basic few-shot template
        print(Fore.YELLOW + "\n1. Basic Few-Shot Template:")
        
        example_template = """
        Input: {input}
        Output: {output}
        """
        
        example_prompt = PromptTemplate(
            input_variables=["input", "output"],
            template=example_template
        )
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Classify the sentiment of these text inputs:",
            suffix="Input: {input}\nOutput:",
            input_variables=["input"],
            example_separator="\n---\n"
        )
        
        test_input = "I love this product! It works perfectly."
        formatted = few_shot_prompt.format(input=test_input)
        
        print(Fore.WHITE + "\nFew-Shot Prompt:")
        print(Fore.WHITE + formatted[:300] + "...")
        
        # Test with LLM
        with get_openai_callback() as cb:
            chain = few_shot_prompt | self.llm
            result = chain.invoke({"input": "The movie was boring and too long."})
            print(Fore.GREEN + f"\nLLM Classification: {result.content}")
            print(Fore.CYAN + f"Cost: ${cb.total_cost:.6f}")
        
        # Example 2: Few-shot chat template
        print(Fore.YELLOW + "\n2. Few-Shot Chat Template:")
        
        chat_examples = [
            {
                "input": "How do I bake a cake?",
                "output": "I can help you bake a cake! First, gather your ingredients..."
            },
            {
                "input": "What's the weather like?",
                "output": "I don't have real-time weather data, but I can help you find it!"
            }
        ]
        
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}"),
            ("ai", "{output}")
        ])
        
        few_shot_chat_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=chat_examples,
        )
        
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            few_shot_chat_prompt,
            ("human", "{input}")
        ])
        
        messages = final_prompt.format_messages(
            input="How do I change a tire?"
        )
        
        print(Fore.WHITE + "\nFew-Shot Chat Prompt:")
        for msg in messages:
            print(Fore.CYAN + f"  {msg.type}: {msg.content[:80]}...")
    
    def dynamic_example_selection(self):
        """Demonstrate dynamic example selection"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "🎯 Dynamic Example Selection")
        print(Fore.CYAN + "="*50)
        
        # Create example database
        examples = [
            {"query": "How do I make pasta?", "answer": "Boil water, add pasta, cook for 8-10 minutes."},
            {"query": "What's the recipe for pancakes?", "answer": "Mix flour, eggs, milk, cook on griddle."},
            {"query": "How to bake bread?", "answer": "Mix flour, yeast, water, knead, let rise, bake."},
            {"query": "Making a salad?", "answer": "Chop vegetables, add dressing, mix well."},
            {"query": "How to cook rice?", "answer": "Rinse rice, add water (2:1 ratio), simmer 15-20 min."},
            {"query": "Scrambled eggs recipe?", "answer": "Beat eggs, cook in butter, stir constantly."},
            {"query": "How to make soup?", "answer": "Sauté vegetables, add broth, simmer."},
            {"query": "Baking cookies?", "answer": "Mix dough, drop on sheet, bake 10-12 minutes."},
        ]
        
        # Example 1: Length-based selection
        print(Fore.YELLOW + "\n1. Length-Based Example Selector:")
        
        example_template = """
        Query: {query}
        Answer: {answer}
        """
        
        example_prompt = PromptTemplate(
            input_variables=["query", "answer"],
            template=example_template
        )
        
        length_selector = LengthBasedExampleSelector(
            examples=examples,
            example_prompt=example_prompt,
            max_length=200  # Maximum length for formatted examples
        )
        
        dynamic_prompt = FewShotPromptTemplate(
            example_selector=length_selector,
            example_prompt=example_prompt,
            prefix="Answer cooking questions:",
            suffix="Query: {input}\nAnswer:",
            input_variables=["input"],
            example_separator="\n---\n"
        )
        
        # Test with different length inputs
        test_queries = [
            "pizza",
            "How do I make homemade pizza with fresh ingredients?"
        ]
        
        for query in test_queries:
            formatted = dynamic_prompt.format(input=query)
            print(Fore.WHITE + f"\nQuery: '{query}'")
            print(Fore.WHITE + f"Prompt length: {len(formatted)} characters")
            print(Fore.WHITE + f"Examples selected: {len(formatted.split('---')) - 1}")
        
        # Example 2: Semantic similarity selection
        print(Fore.YELLOW + "\n2. Semantic Similarity Selector:")
        
        # Create vector store for examples
        try:
            from langchain.vectorstores import FAISS
            
            texts = [ex["query"] for ex in examples]
            metadatas = [{"answer": ex["answer"]} for ex in examples]
            
            vectorstore = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            semantic_selector = SemanticSimilarityExampleSelector(
                vectorstore=vectorstore,
                k=2,  # Number of examples to select
                input_keys=["input"]
            )
            
            semantic_prompt = FewShotPromptTemplate(
                example_selector=semantic_selector,
                example_prompt=example_prompt,
                prefix="Answer cooking questions:",
                suffix="Query: {input}\nAnswer:",
                input_variables=["input"],
                example_separator="\n---\n"
            )
            
            test_query = "How to prepare spaghetti?"
            formatted = semantic_prompt.format(input=test_query)
            
            print(Fore.WHITE + f"\nQuery: '{test_query}'")
            print(Fore.WHITE + "\nSelected similar examples:")
            print(Fore.WHITE + formatted[:500] + "...")
            
        except ImportError:
            print(Fore.RED + "⚠️ FAISS not installed. Install with: pip install faiss-cpu")
    
    def template_partials(self):
        """Demonstrate partial templates"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "🔧 Template Partials")
        print(Fore.CYAN + "="*50)
        
        # Create a template with partial values
        print(Fore.YELLOW + "\n1. Partial Template Application:")
        
        template = """
        You are a {role} helping with {domain} tasks.
        
        User question: {question}
        
        Provide a detailed response considering:
        - User's level: {level}
        - Time constraint: {time_limit}
        
        Response:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["role", "domain", "question", "level", "time_limit"]
        )
        
        # Create partial template with some values pre-filled
        partial_prompt = prompt.partial(
            role="expert tutor",
            level="intermediate",
            time_limit="5 minutes"
        )
        
        print(Fore.WHITE + "\nPartial Template (role, level, time_limit pre-filled):")
        print(Fore.WHITE + f"Remaining variables: {partial_prompt.input_variables}")
        
        # Use the partial template
        formatted = partial_prompt.format(
            domain="Python programming",
            question="How do decorators work?"
        )
        
        print(Fore.WHITE + "\nFormatted Prompt:")
        print(Fore.WHITE + formatted)
        
        # Example 2: Using partials with different configurations
        print(Fore.YELLOW + "\n2. Multiple Partial Configurations:")
        
        configurations = {
            "beginner": {"level": "beginner", "detail": "simple"},
            "expert": {"level": "expert", "detail": "technical"},
            "executive": {"level": "executive", "detail": "high-level"}
        }
        
        base_template = """
        Explain {concept} to a {level} audience.
        Use {detail} language and focus on practical applications.
        
        Explanation:"""
        
        base_prompt = PromptTemplate(
            template=base_template,
            input_variables=["concept", "level", "detail"]
        )
        
        for config_name, config in configurations.items():
            partial = base_prompt.partial(**config)
            result = partial.format(concept="blockchain")
            print(Fore.WHITE + f"\n{config_name.upper()} explanation:")
            print(Fore.WHITE + f"{result[:100]}...")
    
    def custom_output_parsing(self):
        """Demonstrate custom output parsing"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "🎯 Custom Output Parsing")
        print(Fore.CYAN + "="*50)
        
        # Create parser
        parser = CustomOutputParser()
        
        # Example 1: Template with output instructions
        print(Fore.YELLOW + "\n1. Template with Output Instructions:")
        
        template_with_instructions = """
        Analyze this product review and extract:
        1. Overall sentiment (positive/negative/neutral)
        2. Key features mentioned
        3. Suggested improvements
        
        Review: {review}
        
        {format_instructions}
        """
        
        prompt = PromptTemplate(
            template=template_with_instructions,
            input_variables=["review"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        # Test with a review
        test_review = """
        The battery life is amazing - lasts all day! 
        However, the screen is too dim in sunlight and the camera quality could be better.
        Overall, it's a good phone for the price.
        """
        
        formatted_prompt = prompt.format(review=test_review)
        
        print(Fore.WHITE + "\nPrompt with Format Instructions:")
        print(Fore.WHITE + formatted_prompt[:300] + "...")
        
        # Get and parse response
        with get_openai_callback() as cb:
            chain = prompt | self.llm | parser
            result = chain.invoke({"review": test_review})
            
            print(Fore.GREEN + "\n✅ Parsed Output:")
            print(Fore.WHITE + json.dumps(result, indent=2))
            print(Fore.CYAN + f"\n💰 Cost: ${cb.total_cost:.6f}")
        
        # Example 2: Structured data extraction
        print(Fore.YELLOW + "\n2. Structured Data Extraction:")
        
        extraction_template = """
        Extract the following information from the text:
        
        Text: {text}
        
        Extract as JSON with these keys:
        - names: list of person names mentioned
        - dates: list of dates mentioned  
        - locations: list of locations mentioned
        - key_points: list of main points
        
        {format_instructions}
        """
        
        extraction_prompt = PromptTemplate(
            template=extraction_template,
            input_variables=["text"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        test_text = """
        John Smith visited London on January 15, 2024. He met with Sarah Johnson 
        to discuss the new project. They plan to meet again in Paris on March 20th.
        Key decisions: increase budget, hire two developers, launch in Q2.
        """
        
        with get_openai_callback() as cb:
            chain = extraction_prompt | self.llm | parser
            result = chain.invoke({"text": test_text})
            
            print(Fore.GREEN + "\n✅ Extracted Structured Data:")
            print(Fore.WHITE + json.dumps(result, indent=2))
            print(Fore.CYAN + f"💰 Cost: ${cb.total_cost:.6f}")
    
    def template_validation(self):
        """Demonstrate template validation and debugging"""
        print(Fore.CYAN + "\n" + "="*50)
        print(Fore.CYAN + "🔍 Template Validation")
        print(Fore.CYAN + "="*50)
        
        # Example 1: Validating template variables
        print(Fore.YELLOW + "\n1. Template Variable Validation:")
        
        try:
            # Template with missing variable
            template = "Hello {name}, welcome to {city}!"
            prompt = PromptTemplate(
                template=template,
                input_variables=["name"]  # Missing 'city'
            )
            # This would raise an error when used
            print(Fore.RED + "❌ This would fail (missing variable)")
        except Exception as e:
            print(Fore.WHITE + f"Validation error: {e}")
        
        # Correct template
        correct_template = "Hello {name}, welcome to {city}!"
        correct_prompt = PromptTemplate(
            template=correct_template,
            input_variables=["name", "city"],
            validate_template=True
        )
        
        print(Fore.GREEN + "\n✅ Correct template validated successfully")
        print(Fore.WHITE + f"Variables: {correct_prompt.input_variables}")
        
        # Example 2: Debugging template issues
        print(Fore.YELLOW + "\n2. Template Debugging:")
        
        complex_template = """
        Analyze this {document_type}:
        
        {content}
        
        Provide analysis in this format:
        Summary: {summary_length} sentences
        Key points: {num_points} bullet points
        Action items: {num_actions} items
        
        Analysis:"""
        
        # Create prompt with all variables
        prompt = PromptTemplate(
            template=complex_template,
            input_variables=[
                "document_type", 
                "content", 
                "summary_length", 
                "num_points", 
                "num_actions"
            ]
        )
        
        # Test formatting
        test_data = {
            "document_type": "business report",
            "content": "Sales increased by 15% this quarter...",
            "summary_length": "3",
            "num_points": "5",
            "num_actions": "3"
        }
        
        try:
            formatted = prompt.format(**test_data)
            print(Fore.GREEN + "\n✅ Template formatted successfully")
            print(Fore.WHITE + f"\nFormatted length: {len(formatted)} characters")
            print(Fore.WHITE + f"\nPreview:\n{formatted[:200]}...")
        except KeyError as e:
            print(Fore.RED + f"\n❌ Missing variable: {e}")
        except Exception as e:
            print(Fore.RED + f"\n❌ Formatting error: {e}")
        
        # Example 3: Template serialization
        print(Fore.YELLOW + "\n3. Template Serialization:")
        
        # Save template to dictionary
        template_dict = prompt.dict()
        print(Fore.WHITE + "\nTemplate as dictionary:")
        print(Fore.WHITE + json.dumps(template_dict, indent=2))
        
        # Load from dictionary
        from langchain.prompts import load_prompt
        # Note: In practice, you'd save to file and load
        
        print(Fore.GREEN + "\n✅ Templates can be serialized and loaded")
    
    def interactive_template_lab(self):
        """Interactive template laboratory"""
        print(Fore.CYAN + "\n" + "="*70)
        print(Fore.CYAN + "🔬 Interactive Template Laboratory")
        print(Fore.CYAN + "="*70)
        
        while True:
            print(Fore.YELLOW + "\n" + "━" * 50)
            print(Fore.YELLOW + "📚 Template Examples:")
            print(Fore.YELLOW + "1. Basic Templates")
            print(Fore.YELLOW + "2. Chat Prompt Templates")
            print(Fore.YELLOW + "3. Few-Shot Templates")
            print(Fore.YELLOW + "4. Dynamic Example Selection")
            print(Fore.YELLOW + "5. Template Partials")
            print(Fore.YELLOW + "6. Custom Output Parsing")
            print(Fore.YELLOW + "7. Template Validation")
            print(Fore.YELLOW + "8. Exit")
            print(Fore.YELLOW + "━" * 50)
            
            choice = input(Fore.WHITE + "\nSelect example (1-8): ").strip()
            
            if choice == "8":
                print(Fore.YELLOW + "\n👋 Goodbye!")
                break
            
            if choice == "1":
                self.basic_templates()
            elif choice == "2":
                self.chat_prompt_templates()
            elif choice == "3":
                self.few_shot_templates()
            elif choice == "4":
                self.dynamic_example_selection()
            elif choice == "5":
                self.template_partials()
            elif choice == "6":
                self.custom_output_parsing()
            elif choice == "7":
                self.template_validation()
            else:
                print(Fore.RED + "❌ Invalid choice")
            
            input(Fore.YELLOW + "\nPress Enter to continue...")

def main():
    """Main function"""
    try:
        templates = AdvancedPromptTemplates()
        templates.interactive_template_lab()
        
    except Exception as e:
        print(Fore.RED + f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()