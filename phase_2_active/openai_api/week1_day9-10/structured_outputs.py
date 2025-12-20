"""
Structured Outputs - Days 9-10
JSON Mode and Data Extraction with OpenAI
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Type, TypeVar
from enum import Enum
from datetime import datetime
from dataclasses import dataclass, asdict, field
from pydantic import BaseModel, Field, validator
from openai import OpenAI
from dotenv import load_dotenv
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

load_dotenv()

# Type variable for generic Pydantic models
T = TypeVar('T', bound=BaseModel)

class StructuredOutputs:
    """Handles structured outputs using OpenAI's JSON mode"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def extract_with_schema(self, text: str, schema: Type[T], 
                           instruction: str = "Extract information") -> Optional[T]:
        """Extract structured data from text using Pydantic schema"""
        print(Fore.CYAN + f"\n📝 Extracting data with schema: {schema.__name__}")
        
        # Get schema description
        schema_description = self._get_schema_description(schema)
        
        prompt = f"""
        {instruction}
        
        Text to analyze:
        {text[:3000]}  # Limit text length
        
        Extract the information and return it as valid JSON that matches this schema:
        {schema_description}
        
        Return ONLY the JSON object, no other text.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data extraction expert. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=1000,
                response_format={"type": "json_object"}  # Force JSON mode
            )
            
            # Parse JSON response
            json_str = response.choices[0].message.content
            data = json.loads(json_str)
            
            # Convert to Pydantic model
            return schema(**data)
            
        except json.JSONDecodeError as e:
            print(Fore.RED + f"❌ Failed to parse JSON: {e}")
            print(Fore.YELLOW + f"Raw response: {response.choices[0].message.content[:200]}...")
            return None
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
            return None
    
    def _get_schema_description(self, schema: Type[BaseModel]) -> str:
        """Generate human-readable description of a Pydantic schema"""
        description = f"Schema: {schema.__name__}\n"
        description += "Fields:\n"
        
        for field_name, field_info in schema.model_fields.items():
            field_type = str(field_info.annotation).split("'")[1] if "'" in str(field_info.annotation) else str(field_info.annotation)
            description += f"  - {field_name} ({field_type}): {field_info.description or 'No description'}\n"
        
        return description
    
    def batch_extract(self, texts: List[str], schema: Type[T]) -> List[Optional[T]]:
        """Extract data from multiple texts"""
        results = []
        
        for i, text in enumerate(texts, 1):
            print(Fore.CYAN + f"\n📄 Processing text {i}/{len(texts)}...")
            result = self.extract_with_schema(text, schema)
            results.append(result)
        
        return results
    
    def validate_and_fix(self, data: Dict[str, Any], schema: Type[T]) -> Optional[T]:
        """Validate data against schema and attempt to fix issues"""
        try:
            # Try to create model directly
            return schema(**data)
        except Exception as e:
            print(Fore.YELLOW + f"⚠️ Validation failed: {e}")
            print(Fore.CYAN + "🛠️ Attempting to fix data...")
            
            # Use AI to fix the data
            fixed = self._fix_data_with_ai(data, schema, str(e))
            if fixed:
                try:
                    return schema(**fixed)
                except Exception as e2:
                    print(Fore.RED + f"❌ Still invalid after fixing: {e2}")
                    return None
            return None
    
    def _fix_data_with_ai(self, data: Dict[str, Any], schema: Type[T], error: str) -> Optional[Dict[str, Any]]:
        """Use AI to fix invalid data"""
        schema_description = self._get_schema_description(schema)
        
        prompt = f"""
        Fix this invalid data to match the schema.
        
        Schema requirements:
        {schema_description}
        
        Current (invalid) data:
        {json.dumps(data, indent=2)}
        
        Validation error:
        {error}
        
        Return ONLY the fixed JSON object.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data validation expert. Fix invalid data to match schemas."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            fixed_json = response.choices[0].message.content
            return json.loads(fixed_json)
            
        except Exception as e:
            print(Fore.RED + f"❌ Failed to fix data: {e}")
            return None
    
    def generate_from_template(self, template: str, schema: Type[T], 
                              variables: Dict[str, Any] = None) -> Optional[T]:
        """Generate structured data from a template"""
        print(Fore.CYAN + f"\n🔄 Generating data from template with schema: {schema.__name__}")
        
        schema_description = self._get_schema_description(schema)
        
        variables_str = json.dumps(variables, indent=2) if variables else "{}"
        
        prompt = f"""
        Fill this template with appropriate data that matches the schema.
        
        Template:
        {template}
        
        Available variables (use as context):
        {variables_str}
        
        Schema to follow:
        {schema_description}
        
        Generate appropriate data and return as valid JSON.
        Return ONLY the JSON object.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data generation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Higher temperature for creativity
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            json_str = response.choices[0].message.content
            data = json.loads(json_str)
            
            return schema(**data)
            
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
            return None
    
    def transform_format(self, data: Dict[str, Any], from_schema: Type, 
                        to_schema: Type[T]) -> Optional[T]:
        """Transform data from one schema format to another"""
        print(Fore.CYAN + f"\n🔄 Transforming from {from_schema.__name__} to {to_schema.__name__}")
        
        from_description = self._get_schema_description(from_schema)
        to_description = self._get_schema_description(to_schema)
        
        prompt = f"""
        Transform this data from one format to another.
        
        Source data:
        {json.dumps(data, indent=2)}
        
        Source schema:
        {from_description}
        
        Target schema:
        {to_description}
        
        Transform the data appropriately and return as valid JSON.
        Return ONLY the JSON object.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data transformation expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            json_str = response.choices[0].message.content
            transformed = json.loads(json_str)
            
            return to_schema(**transformed)
            
        except Exception as e:
            print(Fore.RED + f"❌ Error: {e}")
            return None

# Example Pydantic models for demonstration
class ContactInfo(BaseModel):
    """Contact information"""
    name: str = Field(description="Full name of the person")
    email: str = Field(description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    company: Optional[str] = Field(None, description="Company name")
    position: Optional[str] = Field(None, description="Job position")
    
    @validator('email')
    def email_must_contain_at(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email address')
        return v

class Product(BaseModel):
    """Product information"""
    name: str = Field(description="Product name")
    category: str = Field(description="Product category")
    price: float = Field(description="Price in USD")
    description: str = Field(description="Product description")
    features: List[str] = Field(description="List of key features")
    in_stock: bool = Field(description="Availability status")
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

class MeetingSummary(BaseModel):
    """Meeting summary"""
    title: str = Field(description="Meeting title")
    date: str = Field(description="Meeting date")
    participants: List[str] = Field(description="List of participants")
    key_decisions: List[str] = Field(description="Key decisions made")
    action_items: List[Dict[str, str]] = Field(description="Action items with assignee and deadline")
    next_meeting: Optional[str] = Field(None, description="Date of next meeting")

class NewsArticle(BaseModel):
    """News article information"""
    headline: str = Field(description="Article headline")
    source: str = Field(description="News source")
    publish_date: str = Field(description="Publication date")
    author: Optional[str] = Field(None, description="Author name")
    summary: str = Field(description="Brief summary")
    key_points: List[str] = Field(description="Key points from article")
    sentiment: str = Field(description="Sentiment: positive, negative, or neutral")

def interactive_demo():
    """Interactive structured outputs demo"""
    print(Fore.CYAN + "\n" + "="*70)
    print(Fore.CYAN + "📊 Structured Outputs & JSON Mode Demo")
    print(Fore.CYAN + "="*70)
    
    try:
        extractor = StructuredOutputs()
        
        # Example texts for demonstration
        example_texts = {
            "contact": """John Smith from Acme Corp reached out. His email is john.smith@acmecorp.com 
            and phone is 555-123-4567. He's the Senior Developer there.""",
            
            "product": """The Quantum Laptop X1 is our flagship product in the electronics category. 
            Priced at $1299.99, it features a quantum processor, 32GB RAM, 2TB SSD, and 8K display. 
            Currently in stock. Perfect for developers and creatives.""",
            
            "meeting": """Weekly team meeting on 2024-12-10 attended by Alice, Bob, and Charlie. 
            Decided to prioritize the new API integration and delay the UI redesign. 
            Action items: Alice to research API options by Friday, Bob to update project timeline by Wednesday. 
            Next meeting scheduled for 2024-12-17.""",
            
            "news": """Tech Giant Announces Breakthrough in AI from The Daily Tech published today. 
            Written by Sarah Johnson. The company revealed a new AI model that's 50% more efficient. 
            Positive market reaction expected. Key points: energy efficiency improved, faster processing, 
            available Q2 2025."""
        }
        
        while True:
            print(Fore.YELLOW + "\n" + "━" * 50)
            print(Fore.YELLOW + "📋 Available Operations:")
            print(Fore.YELLOW + "1. Extract Contact Information")
            print(Fore.YELLOW + "2. Extract Product Details")
            print(Fore.YELLOW + "3. Summarize Meeting Notes")
            print(Fore.YELLOW + "4. Parse News Article")
            print(Fore.YELLOW + "5. Batch Extract Multiple Texts")
            print(Fore.YELLOW + "6. Validate & Fix Invalid Data")
            print(Fore.YELLOW + "7. Generate from Template")
            print(Fore.YELLOW + "8. Transform Between Formats")
            print(Fore.YELLOW + "9. Exit")
            print(Fore.YELLOW + "━" * 50)
            
            choice = input(Fore.WHITE + "\nSelect operation (1-9): ").strip()
            
            if choice == "9":
                print(Fore.YELLOW + "👋 Goodbye!")
                break
            
            if choice == "1":
                print(Fore.CYAN + "\n📇 Extracting contact information...")
                result = extractor.extract_with_schema(
                    example_texts["contact"], 
                    ContactInfo,
                    "Extract contact information from the text"
                )
                
                if result:
                    print(Fore.GREEN + "✅ Successfully extracted:")
                    print(json.dumps(result.dict(), indent=2))
                else:
                    print(Fore.RED + "❌ Failed to extract data")
            
            elif choice == "2":
                print(Fore.CYAN + "\n🛒 Extracting product details...")
                result = extractor.extract_with_schema(
                    example_texts["product"],
                    Product,
                    "Extract product information"
                )
                
                if result:
                    print(Fore.GREEN + "✅ Product extracted:")
                    print(json.dumps(result.dict(), indent=2))
            
            elif choice == "3":
                print(Fore.CYAN + "\n📅 Summarizing meeting notes...")
                result = extractor.extract_with_schema(
                    example_texts["meeting"],
                    MeetingSummary,
                    "Extract meeting summary"
                )
                
                if result:
                    print(Fore.GREEN + "✅ Meeting summary:")
                    print(json.dumps(result.dict(), indent=2))
            
            elif choice == "4":
                print(Fore.CYAN + "\n📰 Parsing news article...")
                result = extractor.extract_with_schema(
                    example_texts["news"],
                    NewsArticle,
                    "Extract news article information"
                )
                
                if result:
                    print(Fore.GREEN + "✅ News article parsed:")
                    print(json.dumps(result.dict(), indent=2))
            
            elif choice == "5":
                print(Fore.CYAN + "\n📚 Batch extracting from multiple texts...")
                texts = list(example_texts.values())
                results = extractor.batch_extract(texts, ContactInfo)
                
                print(Fore.GREEN + f"\n✅ Batch extraction complete. Results: {len(results)}")
                for i, result in enumerate(results, 1):
                    if result:
                        print(Fore.CYAN + f"\nText {i}: ✓ Valid")
                    else:
                        print(Fore.RED + f"\nText {i}: ✗ Failed")
            
            elif choice == "6":
                print(Fore.CYAN + "\n🛠️ Validating and fixing data...")
                
                # Create invalid data
                invalid_contact = {
                    "name": "Jane Doe",
                    "email": "invalid-email",  # Invalid email
                    "phone": "555-987-6543"
                }
                
                print(Fore.YELLOW + "Invalid data:")
                print(json.dumps(invalid_contact, indent=2))
                
                result = extractor.validate_and_fix(invalid_contact, ContactInfo)
                
                if result:
                    print(Fore.GREEN + "✅ Fixed data:")
                    print(json.dumps(result.dict(), indent=2))
                else:
                    print(Fore.RED + "❌ Could not fix data")
            
            elif choice == "7":
                print(Fore.CYAN + "\n🎨 Generating from template...")
                
                template = """Create a product description for a {category} called {name}. 
                It should cost around ${price} and have these features: {features}."""
                
                variables = {
                    "category": "smartwatch",
                    "name": "ChronoFit Pro",
                    "price": 299.99,
                    "features": ["heart rate monitoring", "GPS tracking", "7-day battery"]
                }
                
                result = extractor.generate_from_template(template, Product, variables)
                
                if result:
                    print(Fore.GREEN + "✅ Generated product:")
                    print(json.dumps(result.dict(), indent=2))
            
            elif choice == "8":
                print(Fore.CYAN + "\n🔄 Transforming between formats...")
                
                # Simple contact data
                simple_contact = {
                    "full_name": "Robert Johnson",
                    "email_address": "robert@example.com",
                    "phone_number": "555-111-2222"
                }
                
                # Define a simple source schema
                class SimpleContact(BaseModel):
                    full_name: str
                    email_address: str
                    phone_number: str
                
                print(Fore.YELLOW + "Source data (SimpleContact):")
                print(json.dumps(simple_contact, indent=2))
                
                result = extractor.transform_format(simple_contact, SimpleContact, ContactInfo)
                
                if result:
                    print(Fore.GREEN + "✅ Transformed to ContactInfo:")
                    print(json.dumps(result.dict(), indent=2))
            
            else:
                print(Fore.RED + "❌ Invalid choice")
            
            input(Fore.YELLOW + "\nPress Enter to continue...")
    
    except Exception as e:
        print(Fore.RED + f"\n❌ Error: {e}")

if __name__ == "__main__":
    interactive_demo()