"""
Prompt Lab - Experiment with prompt engineering techniques
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class PromptLab:
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self.client = OpenAI(api_key=api_key)
    
    def test_prompt_variations(self, base_prompt: str, variations: list):
        print(f"\n🧪 Testing {len(variations)} prompt variations")
        
        for i, variation in enumerate(variations, 1):
            print(f"\nVariation {i}: {variation[:80]}...")
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"{base_prompt} {variation}"}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                
                print(f"Response: {response.choices[0].message.content[:100]}...")
                
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    lab = PromptLab()
    base_prompt = "Explain the concept of"
    variations = [
        "machine learning to a beginner",
        "machine learning to a data scientist",
        "machine learning in one sentence"
    ]
    lab.test_prompt_variations(base_prompt, variations)