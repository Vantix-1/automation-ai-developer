"""
Content Summarizer - Text summarization techniques
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class ContentSummarizer:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def summarize(self, text: str, style: str = "concise") -> str:
        style_prompts = {
            "concise": "Provide a concise summary in 2-3 sentences.",
            "detailed": "Provide a detailed summary covering main points.",
            "bullet": "Provide a summary with bullet points.",
            "tldr": "Provide a TL;DR summary."
        }
        
        if style not in style_prompts:
            style = "concise"
        
        prompt = f"""
        Please summarize the following text.
        {style_prompts[style]}
        
        Text to summarize:
        {text[:3000]}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error summarizing: {e}"

if __name__ == "__main__":
    summarizer = ContentSummarizer()
    sample_text = "Artificial intelligence (AI) is intelligence demonstrated by machines..."
    print("📝 Sample Summary:")
    print(summarizer.summarize(sample_text, "concise"))