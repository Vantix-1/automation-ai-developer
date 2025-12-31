"""
Advanced Content Summarizer - Days 6-8
Multi-document, multi-style summarization with advanced features
"""

import os
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import PyPDF2
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

load_dotenv()

class AdvancedContentSummarizer:
    """Advanced summarizer with multiple styles and document support"""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.encoder = tiktoken.encoding_for_model(model)
        
        # Summary styles
        self.styles = {
            "concise": {
                "name": "Concise Summary",
                "instruction": "Provide a concise summary in 2-3 sentences.",
                "max_tokens": 150
            },
            "detailed": {
                "name": "Detailed Summary",
                "instruction": "Provide a comprehensive summary covering all main points.",
                "max_tokens": 300
            },
            "bullet": {
                "name": "Bullet Points",
                "instruction": "Provide key points as bullet points.",
                "max_tokens": 250
            },
            "tldr": {
                "name": "TL;DR",
                "instruction": "Provide a very brief TL;DR summary.",
                "max_tokens": 100
            },
            "executive": {
                "name": "Executive Summary",
                "instruction": "Provide an executive summary suitable for business leaders.",
                "max_tokens": 200
            },
            "qna": {
                "name": "Q&A Format",
                "instruction": "Summarize by answering: What? Why? How? So what?",
                "max_tokens": 250
            }
        }
    
    def count_tokens(self, text: str) -> int:
        """Count tokens accurately"""
        return len(self.encoder.encode(text))
    
    def chunk_text(self, text: str, max_tokens: int = 2000) -> List[str]:
        """Split text into chunks based on token limit"""
        tokens = self.encoder.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.encoder.decode(chunk_tokens)
            chunks.append(chunk_text)
        
        return chunks
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {e}")
    
    def extract_text_from_url(self, url: str) -> str:
        """Extract text from webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text[:10000]  # Limit to first 10k chars
            
        except Exception as e:
            raise Exception(f"Error fetching URL: {e}")
    
    def summarize_chunk(self, text: str, style: str, custom_instruction: Optional[str] = None) -> str:
        """Summarize a single text chunk"""
        if style not in self.styles:
            style = "concise"
        
        style_info = self.styles[style]
        instruction = custom_instruction if custom_instruction else style_info["instruction"]
        
        prompt = f"""
        {instruction}
        
        Text to summarize:
        {text}
        
        Please provide a clear, well-structured summary.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert summarizer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=style_info["max_tokens"]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error summarizing chunk: {e}"
    
    def summarize_long_text(self, text: str, style: str = "concise", custom_instruction: Optional[str] = None) -> Dict[str, Any]:
        """Summarize long text using chunking strategy"""
        print(f"📄 Processing text ({len(text):,} characters)...")
        
        # Chunk the text
        chunks = self.chunk_text(text)
        print(f"📊 Split into {len(chunks)} chunk(s)")
        
        summaries = []
        
        # Process each chunk with progress bar
        for i, chunk in enumerate(tqdm(chunks, desc="Summarizing chunks"), 1):
            summary = self.summarize_chunk(chunk, style, custom_instruction)
            summaries.append(f"Chunk {i} Summary:\n{summary}\n")
        
        # Combine summaries
        combined_summaries = "\n".join(summaries)
        
        # Create final summary of summaries
        if len(chunks) > 1:
            print("🔄 Creating final summary from chunk summaries...")
            final_prompt = f"""
            Combine these individual chunk summaries into one coherent final summary.
            
            Chunk Summaries:
            {combined_summaries}
            
            Provide a final comprehensive summary that captures all key points.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert at combining summaries."},
                        {"role": "user", "content": final_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400
                )
                
                final_summary = response.choices[0].message.content.strip()
                
            except Exception as e:
                final_summary = f"Error creating final summary: {e}"
        else:
            final_summary = summaries[0].replace("Chunk 1 Summary:\n", "")
        
        # Calculate statistics
        original_tokens = self.count_tokens(text)
        summary_tokens = self.count_tokens(final_summary)
        compression_ratio = (1 - (summary_tokens / original_tokens)) * 100 if original_tokens > 0 else 0
        
        return {
            "summary": final_summary,
            "original_length": len(text),
            "original_tokens": original_tokens,
            "summary_length": len(final_summary),
            "summary_tokens": summary_tokens,
            "compression_ratio": round(compression_ratio, 1),
            "chunks_processed": len(chunks),
            "style": style
        }
    
    def batch_summarize(self, texts: List[str], style: str = "concise") -> List[Dict[str, Any]]:
        """Summarize multiple texts"""
        results = []
        
        for i, text in enumerate(tqdm(texts, desc="Processing texts"), 1):
            print(f"\n📝 Processing text {i}/{len(texts)}...")
            result = self.summarize_long_text(text, style)
            results.append(result)
        
        return results
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text for key metrics"""
        tokens = self.count_tokens(text)
        words = len(text.split())
        sentences = len(re.split(r'[.!?]+', text))
        paragraphs = len(text.split('\n\n'))
        
        # Estimate reading time (200 words per minute)
        reading_time = words / 200
        
        # Analyze sentiment and key topics (simplified)
        analysis_prompt = f"""
        Analyze this text and provide:
        1. Overall sentiment (positive, negative, neutral)
        2. 3-5 key topics/themes
        3. Writing style (formal, informal, technical, etc.)
        
        Text: {text[:2000]}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a text analysis expert."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            analysis = response.choices[0].message.content
            
        except Exception as e:
            analysis = f"Analysis error: {e}"
        
        return {
            "tokens": tokens,
            "words": words,
            "sentences": sentences,
            "paragraphs": paragraphs,
            "reading_time_minutes": round(reading_time, 1),
            "analysis": analysis
        }
    
    def export_summary(self, summary_data: Dict[str, Any], format: str = "markdown") -> str:
        """Export summary in different formats"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if format == "markdown":
            content = f"""# Summary Report
**Generated:** {timestamp}
**Style:** {summary_data['style']}

## Statistics
- Original length: {summary_data['original_length']:,} characters
- Original tokens: {summary_data['original_tokens']:,}
- Summary length: {summary_data['summary_length']:,} characters
- Summary tokens: {summary_data['summary_tokens']:,}
- Compression: {summary_data['compression_ratio']}%
- Chunks processed: {summary_data['chunks_processed']}

## Summary
{summary_data['summary']}
"""
        elif format == "json":
            import json
            content = json.dumps(summary_data, indent=2, ensure_ascii=False)
        else:  # text
            content = f"""SUMMARY REPORT
Generated: {timestamp}
Style: {summary_data['style']}

STATISTICS
Original length: {summary_data['original_length']:,} characters
Original tokens: {summary_data['original_tokens']:,}
Summary length: {summary_data['summary_length']:,} characters
Summary tokens: {summary_data['summary_tokens']:,}
Compression: {summary_data['compression_ratio']}%
Chunks processed: {summary_data['chunks_processed']}

SUMMARY
{summary_data['summary']}
"""
        
        return content

def interactive_summarizer():
    """Interactive summarizer demo"""
    print("\n" + "="*70)
    print("🚀 Advanced Content Summarizer - Days 6-8")
    print("="*70)
    
    try:
        summarizer = AdvancedContentSummarizer()
        
        print("\n📝 Available Summary Styles:")
        for i, (style_key, style_info) in enumerate(summarizer.styles.items(), 1):
            print(f"  {i}. {style_info['name']} - {style_info['instruction'][:50]}...")
        
        while True:
            print("\n" + "="*50)
            print("📋 Choose Input Method:")
            print("1. Enter text directly")
            print("2. Load from file (.txt)")
            print("3. Load from PDF")
            print("4. Fetch from URL")
            print("5. Exit")
            print("="*50)
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "5":
                print("👋 Goodbye!")
                break
            
            text = ""
            
            if choice == "1":
                print("\n📝 Enter/Paste your text (press Enter twice to finish):")
                lines = []
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        lines.pop()
                        break
                    lines.append(line)
                text = "\n".join(lines)
                
            elif choice == "2":
                filepath = input("Enter text file path: ").strip()
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    print(f"✅ Loaded {len(text):,} characters from file")
                except Exception as e:
                    print(f"❌ Error reading file: {e}")
                    continue
                    
            elif choice == "3":
                filepath = input("Enter PDF file path: ").strip()
                try:
                    text = summarizer.extract_text_from_pdf(filepath)
                    print(f"✅ Extracted {len(text):,} characters from PDF")
                except Exception as e:
                    print(f"❌ Error reading PDF: {e}")
                    continue
                    
            elif choice == "4":
                url = input("Enter URL: ").strip()
                try:
                    text = summarizer.extract_text_from_url(url)
                    print(f"✅ Fetched {len(text):,} characters from URL")
                except Exception as e:
                    print(f"❌ Error fetching URL: {e}")
                    continue
            
            else:
                print("❌ Invalid choice")
                continue
            
            if not text or len(text.strip()) < 10:
                print("❌ Text too short or empty")
                continue
            
            # Analyze text first
            print("\n🔍 Analyzing text...")
            analysis = summarizer.analyze_text(text)
            print(f"📊 Text Analysis:")
            print(f"   Words: {analysis['words']:,}")
            print(f"   Tokens: {analysis['tokens']:,}")
            print(f"   Reading time: {analysis['reading_time_minutes']} minutes")
            print(f"   Analysis: {analysis['analysis'][:100]}...")
            
            # Select style
            print("\n🎯 Select summary style:")
            styles = list(summarizer.styles.keys())
            for i, style_key in enumerate(styles, 1):
                print(f"  {i}. {summarizer.styles[style_key]['name']}")
            
            style_choice = input("\nEnter style number or name: ").strip()
            if style_choice.isdigit() and 1 <= int(style_choice) <= len(styles):
                style = styles[int(style_choice) - 1]
            elif style_choice in styles:
                style = style_choice
            else:
                style = "concise"
            
            # Optional custom instruction
            custom = input("\nOptional: Enter custom instructions (or press Enter to skip): ").strip()
            custom_instruction = custom if custom else None
            
            # Summarize
            print(f"\n📝 Creating {summarizer.styles[style]['name']}...")
            result = summarizer.summarize_long_text(text, style, custom_instruction)
            
            # Display results
            print("\n" + "="*70)
            print("✅ SUMMARY COMPLETE")
            print("="*70)
            print(f"\n📊 Statistics:")
            print(f"   Original: {result['original_length']:,} chars, {result['original_tokens']:,} tokens")
            print(f"   Summary: {result['summary_length']:,} chars, {result['summary_tokens']:,} tokens")
            print(f"   Compression: {result['compression_ratio']}%")
            print(f"   Style: {result['style']}")
            
            print(f"\n📝 Summary:")
            print("-"*50)
            print(result['summary'])
            print("-"*50)
            
            # Export option
            export = input("\n💾 Export summary? (y/n): ").lower()
            if export == 'y':
                format_choice = input("Format (markdown/json/text): ").strip().lower() or "markdown"
                export_content = summarizer.export_summary(result, format_choice)
                
                filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_choice if format_choice != 'markdown' else 'md'}"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(export_content)
                print(f"✅ Summary saved to {filename}")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_summarizer()