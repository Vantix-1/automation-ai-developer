"""
Streaming Demo - Days 6-8
Real-time streaming with progress tracking and interactive features
"""

import os
import time
import threading
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
from tqdm import tqdm

load_dotenv()

class StreamingChatDemo:
    """Demonstrate advanced streaming features"""
    
    def __init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
        self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def stream_with_progress(self, prompt: str, max_tokens: int = 500):
        """Stream response with visual progress bar"""
        print(f"\n📤 Sending prompt ({self.count_tokens(prompt)} tokens)...")
        
        # Create progress bar
        pbar = tqdm(total=max_tokens, desc="Generating", unit="token", ncols=80)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True
        )
        
        full_response = ""
        tokens_received = 0
        
        print("\n🤖 AI Response (streaming):")
        print("-" * 50)
        
        try:
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content
                    
                    # Update progress based on token count
                    new_tokens = self.count_tokens(content)
                    tokens_received += new_tokens
                    pbar.update(new_tokens)
            
            pbar.close()
            print("\n" + "-" * 50)
            print(f"\n✅ Complete! Received {tokens_received} tokens")
            
            return full_response
            
        except KeyboardInterrupt:
            pbar.close()
            print("\n\n⏹️ Generation stopped by user")
            return full_response
        except Exception as e:
            pbar.close()
            print(f"\n❌ Error: {e}")
            return ""
    
    def stream_with_typing_effect(self, prompt: str, typing_speed: float = 0.01):
        """Simulate typing effect while streaming"""
        print(f"\n⌨️ Typing Effect Demo (speed: {typing_speed}s per character)")
        print("-" * 50)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            stream=True
        )
        
        full_response = ""
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                for char in content:
                    print(char, end="", flush=True)
                    time.sleep(typing_speed)
                    full_response += char
        
        print("\n" + "-" * 50)
        return full_response
    
    def multi_stream_demo(self, prompts: list):
        """Demonstrate multiple concurrent streams"""
        print("\n🔄 Multiple Stream Demo")
        print("=" * 50)
        
        results = []
        threads = []
        
        def process_prompt(prompt: str, index: int):
            print(f"\n📤 Stream {index + 1}: Starting...")
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                stream=True
            )
            
            result = f"\nStream {index + 1} Result:\n"
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    result += chunk.choices[0].delta.content
            
            results.append((index, result))
            print(f"✅ Stream {index + 1}: Complete")
        
        # Start threads
        for i, prompt in enumerate(prompts):
            thread = threading.Thread(target=process_prompt, args=(prompt, i))
            threads.append(thread)
            thread.start()
            time.sleep(0.5)  # Stagger starts
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Display results
        results.sort(key=lambda x: x[0])
        print("\n" + "=" * 50)
        print("📋 All Streams Complete:")
        for idx, result in results:
            print(f"\n📝 Stream {idx + 1}:")
            print(result[:200] + "..." if len(result) > 200 else result)
    
    def token_counter_stream(self, prompt: str):
        """Stream with real-time token counting"""
        print("\n🔢 Token Counter Stream")
        print("=" * 50)
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            stream=True
        )
        
        full_response = ""
        token_count = 0
        start_time = time.time()
        
        print("\nReal-time statistics:")
        print("-" * 30)
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
                
                # Update token count
                new_tokens = self.count_tokens(content)
                token_count += new_tokens
                
                # Update stats every few chunks
                if token_count % 5 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_second = token_count / elapsed if elapsed > 0 else 0
                    print(f"\rTokens: {token_count} | Speed: {tokens_per_second:.1f} tokens/sec", end="", flush=True)
        
        elapsed = time.time() - start_time
        tokens_per_second = token_count / elapsed if elapsed > 0 else 0
        
        print(f"\n\n📊 Final Statistics:")
        print(f"Total tokens: {token_count}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Speed: {tokens_per_second:.1f} tokens/second")
        
        return full_response

def interactive_streaming_demo():
    """Interactive streaming demonstration"""
    print("\n" + "="*70)
    print("🎬 Advanced Streaming Demo - Days 6-8")
    print("="*70)
    
    try:
        demo = StreamingChatDemo()
        
        while True:
            print("\n" + "="*50)
            print("📺 Streaming Demo Options:")
            print("1. Stream with Progress Bar")
            print("2. Typing Effect Simulation")
            print("3. Multiple Concurrent Streams")
            print("4. Real-time Token Counter")
            print("5. Exit")
            print("="*50)
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "5":
                print("👋 Goodbye!")
                break
            
            if choice == "1":
                prompt = input("\nEnter prompt: ").strip()
                if prompt:
                    demo.stream_with_progress(prompt)
            
            elif choice == "2":
                prompt = input("\nEnter prompt: ").strip()
                if prompt:
                    speed = float(input("Typing speed (seconds per character, default 0.01): ") or "0.01")
                    demo.stream_with_typing_effect(prompt, speed)
            
            elif choice == "3":
                print("\nEnter multiple prompts (enter 'done' when finished):")
                prompts = []
                while True:
                    prompt = input(f"Prompt {len(prompts) + 1}: ").strip()
                    if prompt.lower() == 'done':
                        break
                    if prompt:
                        prompts.append(prompt)
                
                if prompts:
                    demo.multi_stream_demo(prompts)
            
            elif choice == "4":
                prompt = input("\nEnter prompt: ").strip()
                if prompt:
                    demo.token_counter_stream(prompt)
            
            else:
                print("❌ Invalid choice")
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_streaming_demo()